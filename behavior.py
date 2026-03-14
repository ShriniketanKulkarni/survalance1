# ─── behavior.py ──────────────────────────────────────────────────────────────
"""
Suspicious activity engine — 3 core rules + weapon detection.

Rules:
  1. Loitering     — person stationary for > LOITER_SECONDS
  2. Abandoned Bag — bag unattended for > ABANDON_SECONDS
  3. Zone Breach   — person enters restricted polygon
  4. Weapon        — knife / gun detected (passed from detector)
"""

import time
import math
from typing import Any, Dict, List, Optional, Tuple
import config


def _point_in_polygon(px: int, py: int, polygon: List[List[int]]) -> bool:
    """Ray-casting algorithm."""
    if len(polygon) < 3:
        return False
    inside = False
    cx, cy = polygon[-1]
    for nx, ny in polygon:
        if ((ny > py) != (cy > py)) and (
            px < (cx - nx) * (py - ny) / (cy - ny + 1e-9) + nx
        ):
            inside = not inside
        cx, cy = nx, ny
    return inside


# ── Typed track state ──────────────────────────────────────────────────────────
class _TrackState:
    __slots__ = ("positions",)

    def __init__(self, cx: int, cy: int, now: float) -> None:
        self.positions: List[Tuple[int, int, float]] = [(cx, cy, now)]

    def update(self, cx: int, cy: int, now: float) -> None:
        self.positions.append((cx, cy, now))
        if len(self.positions) > 300:
            self.positions.pop(0)


# ── Bag state ─────────────────────────────────────────────────────────────────
class _BagState:
    __slots__ = ("first_alone",)

    def __init__(self, now: float) -> None:
        self.first_alone: float = now


# ── Main analyzer ─────────────────────────────────────────────────────────────
class BehaviorAnalyzer:
    def __init__(self) -> None:
        self._tracks:        Dict[Any, _TrackState] = {}
        self._bags:          Dict[int, _BagState]   = {}
        self._run_cooldown:  Dict[Any, float]        = {}  # tid → last alert time

    def analyze(
        self,
        tracks:      List[Dict],
        bags:        List[Dict],
        weapons:     List[Dict],
        frame_shape: Tuple,
    ) -> List[Dict]:
        """
        Returns list of alert dicts:
            { type, track_id, message, severity, rule }
Rules:
  1 = LOITERING
  2 = ABANDONED BAG
  3 = ZONE INTRUSION
  4 = WEAPON
  5 = RUNNING
        """
        alerts: List[Dict] = []
        now = time.time()
        live_ids = {t["track_id"] for t in tracks}

        # Purge dead tracks
        for tid in [k for k in self._tracks if k not in live_ids]:
            self._tracks.pop(tid)

        # ── 1. LOITERING ──────────────────────────────────────────────────────
        for t in tracks:
            tid        = t["track_id"]
            cx: int    = t["centroid"][0]
            cy: int    = t["centroid"][1]

            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(cx, cy, now)
            state = self._tracks[tid]
            state.update(cx, cy, now)

            if len(state.positions) >= 2:
                # ── Rule 1: LOITERING ─────────────────────────────────────
                ox, oy, ot = state.positions[0]
                dist    = math.hypot(cx - ox, cy - oy)
                elapsed = now - ot
                if elapsed >= config.LOITER_SECONDS and dist < config.LOITER_RADIUS_PX:
                    alerts.append({
                        "type":     "LOITERING",
                        "rule":     1,
                        "track_id": tid,
                        "message":  f"Person ID-{tid} stationary for {int(elapsed)}s",
                        "severity": "HIGH",
                    })

                # ── Rule 5: RUNNING ───────────────────────────────────────────
                # Measure cumulative path length over the last 1 second.
                # This is robust to zig-zag movement and varying FPS.
                RUN_WINDOW = 1.0    # seconds of history to measure over
                total_dist = 0.0
                dt_total   = 0.0
                positions  = state.positions
                # Walk backwards from end, accumulate until > RUN_WINDOW seconds ago
                for i in range(len(positions) - 1, 0, -1):
                    x1r, y1r, t1r = positions[i]
                    x0r, y0r, t0r = positions[i - 1]
                    seg_dt = t1r - t0r
                    if dt_total + seg_dt > RUN_WINDOW:
                        break
                    total_dist += math.hypot(x1r - x0r, y1r - y0r)
                    dt_total += seg_dt
                if dt_total >= 0.4:   # need at least 0.4s of history
                    px_per_sec = total_dist / dt_total
                    # Suppress repeated alerts for same track (2 s cooldown)
                    last_run = self._run_cooldown.get(tid, 0)
                    if px_per_sec >= config.RUNNING_PX_PER_SEC and (now - last_run) > 2.0:
                        self._run_cooldown[tid] = now
                        alerts.append({
                            "type":     "RUNNING",
                            "rule":     5,
                            "track_id": tid,
                            "message":  f"Person ID-{tid} running at {int(px_per_sec)} px/s",
                            "severity": "MEDIUM",
                        })

        # ── 2. ABANDONED BAG ──────────────────────────────────────────────────
        person_centroids = [(t["centroid"][0], t["centroid"][1]) for t in tracks]

        for i, bag in enumerate(bags):
            bx1, by1, bx2, by2 = bag["bbox"]
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2

            # Check if any person is within proximity
            near_person = any(
                math.hypot(bcx - px, bcy - py) < config.ABANDON_RADIUS_PX
                for px, py in person_centroids
            )

            if not near_person:
                if i not in self._bags:
                    self._bags[i] = _BagState(now)
                else:
                    elapsed = now - self._bags[i].first_alone
                    if elapsed >= config.ABANDON_SECONDS:
                        alerts.append({
                            "type":     "ABANDONED BAG",
                            "rule":     2,
                            "track_id": None,
                            "message":  f"Unattended bag for {int(elapsed)}s",
                            "severity": "HIGH",
                        })
            else:
                self._bags.pop(i, None)   # owner is back

        # ── 3. RESTRICTED ZONE INTRUSION ──────────────────────────────────────
        for zone in config.RESTRICTED_ZONES:
            for t in tracks:
                tid     = t["track_id"]
                cx, cy  = t["centroid"][0], t["centroid"][1]
                if _point_in_polygon(cx, cy, zone):
                    alerts.append({
                        "type":     "ZONE INTRUSION",
                        "rule":     3,
                        "track_id": tid,
                        "message":  f"Person ID-{tid} entered restricted zone",
                        "severity": "CRITICAL",
                    })

        # ── 4 & 6. WEAPON DETECTION ───────────────────────────────────────────
        for w in weapons:
            wtype = w.get("weapon_type", "knife")
            label = w.get("label", "Weapon")
            conf  = w.get("conf", 0)

            # ── Suppress detections inside the Weapon Zone ────────────────────
            # If a weapon zone is defined, weapons whose center falls inside it
            # are silently ignored (e.g. a display case, security desk, etc.)
            if config.FIREARMS_ZONE:
                bx1, by1, bx2, by2 = w["bbox"]
                wcx = (bx1 + bx2) // 2
                wcy = (by1 + by2) // 2
                if _point_in_polygon(wcx, wcy, config.FIREARMS_ZONE):
                    continue   # inside safe/ignore zone — skip alert

            if wtype == "firearm":
                alerts.append({
                    "type":     "FIREARM DETECTED",
                    "rule":     4,
                    "track_id": None,
                    "message":  f"🔫 {label} detected (conf {conf:.0%}) — IMMEDIATE THREAT",
                    "severity": "CRITICAL",
                })
            else:   # knife / blade / scissors
                alerts.append({
                    "type":     "KNIFE DETECTED",
                    "rule":     6,
                    "track_id": None,
                    "message":  f"🔪 {label} detected (conf {conf:.0%})",
                    "severity": "CRITICAL",
                })

        return alerts
