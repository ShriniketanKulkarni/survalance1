import traceback
import sys

try:
    import test_yolo_raw
except Exception as e:
    with open("crash.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    print("Crash saved to crash.txt")
