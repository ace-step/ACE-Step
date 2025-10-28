# Requires: pip install pyautogui
import time
import itertools
import pyautogui

pyautogui.FAILSAFE = False  # optional: avoid moving mouse to corner to stop
interval_seconds = 2.0

print("Focus your editor window now. Press Ctrl+C to stop.")
try:
    for key in itertools.cycle(("left", "right")):
        pyautogui.press(key)
        time.sleep(interval_seconds)
except KeyboardInterrupt:
    print("Stopped by user.")