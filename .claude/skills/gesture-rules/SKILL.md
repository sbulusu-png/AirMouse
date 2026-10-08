---
name: gesture-rules
description: Standards for MediaPipe landmark extraction, smoothing math, and web IPC
---

When building or editing gesture-related code in this project:
1. Always mirror video feeds horizontally before processing coordinates.
2. Use Exponential Moving Average (EMA) filtering for landmark interpolation.
3. Keep PyAutoGUI failsafe enabled (`pyautogui.FAILSAFE = True`).
4. Ensure MediaPipe normalized coordinates (0.0–1.0) are correctly mapped to screen pixel dimensions.
