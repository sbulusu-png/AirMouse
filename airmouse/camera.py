"""Webcam capture on a background thread that always returns mirrored frames.

Reading a frame blocks until the camera delivers it (~33 ms at 30 fps). Capturing on
its own thread lets hand tracking of frame N overlap with the capture of frame N+1.
"""

import threading

import cv2
import numpy as np

FRAME_TIMEOUT_S = 2.0


class Camera:
    def __init__(self, index: int, width: int, height: int, fps: int = 30) -> None:
        # Media Foundation measured faster than DirectShow here, both to open (1.2 s vs
        # 4.9 s) and to stream (30 vs 22 fps at 720p).
        self._cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        # MJPG lets most webcams deliver 720p at 30 fps over USB.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Without an explicit rate some drivers default to 15 fps.
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._read_seq = 0
        self._failed = False
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, name="airmouse-camera",
                                        daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._failed = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self) -> np.ndarray | None:
        """Wait for the next new frame and return it mirrored (BGR); None if the camera died.

        Mirroring makes on-screen motion match the hand: move right, cursor goes right.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._read_seq or self._failed,
                                       timeout=FRAME_TIMEOUT_S):
                return None
            if self._seq == self._read_seq or self._frame is None:  # failed, nothing new
                return None
            self._read_seq = self._seq
            frame = self._frame
        return cv2.flip(frame, 1)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=FRAME_TIMEOUT_S)
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
