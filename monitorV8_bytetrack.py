"""
Real-Time Object Tracking in a Specified Window Using YOLOv8 and BYTETracker

This script captures frames from a specific window (e.g., a game window) using Windows API,
performs object detection with a custom-trained YOLOv8 model, and applies BYTETracker
for multi-object tracking. It visualizes bounding boxes, object IDs, class names, and
draws movement trails of tracked objects.

Requirements:
- Windows OS
- Python 3.x
- ultralytics (YOLOv8), opencv-python, numpy, pywin32

"""

import cv2
import numpy as np
import torch
import time
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ctypes import windll
import win32gui
import win32ui
import win32con


# ---------- Window Capture Class ----------
class WindowCapture:
    def __init__(self, window_name):
        # Find the window handle by its title
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Window [{window_name}] not found")
        # Make the process DPI aware to capture high-DPI windows correctly
        windll.user32.SetProcessDPIAware()
        # Get window coordinates
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.width = self.window_rect[2] - self.window_rect[0]
        self.height = self.window_rect[3] - self.window_rect[1]

    def capture(self):
        hwnd = self.hwnd
        w, h = self.width, self.height

        # Get device context (DC) for the window
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create a bitmap object compatible with the DC
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # Copy the window image into the bitmap
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

        # Convert bitmap to numpy array
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (h, w, 4)  # BGRA format

        # Release resources
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        if result != 1:
            raise Exception("Screenshot failed")

        # Convert BGRA to BGR for OpenCV processing
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ---------- Tracker with Trajectory Visualization ----------
class TrackerWithTrails:
    def __init__(self, window_name):
        # Initialize the window capture object
        self.capture_device = WindowCapture(window_name)
        # Load the YOLOv8 model (use your model path)
        self.model = YOLO('../exp/weights/best.pt')
        # Get class names from the model
        self.class_names = self.model.model.names
        # Dictionary to store object trajectories {track_id: [(x,y), ...]}
        self.trajectories = {}

        # Tracker configuration parameters
        args = SimpleNamespace(
            new_track_thresh=0.4,   # Lower threshold for creating new tracks
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            match_thresh=0.8,       # Higher IOU threshold for matching
            track_buffer=50,        # Number of frames to keep "lost" tracks
            fuse_score=True
        )

        # Initialize BYTETracker with parameters and frame rate
        self.tracker = BYTETracker(args, frame_rate=30)

    def update(self):
        # Capture current frame from the window
        frame = self.capture_device.capture()
        # Run YOLOv8 inference
        yolo_result = self.model(frame)[0]

        # Convert detections to numpy arrays
        confs = yolo_result.boxes.conf.cpu().numpy()
        xywhs = yolo_result.boxes.xywh.cpu().numpy()
        clss = yolo_result.boxes.cls.cpu().numpy()

        # Filter detections by confidence threshold
        conf_threshold = 0.4
        mask = confs > conf_threshold

        # Prepare detection results for BYTETracker (filtered)
        results = SimpleNamespace(
            conf=confs[mask],
            xywh=xywhs[mask],
            cls=clss[mask]
        )

        # Update tracker with new detections
        tracks = self.tracker.update(results, frame)

        # Visualize tracked objects and their trajectories
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            score = float(track[5])
            cls_id = int(track[6])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update trajectory points for the track
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((cx, cy))
            # Keep trajectory length to max 30 points
            if len(self.trajectories[track_id]) > 30:
                self.trajectories[track_id].pop(0)

            # Get class name string
            cls_name = self.class_names.get(cls_id, "Unknown")
            label = f"{cls_name} ID {track_id}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw trajectory lines
            pts = self.trajectories[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

        return frame

    def run(self):
        while True:
            frame = self.update()
            cv2.imshow("YOLOv8 + ByteTrack", frame)
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# ---------- Main Entry Point ----------
if __name__ == "__main__":
    # Replace "Pal  " with your target window title, be precise including spaces
    tracker = TrackerWithTrails("Pal  ")
    tracker.run()
