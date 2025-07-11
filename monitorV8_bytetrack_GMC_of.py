"""
Real-Time Object Tracking with YOLOv8, BYTETracker, and Motion Compensation (GMC)

This script captures a target application or game window, detects objects using a trained YOLOv8 model,
tracks them with BYTETracker, and applies motion compensation using optical flow and affine transformation
to handle camera or scene movement. It also visualizes object trajectories for better understanding.

Main Features:
- Window-level screen capture (Windows API + OpenCV)
- YOLOv8 object detection (Ultralytics, custom-trained model)
- BYTETracker for real-time multi-object tracking
- Motion compensation using optical flow (cv2.calcOpticalFlowPyrLK)
- Trajectory visualization with IDs and class names
- Real-time rendering with OpenCV


"""

import cv2
import numpy as np
import time
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ctypes import windll
import win32gui
import win32ui

# ---------- Window Capture ----------
class WindowCapture:
    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Window [{window_name}] not found")
        windll.user32.SetProcessDPIAware()
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.width = self.window_rect[2] - self.window_rect[0]
        self.height = self.window_rect[3] - self.window_rect[1]

    def capture(self):
        hwnd = self.hwnd
        w, h = self.width, self.height

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (h, w, 4)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        if result != 1:
            raise Exception("Screenshot failed")
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ---------- Tracker with Trajectory and Motion Compensation ----------
class TrackerWithTrails:
    def __init__(self, window_name):
        self.capture_device = WindowCapture(window_name)
        self.model = YOLO('../runs/trainV8/weights/best.pt')  # Replace with your trained model path
        self.class_names = self.model.model.names
        self.trajectories = {}

        args = SimpleNamespace(
            new_track_thresh=0.4,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            match_thresh=0.8,
            track_buffer=50,
            fuse_score=True
        )
        self.tracker = BYTETracker(args, frame_rate=30)
        self.prev_gray = None

    # Estimate camera motion using optical flow and affine transform
    def compensate_motion(self, prev_gray, curr_gray):
        feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is None:
            return None

        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 6 or len(good_curr) < 6:
            return None

        M, inliers = cv2.estimateAffinePartial2D(good_curr, good_prev)
        return M

    def update(self):
        frame = self.capture_device.capture()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Estimate motion matrix
        if self.prev_gray is not None:
            M = self.compensate_motion(self.prev_gray, curr_gray)
        else:
            M = None

        yolo_result = self.model(frame)[0]

        # Extract and filter detections
        confs = yolo_result.boxes.conf.cpu().numpy()
        xywhs = yolo_result.boxes.xywh.cpu().numpy()
        clss = yolo_result.boxes.cls.cpu().numpy()

        conf_threshold = 0.4
        mask = confs > conf_threshold
        confs = confs[mask]
        xywhs = xywhs[mask]
        clss = clss[mask]

        # Apply motion compensation to center points
        if M is not None:
            cx = xywhs[:, 0]
            cy = xywhs[:, 1]
            pts = np.vstack([cx, cy, np.ones_like(cx)])  # Shape: (3, N)
            compensated_pts = M @ pts  # Shape: (2, N)
            xywhs[:, 0], xywhs[:, 1] = compensated_pts[0, :], compensated_pts[1, :]

        results = SimpleNamespace(conf=confs, xywh=xywhs, cls=clss)
        tracks = self.tracker.update(results, frame)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            cls_id = int(track[6])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((cx, cy))
            if len(self.trajectories[track_id]) > 30:
                self.trajectories[track_id].pop(0)

            cls_name = self.class_names.get(cls_id, "Unknown")
            label = f"{cls_name} ID {track_id}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw trajectory lines
            pts = self.trajectories[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

        self.prev_gray = curr_gray.copy()
        return frame

    def run(self):
        while True:
            frame = self.update()
            cv2.imshow("YOLOv8 + ByteTrack + GMC", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

# ---------- Main Entry ----------
if __name__ == "__main__":
    tracker = TrackerWithTrails("Pal  ")  # Replace with your target window title
    tracker.run()
