"""
Real-Time Multi-Object Tracking with YOLOv8, BYTETracker, and Motion Compensation

This script captures a specific application or game window (e.g. a running game),
performs object detection using a trained YOLOv8 model, and applies multi-object tracking
using BYTETracker. It also compensates for camera/view movement using ORB features
and homography to improve tracking stability.

Key Features:
- Window-based screen capture (Win32 API)
- YOLOv8 object detection (Ultralytics, using trained .pt model)
- BYTETracker for real-time multi-object tracking
- Global Motion Compensation (GMC) using ORB + BFMatcher + Homography
- Smooth object ID tracking with trajectory visualization
- Suitable for game AI analysis, bot development, visual logging, etc.

"""



import cv2
import numpy as np
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


# ---------- Tracker with Motion Compensation ----------
class TrackerWithTrails:
    def __init__(self, window_name):
        self.capture_device = WindowCapture(window_name)
        self.model = YOLO('../exp/weights/best.pt')
        self.class_names = self.model.model.names  # class index mapping
        self.trajectories = {}

        args = SimpleNamespace(
            new_track_thresh=0.3,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            match_thresh=0.9,
            track_buffer=100,
            fuse_score=True
        )

        self.tracker = BYTETracker(args, frame_rate=30)

        # Initialize previous grayscale frame for homography estimation
        self.prev_gray = None

    def update(self):
        frame = self.capture_device.capture()
        yolo_result = self.model(frame)[0]

        # Convert YOLO results to numpy
        confs = yolo_result.boxes.conf.cpu().numpy()
        xywhs = yolo_result.boxes.xywh.cpu().numpy()
        clss = yolo_result.boxes.cls.cpu().numpy()

        # Confidence threshold filter
        conf_threshold = 0.4
        mask = confs > conf_threshold

        # Construct BYTETracker input format
        results = SimpleNamespace(
            conf=confs[mask],
            xywh=xywhs[mask],
            cls=clss[mask]
        )

        # Compute grayscale image for homography
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = None
        if self.prev_gray is not None:
            # ORB feature matching
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)

            # Brute-force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        self.prev_gray = gray.copy()

        # Run BYTETracker
        tracks = self.tracker.update(results, frame)

        # Compensate trajectory coordinates using homography matrix
        if H is not None:
            H_inv = np.linalg.inv(H)
            for track_id in self.trajectories:
                pts = np.array(self.trajectories[track_id])
                if len(pts) == 0:
                    continue
                pts_hom = np.hstack([pts, np.ones((pts.shape[0],1))])
                pts_transformed = (H @ pts_hom.T).T
                pts_transformed = pts_transformed[:, :2] / pts_transformed[:, 2:3]
                self.trajectories[track_id] = pts_transformed.astype(int).tolist()

        # Draw boxes and trajectories
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            score = float(track[5])
            cls_id = int(track[6])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # Update trajectory
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((cx, cy))
            if len(self.trajectories[track_id]) > 30:
                self.trajectories[track_id].pop(0)

            # Class name
            cls_name = self.class_names.get(cls_id, "Unknown")
            label = f"{cls_name} ID {track_id}"

            # Draw box + ID + class
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Draw trajectory lines
            pts = self.trajectories[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0,255,255), 2)

        return frame

    def run(self):
        while True:
            frame = self.update()
            cv2.imshow("YOLOv8 + ByteTrack + GMC", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = TrackerWithTrails("Pal  ")  # your game window title
    tracker.run()
