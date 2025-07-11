"""
Real-Time Object Detection in Game Window using YOLOv5 (Ultralytics) and Win32 API

This script captures the screen content of a specified game or application window,
performs object detection using a trained YOLOv5 model, and displays the results in real-time.

Key Features:
- Window capture using Windows API (PrintWindow via win32gui/win32ui)
- YOLOv5 object detection (custom-trained model supported)
- Real-time FPS display and detection visualization

"""

import torch
import cv2
import numpy as np
import time
from ctypes import windll
import win32gui
import win32ui


class GameCapture:
    def __init__(self, window_name):
        # Set DPI awareness to capture high-resolution windows properly
        windll.user32.SetProcessDPIAware()

        # Find the window handle using its title
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window [{window_name}] not found. Please check:\n'
                            f'1. Is the game/application running?\n'
                            f'2. Is the window title correct?\n'
                            f'3. Try running this script as Administrator.')

        # Get window client area size
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        self.w, self.h = right - left, bottom - top

        # Load YOLOv5 model from Ultralytics with custom-trained weights
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='../runs/train/exp3/weights/best.pt',
                                    force_reload=True)
        self.model.conf = 0.3  # Confidence threshold

        # Create OpenCV window for displaying detection
        cv2.namedWindow("AI Detection", cv2.WINDOW_NORMAL)

    def capture(self):
        """Capture the target window content using Windows API (PrintWindow)."""
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, self.w, self.h)
        save_dc.SelectObject(bitmap)

        # Perform screen capture using PrintWindow (flag 3 = client + layered)
        result = windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)

        # Extract bitmap as NumPy image
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Release GDI objects
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        if not result:
            raise RuntimeError(f"Failed to capture screen. Return code: {result}")
        return img

    def run(self):
        print("Press 'Q' to quit detection...")
        try:
            while True:
                start_time = time.time()

                # 1. Capture frame from the game window
                frame = self.capture()

                # 2. Run YOLOv5 inference on the captured frame (resize internally)
                results = self.model(frame, size=640)

                # 3. Render detection results (either using built-in render or manual)
                if hasattr(results, 'render'):
                    vis = np.squeeze(results.render())
                else:
                    vis = frame.copy()
                    for *xyxy, conf, cls in results.pred[0]:
                        cv2.rectangle(vis, (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                # 4. Display the detection window
                cv2.imshow("AI Detection", vis)
                fps = 1 / (time.time() - start_time)
                print(f"FPS: {fps:.1f} | Detected objects: {len(results.pred[0])}", end='\r')

                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Replace with your actual game or application window title
        detector = GameCapture("Pal  ")
        # Example: "Pal  " is the window title of the game Palworld
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")
