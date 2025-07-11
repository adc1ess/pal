"""
Real-Time Object Detection in Game Window using YOLOv8 and Win32 API

This script captures the content of a specified game or application window
using Windows API and performs real-time object detection using a trained
YOLOv8 model from Ultralytics. Results are displayed on screen with FPS
and detection counts.

Key Features:
- Window capture using PrintWindow (Win32 API)
- YOLOv8 inference with custom-trained model
- Real-time object visualization
- FPS monitoring and detection count

"""

import cv2
import numpy as np
import time
from ctypes import windll
import win32gui
import win32ui
from ultralytics import YOLO


class GameCapture:
    def __init__(self, window_name):
        windll.user32.SetProcessDPIAware()
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window [{window_name}] not found. Please check:\n'
                            f'1. Is the game running?\n'
                            f'2. Is the window title correct?\n'
                            f'3. Try running as Administrator.')

        # Get client area dimensions
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        self.w, self.h = right - left, bottom - top

        # Load YOLOv8 model (replace with your trained model path)
        self.model = YOLO('F:/CVdataset/yolov5-master/runs/trainV8/weights/best.pt')

        # Create OpenCV display window
        cv2.namedWindow("AI Detection", cv2.WINDOW_NORMAL)

    def capture(self):
        """Capture the game window using PrintWindow API."""
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, self.w, self.h)
        save_dc.SelectObject(bitmap)

        result = windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Release GDI resources
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        if not result:
            raise RuntimeError(f"Screenshot failed. Return value: {result}")
        return img

    def run(self):
        print("Press 'Q' to quit detection...")
        try:
            while True:
                start_time = time.time()

                # 1. Capture game frame
                frame = self.capture()

                # 2. YOLOv8 inference (auto-resize inside YOLOv8)
                results = self.model(frame)

                # 3. Visualization (built-in .plot())
                vis = results[0].plot()

                # 4. Display window
                cv2.imshow("AI Detection", vis)
                fps = 1 / (time.time() - start_time)
                print(f"FPS: {fps:.1f} | Detections: {len(results[0].boxes)}", end='\r')

                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Replace with your game or app window title
        detector = GameCapture("Pal  ")
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")
