import cv2
import numpy as np

from model import MNISTConvNet         
from inference import load_model
from preprocess import preprocess_canvas
model = load_model()
last_prediction = None
last_confidence  = None

try:
    hsv_value = np.load("hsv_value.npy")
    lower_range = np.array(hsv_value[:3])
    upper_range = np.array(hsv_value[3:])
except FileNotFoundError:
    print("Warning: hsv_value.npy not found. Run hsv_value_trackbar.py first.")
    lower_range = np.array([5,  100, 100])
    upper_range = np.array([15, 255, 255])

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
canvas = None

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0          
ALPHA = 0.4                         
MIN_MOVE = 5                        

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask,  kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = cx, cy
            else:
                smooth_x = int(ALPHA * cx + (1 - ALPHA) * smooth_x)
                smooth_y = int(ALPHA * cy + (1 - ALPHA) * smooth_y)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = smooth_x, smooth_y

            dist = ((smooth_x - prev_x) ** 2 + (smooth_y - prev_y) ** 2) ** 0.5
            if dist >= MIN_MOVE:
                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (0, 255, 0), 10)
                prev_x, prev_y = smooth_x, smooth_y

            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0
    else:
        prev_x, prev_y = 0, 0
        smooth_x, smooth_y = 0, 0

    frame = cv2.add(canvas, frame)

    if last_prediction is not None:
        if last_confidence > 80:
            col = (0, 255, 0)      
        elif last_confidence > 50:
            col = (0, 255, 255)    
        else:
            col = (0, 0, 255)      

        cv2.putText(frame, f'Digit: {last_prediction}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, col, 4)
        cv2.putText(frame, f'Conf:  {last_confidence:.1f}%', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)

    cv2.putText(frame, "Enter/Space: predict | c: clear | q/Esc: quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (200, 200, 200), 1)

    cv2.imshow("Live Writing", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)

    if key in (13, 32):                          
        result = preprocess_canvas(canvas, model)
        if result:
            last_prediction, last_confidence = result
            print(f"Prediction: {last_prediction}  Confidence: {last_confidence:.1f}%")

    elif key == ord('c'):                        
        canvas = np.zeros_like(frame)
        last_prediction = None
        last_confidence  = None
        prev_x, prev_y   = 0, 0
        smooth_x, smooth_y = 0, 0
        print("Canvas cleared.")

    elif key in (ord('q'), 27):                  
        break

cap.release()
cv2.destroyAllWindows()
