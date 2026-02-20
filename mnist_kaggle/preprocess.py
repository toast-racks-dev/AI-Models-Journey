import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_canvas(canvas_bgr, model):
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)

    if np.count_nonzero(gray) < 500:
        return None

    # Step 3 — Tight bounding box around all non-zero pixels
    points = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(points)

    canvas_h, canvas_w = gray.shape
    pad_x = int(w * 0.20)
    pad_y = int(h * 0.20)

    x1 = max(x - pad_x, 0)  
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, canvas_w)
    y2 = min(y + h + pad_y, canvas_h)
    crop = gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(crop, 10, 255, cv2.THRESH_BINARY)

    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = (resized / 255.0 - 0.1307) / 0.3081

    tensor = torch.FloatTensor(normalized).reshape(1, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    probabilities = F.softmax(output, dim=1)
    confidence = torch.max(probabilities).item() * 100.0
    digit = torch.argmax(probabilities).item()

    return (digit, confidence)
