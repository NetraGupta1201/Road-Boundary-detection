import cv2
import numpy as np
from ultralytics import YOLO

# ----------- CONFIG -----------
MODEL_PATH = "C:/Users/netra/Downloads/best.pt"
SOURCE = "C:/Users/netra/Downloads/video.mp4"
CONF_THRESH = 0.25
ALPHA = 0.4                 # Mask transparency
MIN_REGION_AREA = 500       # Minimum area for valid road segments
KERNEL_SIZE = 5             # For morphology
# ------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# Postprocessing function
def refine_mask(binary_mask):
    refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_REGION_AREA:
            refined[labels == i] = 0

    blurred = cv2.GaussianBlur(refined, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed

# Start video capture
cap = cv2.VideoCapture(SOURCE)

# Get frame size
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    combined_mask = np.zeros((h, w), dtype=np.float32)

    # Run prediction
    results = model.predict(source=frame, conf=CONF_THRESH, save=False, imgsz=640)

    for r in results:
        if r.masks is not None:
            for mask_tensor, conf in zip(r.masks.data, r.boxes.conf):
                conf = conf.item()
                binary_mask = mask_tensor.cpu().numpy()
                binary_mask = cv2.resize(binary_mask, (w, h))
                combined_mask += binary_mask * conf

    # Normalize and threshold
    combined_mask = np.clip(combined_mask, 0, 1)
    combined_mask = (combined_mask > 0.4).astype(np.uint8) * 255

    # Refine the mask
    refined_mask = refine_mask(combined_mask)

    # Create color mask
    color_mask = np.zeros_like(frame)
    color_mask[:] = (0, 255, 0)  # Green
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=refined_mask)

    # Overlay on frame
    overlay = cv2.addWeighted(overlay, 1 - ALPHA, color_mask, ALPHA, 0)

    # Display
    cv2.imshow("YOLOv11 Road Segmentation (Refined)", overlay)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
