import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# ---------- CONFIG ----------
MODEL_PATH = 'C:/Users/netra/Downloads/best.pt'
IMG_DIR = r"C:\Users\netra\Downloads\original\filtered"
original_img_dir = Path(r"C:\Users\netra\Downloads\original")
CONF_THRESH = 0.25
ALPHA = 0.4  # Transparency for overlay
MIN_REGION_AREA = 500  # Minimum area for valid road segments
KERNEL_SIZE = 5  # For morphology
# ----------------------------

# Load model
model = YOLO(MODEL_PATH)

# Structuring element for morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# --- Post-processing function ---
def refine_mask(binary_mask):
    # Morphological operations
    refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_REGION_AREA:
            refined[labels == i] = 0

    # Smooth jagged edges
    blurred = cv2.GaussianBlur(refined, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed

# --- Get image paths ---
img_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# --- Predict and visualize ---
for i, img_path in enumerate(img_paths):
    basename = Path(img_path).stem
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    overlay = img.copy()

    results = model.predict(source=img_path, conf=CONF_THRESH, save=False)

    # Confidence-weighted mask union
    combined_mask = np.zeros((h, w), dtype=np.float32)

    for r in results:
        if r.masks is not None:
            for mask_tensor, conf in zip(r.masks.data, r.boxes.conf):
                conf = conf.item()
                binary_mask = mask_tensor.cpu().numpy()
                binary_mask = cv2.resize(binary_mask, (w, h))
                combined_mask += binary_mask * conf

    # Normalize and binarize
    combined_mask = np.clip(combined_mask, 0, 1)
    combined_mask = (combined_mask > 0.4).astype(np.uint8) * 255

    # Refine mask
    refined_mask = refine_mask(combined_mask)

    # Overlay on original image
    color_mask = np.zeros_like(img)
    color_mask[:] = (0, 255, 0)  # Green
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=refined_mask)
    overlay = cv2.addWeighted(overlay, 1 - ALPHA, color_mask, ALPHA, 0)
    
    cv2.imwrite(str(original_img_dir / f'lay_{basename}.jpg'), overlay)  # Save overlay
    cv2.imwrite(str(original_img_dir / f'bin_{basename}.png'), refined_mask)  # Save binary mask instead of overlay
    # Show overlay
    # plt.figure(figsize=(12, 6))
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.title(f"{i+1}: {os.path.basename(img_path)}")
    # plt.axis('off')
    # plt.show()
