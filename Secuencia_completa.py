import os
import cv2
import numpy as np
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ============================================================
# IMPORTANT NOTe
# ============================================================
# Press the lowercase 'q' key to close the program and
# terminate all OpenCV visualization windows.
# ============================================================

# ============================================================
# GENERAL CONFIGURATION
# ============================================================

# Path to the segmentation model (U-Net)
MODEL_PATH  = "seg_unet.h5"

# Path to the classification model (ResNet50)
MODEL_CLASIFICADOR_PATH = "clc_resnet50.h5"

# Input size expected by the segmentation model
IMG_SIZE_SEG  = 128

# Input size expected by the classification model
IMG_SIZE_CLS  = 224

# Segmentation probability threshold
UMBRAL_SEG    = 0.50

# Extra pixels added around the detected bounding box
PADDING       = 20

# Minimum contour area (in pixels) to be considered a valid potato
MIN_AREA_PIX  = 1500

# Conversion factor from pixel area to cm² (calibrated experimentally)
CM2_PIXEL = (18.10 / 5968.0)

# Area threshold (cm²) used to decide First / Second class
UMBRAL_SIZE = 60

# Classification labels
CLASSES = ["Buena", "Mala"]

# ============================================================
# TEST IMAGE CONFIGURATION
# ============================================================

# Folder containing test images
TEST_DIR = "test images"

# Image to be processed
IMAGE_FILENAME = "0_21.png" #<------ Here you decide which image to use from the test folder

# Full path to the test image
IMAGE_PATH = os.path.join(TEST_DIR, IMAGE_FILENAME)

# ============================================================
# IMAGE PROCESSING UTILITIES
# ============================================================

def suavizar_mascara(mask):
    """
    Smooths the binary segmentation mask using:
    - Gaussian blur
    - Binary thresholding
    - Morphological closing

    This helps remove noise and fill small holes.
    """
    blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)
    return smooth_mask


def expandir_bbox(x, y, w, h, padding, ancho_img, alto_img):
    """
    Expands a bounding box by a fixed padding,
    while ensuring it stays inside image boundaries.
    """
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, ancho_img)
    y2 = min(y + h + padding, alto_img)
    return x1, y1, x2, y2


def es_crop_valido(crop_mask, umbral_pix=300):
    """
    Checks whether a cropped mask contains enough
    foreground pixels to be considered valid.
    """
    return int((crop_mask > 0).sum()) >= umbral_pix

# ============================================================
# COMPLETE PIPELINE (SEGMENTATION + CLASSIFICATION)
# ============================================================

def secuencia_completa(imagen):
    """
    Executes the full pipeline:
    1. Segmentation
    2. Contour extraction
    3. Bounding box generation
    4. Area computation
    5. Classification
    """

    H_img, W_img = imagen.shape[:2]

    # -----------------------------
    # SEGMENTATION
    # -----------------------------
    x_seg = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    x_seg = cv2.resize(x_seg, (IMG_SIZE_SEG, IMG_SIZE_SEG))
    x_seg = x_seg.astype("float32")

    # Predict segmentation mask
    pred_mask = segmentador.predict(
        np.expand_dims(x_seg, 0), verbose=0
    )[0, :, :, 0]

    # Resize mask back to original image size
    mask = cv2.resize(pred_mask, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # Binarize mask
    mask_bin = (mask > UMBRAL_SEG).astype(np.uint8) * 255

    # Smooth mask to improve contour quality
    mask_bin = suavizar_mascara(mask_bin)

    # -----------------------------
    # CONTOUR DETECTION
    # -----------------------------
    contornos, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small contours
    contornos = [
        c for c in contornos if cv2.contourArea(c) >= MIN_AREA_PIX
    ]

    if not contornos:
        return imagen, None, None, mask_bin

    # Select the largest contour (main potato)
    contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno)

    # Expand bounding box
    x1, y1, x2, y2 = expandir_bbox(
        x, y, w, h, PADDING, W_img, H_img
    )

    # Crop image and mask
    crop_img = imagen[y1:y2, x1:x2]
    crop_mask = mask_bin[y1:y2, x1:x2]

    # Compute area in cm²
    area_mask = np.sum(crop_mask > 0)
    area_cm2 = area_mask * CM2_PIXEL

    # Validate crop
    if crop_img.size == 0 or not es_crop_valido(crop_mask, umbral_pix=500):
        return imagen, None, None, mask_bin

    # Apply mask to image
    crop_segmentado = cv2.bitwise_and(
        crop_img, crop_img, mask=crop_mask
    )

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------
    x_cls = cv2.cvtColor(crop_segmentado, cv2.COLOR_BGR2RGB)
    x_cls = cv2.resize(x_cls, (IMG_SIZE_CLS, IMG_SIZE_CLS)).astype(np.float32)
    x_cls = resnet_preprocess(x_cls)

    logits = clasificador.predict(
        np.expand_dims(x_cls, 0), verbose=0
    )[0]

    clase_idx = int(np.argmax(logits))
    clase_final = CLASSES[clase_idx]

    # -----------------------------
    # DRAW RESULTS
    # -----------------------------
    color = (0, 255, 0) if clase_idx == 0 else (0, 0, 255)
    texto = f"{clase_final} ({float(logits[clase_idx]):.2f})"

    imagen_resultado = imagen.copy()
    cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        imagen_resultado,
        texto,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )

    return imagen_resultado, clase_final, area_cm2, mask_bin

# ============================================================
# LOAD MODELS
# ============================================================

segmentador = keras.models.load_model(MODEL_PATH, compile=False)
clasificador = keras.models.load_model(MODEL_CLASIFICADOR_PATH, compile=False)

print("Loading system...")

# Warm-up segmentation model
_dummy_seg = np.zeros((1, IMG_SIZE_SEG, IMG_SIZE_SEG, 3), dtype=np.float32)
segmentador.predict(_dummy_seg, verbose=0)
segmentador.predict(_dummy_seg, verbose=0)

# Warm-up classification model
_dummy_cls = np.zeros((1, IMG_SIZE_CLS, IMG_SIZE_CLS, 3), dtype=np.float32)
clasificador.predict(_dummy_cls, verbose=0)
clasificador.predict(_dummy_cls, verbose=0)

print("System ready.")

# ============================================================
# LOAD AND PROCESS IMAGE
# ============================================================

imagen = cv2.imread(IMAGE_PATH)
if imagen is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

imagen_res, clase, area_cm2, mask_bin = secuencia_completa(imagen)

print(f"Class: {clase}")
print(f"Area (cm²): {area_cm2}")

# Optional quality decision (First / Second / Third)
if clase is not None and area_cm2 is not None:
    if clase == "Buena":
        calidad = "Primera" if area_cm2 > UMBRAL_SIZE else "Segunda"
    else:
        calidad = "Tercera"
    print(f"Final quality: {calidad}")

# Display results
cv2.imshow("Original Image", imagen)
cv2.imshow("Result", imagen_res)
cv2.imshow("Mask", mask_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()
