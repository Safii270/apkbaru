from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import cv2
import numpy as np

from core import process_image, process_image_side, estimate_weight, remove_small_objects, fill_holes, canny_from_binary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "CekAnak API v2 running"}


@app.post("/process-images")
async def process_images(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
):
    try:
        front_path = os.path.join(UPLOAD_DIR, f"front_{uuid.uuid4()}.jpg")
        side_path  = os.path.join(UPLOAD_DIR, f"side_{uuid.uuid4()}.jpg")

        with open(front_path, "wb") as buffer:
            shutil.copyfileobj(front_image.file, buffer)
        with open(side_path, "wb") as buffer:
            shutil.copyfileobj(side_image.file, buffer)

        front_result = process_image(front_path)
        if not front_result["success"]:
            return {"success": False, "message": front_result["message"]}

        side_result = process_image_side(side_path)
        if not side_result["success"]:
            return {"success": False, "message": side_result["message"]}

        tinggi_cm = front_result["tinggi_cm"]
        lebar_cm  = front_result["lebar_cm"]
        tebal_cm  = side_result["tebal_cm"]

        berat_kg = estimate_weight(
            tinggi_cm=tinggi_cm,
            lebar_cm=lebar_cm,
            tebal_cm=tebal_cm,
        )

        os.remove(front_path)
        os.remove(side_path)

        return {
            "success": True,
            "data": {
                "height_cm":    tinggi_cm,
                "weight_kg":    berat_kg,
                "width_cm":     lebar_cm,
                "thickness_cm": tebal_cm,
            },
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


# ── ENDPOINT DEBUG ──────────────────────────────────────────
@app.post("/debug-images")
async def debug_images(
    front_image: UploadFile = File(...),
):
    front_path = None
    try:
        front_path = os.path.join(UPLOAD_DIR, f"debug_{uuid.uuid4()}.jpg")
        with open(front_path, "wb") as buffer:
            shutil.copyfileobj(front_image.file, buffer)

        img_bgr = cv2.imread(front_path)
        if img_bgr is None:
            return {"error": "Gagal baca gambar"}

        orig_shape = list(img_bgr.shape)
        img_bgr    = cv2.resize(img_bgr, (800, 1200))

        # Cek HSV rata-rata
        hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv)

        # Green mask
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask  = cv2.inRange(hsv, lower_green, upper_green)
        object_mask = cv2.bitwise_not(green_mask)

        green_pixels  = int(np.sum(green_mask  > 0))
        object_pixels = int(np.sum(object_mask > 0))

        # Edge detection
        kernel  = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN,  kernel)
        cleaned = cv2.morphologyEx(cleaned,     cv2.MORPH_CLOSE, kernel)
        cleaned = remove_small_objects(cleaned, min_area=1000)
        cleaned = fill_holes(cleaned)
        edge    = canny_from_binary(cleaned)

        edge_pixels    = int(np.sum(edge > 0))
        sum_bar        = (edge > 0).astype(np.uint8).sum(axis=1)
        rows_with_edge = int(np.sum(sum_bar > 0))

        return {
            "orig_shape":     orig_shape,
            "resized_shape":  [1200, 800],
            "mean_hsv":       [round(mean_hsv[0], 1), round(mean_hsv[1], 1), round(mean_hsv[2], 1)],
            "green_pixels":   green_pixels,
            "object_pixels":  object_pixels,
            "edge_pixels":    edge_pixels,
            "rows_with_edge": rows_with_edge,
            "total_pixels":   800 * 1200,
            "green_pct":      round(green_pixels / (800*1200) * 100, 1),
            "object_pct":     round(object_pixels / (800*1200) * 100, 1),
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if front_path and os.path.exists(front_path):
            os.remove(front_path)