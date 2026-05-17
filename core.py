import math
import cv2
import numpy as np

RASIO = 0.05275

def remove_green_background(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    object_mask = cv2.bitwise_not(green_mask)
    kernel      = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN,  kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    return object_mask


def remove_small_objects(binary_img, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    result = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result


def hitung_batas(edge_img, threshold_col=0):
    edge_bin = (edge_img > 0).astype(np.uint8)

    sum_bar = edge_bin.sum(axis=1)
    rows    = np.where(sum_bar > 0)[0]
    if len(rows) == 0:
        raise ValueError("Objek tidak terdeteksi pada arah baris.")
    bts_atas  = rows[0]
    bts_bawah = rows[-1]

    sum_kol = edge_bin.sum(axis=0)
    cols    = np.where(sum_kol > threshold_col)[0]
    if len(cols) == 0:
        cols = np.where(sum_kol > 0)[0]
    if len(cols) == 0:
        raise ValueError("Objek tidak terdeteksi pada arah kolom.")
    bts_kiri  = cols[0]
    bts_kanan = cols[-1]

    return bts_atas, bts_bawah, bts_kiri, bts_kanan


def canny_from_binary(binary_img):
    return cv2.Canny(binary_img, 50, 150)


def fill_holes(binary_img):
    flood = binary_img.copy()
    h, w  = binary_img.shape
    mask  = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary_img | flood_inv


def _proses_mask(img_bgr, min_area=1000):
    """
    Buat edge dari gambar:
    1. Green removal → object mask
    2. remove small objects
    3. Langsung Canny dari mask (skip fill_holes yang bermasalah)
    """
    mask = remove_green_background(img_bgr)
    mask = remove_small_objects(mask, min_area=min_area)

    # Pastikan binary 0/255
    mask = (mask > 0).astype(np.uint8) * 255

    # Blur dulu agar Canny bekerja lebih baik pada mask
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edge = cv2.Canny(mask_blur, 30, 100)

    return edge


def process_image(image_path):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"success": False, "message": "Gagal membaca citra depan."}

        img_bgr = cv2.resize(img_bgr, (800, 1200))
        edge    = _proses_mask(img_bgr, min_area=1000)

        bts_atas, bts_bawah, bts_kiri, bts_kanan = hitung_batas(edge, threshold_col=0)

        tinggi_pixel = bts_bawah - bts_atas
        tinggi_cm    = tinggi_pixel * RASIO
        lebar_pixel  = bts_kanan - bts_kiri
        lebar_cm     = lebar_pixel * RASIO

        return {
            "success":   True,
            "tinggi_cm": round(tinggi_cm, 2),
            "lebar_cm":  round(lebar_cm, 2),
            "A":         lebar_pixel * 0.385,
            "t":         tinggi_pixel,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


def process_image_side(image_path):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"success": False, "message": "Gagal membaca citra samping."}

        img_bgr = cv2.resize(img_bgr, (800, 1200))
        edge    = _proses_mask(img_bgr, min_area=1000)

        bts_atas, bts_bawah, bts_kiri, bts_kanan = hitung_batas(edge, threshold_col=0)

        lebar_pixel  = bts_kanan - bts_kiri
        tebal_cm     = lebar_pixel * RASIO
        tinggi_pixel = bts_bawah - bts_atas
        tinggi_cm    = tinggi_pixel * RASIO

        return {
            "success":   True,
            "tebal_cm":  round(tebal_cm, 2),
            "tinggi_cm": round(tinggi_cm, 2),
            "B":         lebar_pixel * 0.385,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    A = (lebar_cm / RASIO) * 0.385
    B = (tebal_cm / RASIO) * 0.385
    t = tinggi_cm / RASIO

    h = ((A - B) ** 2) / ((A + B) ** 2)

    BSA = (
        2 * (math.pi / 2 * (A * B)) +
        (math.pi / 2 * (A + B) * (1 + ((3 * h) / (10 + math.sqrt(4 - 3 * h)))) * t)
    ) * 1e-5 * RASIO

    berat_kg = (BSA ** 2) * 3600 / (t * RASIO)

    return round(berat_kg, 2)