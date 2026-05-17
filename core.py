import math
import cv2
import numpy as np

RASIO = 0.05275

# =========================
# PEMROSESAN CITRA
# =========================
def remove_green_background(img_bgr):
    """Hapus background hijau (green screen) menggunakan HSV masking"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    object_mask = cv2.bitwise_not(green_mask)
    kernel      = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN,  kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    return object_mask


def threshold_otsu_scaled(gray_img, scale=1.0):
    otsu_thresh_val, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled_thresh = min(255, int(otsu_thresh_val * scale))
    _, binary = cv2.threshold(gray_img, scaled_thresh, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_objects(binary_img, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    result = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result


def fill_holes(binary_img):
    flood = binary_img.copy()
    h, w  = binary_img.shape
    mask  = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary_img | flood_inv


def canny_from_binary(binary_img):
    return cv2.Canny(binary_img, 50, 150)


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


def _buat_binary(img_bgr, scale=1.0, use_green_removal=True):
    """Helper: buat binary mask dari gambar BGR"""
    if use_green_removal:
        binary = remove_green_background(img_bgr)
    else:
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        binary = threshold_otsu_scaled(gray, scale=scale)

    binary = remove_small_objects(binary, min_area=1000)
    binary = fill_holes(binary)
    return binary


# =========================
# PROSES GAMBAR DEPAN
# =========================
def process_image(image_path):
    """
    Proses citra hadap depan.
    Dipanggil dari api.py → dikirim hasilnya ke Flutter (photoindent.dart).
    Return: dict {success, tinggi_cm, lebar_cm, A, t}
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"success": False, "message": "Gagal membaca citra depan."}

        # Resize standar agar konsisten di semua resolusi HP
        img_bgr = cv2.resize(img_bgr, (800, 1200))

        binary = _buat_binary(img_bgr, scale=1.2, use_green_removal=True)
        edge   = canny_from_binary(binary)

        bts_atas, bts_bawah, bts_kiri, bts_kanan = hitung_batas(edge, threshold_col=0)

        tinggi_pixel = bts_bawah - bts_atas
        tinggi_cm    = tinggi_pixel * RASIO
        lebar_pixel  = bts_kanan - bts_kiri
        lebar_cm     = lebar_pixel * RASIO
        A            = lebar_pixel * 0.385
        t            = tinggi_pixel

        return {
            "success":   True,
            "tinggi_cm": round(tinggi_cm, 2),
            "lebar_cm":  round(lebar_cm, 2),
            "A":         A,
            "t":         t,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


# =========================
# PROSES GAMBAR SAMPING
# =========================
def process_image_side(image_path):
    """
    Proses citra hadap samping.
    Dipanggil dari api.py → dikirim hasilnya ke Flutter (photoindent.dart).
    Return: dict {success, tebal_cm, tinggi_cm, B}
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"success": False, "message": "Gagal membaca citra samping."}

        img_bgr = cv2.resize(img_bgr, (800, 1200))

        binary = _buat_binary(img_bgr, scale=1.0, use_green_removal=True)
        edge   = canny_from_binary(binary)

        bts_atas, bts_bawah, bts_kiri, bts_kanan = hitung_batas(edge, threshold_col=0)

        lebar_pixel  = bts_kanan - bts_kiri
        tebal_cm     = lebar_pixel * RASIO
        B            = lebar_pixel * 0.385
        tinggi_pixel = bts_bawah - bts_atas
        tinggi_cm    = tinggi_pixel * RASIO

        return {
            "success":   True,
            "tebal_cm":  round(tebal_cm, 2),
            "tinggi_cm": round(tinggi_cm, 2),
            "B":         B,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}


# =========================
# ESTIMASI BERAT BADAN (BSA)
# =========================
def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    """
    Estimasi berat badan menggunakan formula BSA ellipse.
    Dipanggil dari api.py setelah dapat tinggi & lebar dari process_image.
    Hasilnya dikirim ke Flutter sebagai weight_kg → photoindent.dart.
    """
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