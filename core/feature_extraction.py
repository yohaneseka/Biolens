import cv2 as cv
import numpy as np
import pandas as pd
import os
import warnings
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")


def isolate_target_cell(crop_bgr):
    """
    Dari crop BGR, kembalikan binary mask hanya sel RBC terdekat ke tengah crop.
    Konsisten dengan notebook: LAB L-channel Otsu INV + MORPH_CLOSE 7x7.
    """
    h, w  = crop_bgr.shape[:2]
    cy_c  = h / 2.0
    cx_c  = w / 2.0

    lab   = cv.cvtColor(crop_bgr, cv.COLOR_BGR2LAB)
    L_ch  = lab[:, :, 0]

    _, raw_mask = cv.threshold(L_ch, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed  = cv.morphologyEx(raw_mask, cv.MORPH_CLOSE, k_close)

    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        closed, connectivity=8
    )

    MIN_AREA   = 200
    best_label = -1
    best_dist  = float('inf')

    for lbl in range(1, n_labels):
        area = stats[lbl, cv.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        ccx, ccy = centroids[lbl]
        dist = np.sqrt((ccx - cx_c) ** 2 + (ccy - cy_c) ** 2)
        if dist < best_dist:
            best_dist  = dist
            best_label = lbl

    if best_label == -1:
        return np.zeros((h, w), dtype=np.uint8), False

    cell_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    return cell_mask, True


def quality_filter_cells(cropped_cells, bounding_boxes, cell_masks, img_shape,
                         border_margin=5, min_area=200, max_area=5000,
                         min_dim=15, min_ar=0.6, max_ar=1.6):
    img_h, img_w = img_shape[:2]
    filtered_cells, filtered_boxes, filtered_masks, cell_labels = [], [], [], []
    stats = {"border": 0, "small": 0, "large": 0, "shape": 0, "dim": 0, "passed": 0}

    for idx in range(len(cropped_cells)):
        x, y, w, h = bounding_boxes[idx]
        area = w * h
        original_label = idx + 1

        if (x <= border_margin or y <= border_margin or
                x + w >= img_w - border_margin or y + h >= img_h - border_margin):
            stats["border"] += 1; continue
        if area < min_area:
            stats["small"] += 1; continue
        if area > max_area:
            stats["large"] += 1; continue
        ar = w / h if h > 0 else 0
        if ar < min_ar or ar > max_ar:
            stats["shape"] += 1; continue
        if w < min_dim or h < min_dim:
            stats["dim"] += 1; continue

        filtered_cells.append(cropped_cells[idx])
        filtered_boxes.append(bounding_boxes[idx])
        filtered_masks.append(cell_masks[idx])
        cell_labels.append(original_label)
        stats["passed"] += 1

    return filtered_cells, filtered_boxes, filtered_masks, cell_labels, stats


def extract_all_features(cell_img_bgr, cell_mask_external, cell_label, bbox_coords=None):
    """
    Ekstrak semua fitur dari satu sel.
    Konsisten dengan notebook: pakai isolate_target_cell() untuk cell_mask,
    bukan mask dari segmentasi luar.
    """
    h, w = cell_img_bgr.shape[:2]

    # ── Konversi ruang warna (sama dengan notebook)
    lab  = cv.cvtColor(cell_img_bgr, cv.COLOR_BGR2LAB)
    L_ch = lab[:, :, 0]
    b_ch, g_ch, r_ch = cv.split(cell_img_bgr)

    # ── 1. Cell mask via isolate_target_cell (konsisten dengan notebook)
    cell_mask, found = isolate_target_cell(cell_img_bgr)
    if not found:
        cell_mask = cell_mask_external  # fallback

    # ── 2. Fitur morfologi
    area = perimeter = maj_ax = min_ax = 0.0
    compactness = eccentricity = solidity = aspect_ratio = rectangularity = 0.0

    contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        c         = max(contours, key=cv.contourArea)
        area      = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        _, _, wc, hc = cv.boundingRect(c)
        rectangularity = area / (wc * hc) if wc * hc > 0 else 0.0
        if len(c) >= 5:
            try:
                (_, _), (min_ax, maj_ax), _ = cv.fitEllipse(c)
                aspect_ratio = maj_ax / min_ax if min_ax > 0 else 0.0
                eccentricity = (np.sqrt(1 - (min_ax ** 2 / maj_ax ** 2))
                                if maj_ax > min_ax else 0.0)
            except Exception:
                pass
        hull      = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0.0
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0

    # ── 3. Fitur warna dari pixel BGR asli di dalam cell_mask (sama dengan notebook)
    mask_px = cell_mask == 255
    r_pix   = r_ch[mask_px]
    g_pix   = g_ch[mask_px]
    b_pix   = b_ch[mask_px]
    if len(r_pix) == 0:
        r_pix = g_pix = b_pix = np.array([0], dtype=np.uint8)

    c_mean_r = float(np.mean(r_pix)); c_std_r  = float(np.std(r_pix))
    c_skew_r = float(skew(r_pix));   c_kurt_r = float(kurtosis(r_pix))
    c_mean_g = float(np.mean(g_pix)); c_std_g  = float(np.std(g_pix))
    c_skew_g = float(skew(g_pix));   c_kurt_g = float(kurtosis(g_pix))
    c_mean_b = float(np.mean(b_pix)); c_std_b  = float(np.std(b_pix))
    c_skew_b = float(skew(b_pix));   c_kurt_b = float(kurtosis(b_pix))

    # ── 4. CP mask — percentile-75 L-channel (sama dengan notebook)
    cp_mask  = np.zeros((h, w), dtype=np.uint8)
    l_pix_in = L_ch[mask_px]
    if len(l_pix_in) > 0:
        thresh_cp = float(np.percentile(l_pix_in, 75))
        _, cp_raw = cv.threshold(L_ch, thresh_cp, 255, cv.THRESH_BINARY)
        cp_mask   = cv.bitwise_and(cp_raw, cp_raw, mask=cell_mask)

    # ── 5. Fitur central pallor
    cp_contours, _ = cv.findContours(cp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cp_area = cp_perim = cp_maj = cp_min = 0.0
    cp_comp = cp_ecc = cp_solid = cp_ratio = 0.0

    if cp_contours:
        c_cp     = max(cp_contours, key=cv.contourArea)
        cp_area  = cv.contourArea(c_cp)
        cp_perim = cv.arcLength(c_cp, True)
        # CP_Ratio = cp_area / area sel (sama dengan notebook)
        cp_ratio = cp_area / area if area > 0 else 0.0
        if len(c_cp) >= 5:
            try:
                _, (cp_min, cp_maj), _ = cv.fitEllipse(c_cp)
                cp_ecc = (np.sqrt(1 - (cp_min ** 2 / cp_maj ** 2))
                          if cp_maj > cp_min else 0.0)
            except Exception:
                pass
        hull_cp      = cv.convexHull(c_cp)
        hull_cp_area = cv.contourArea(hull_cp)
        cp_solid = cp_area / hull_cp_area if hull_cp_area > 0 else 0.0
        cp_comp  = (cp_perim ** 2) / (4 * np.pi * cp_area) if cp_area > 0 else 0.0

    # ── 6. GLCM dari grayscale asli (levels=256, sama dengan notebook)
    gray = cv.cvtColor(cell_img_bgr, cv.COLOR_BGR2GRAY)
    try:
        glcm          = graycomatrix(gray, distances=[1],
                                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                     levels=256, symmetric=True, normed=True)
        contrast_g    = graycoprops(glcm, 'contrast')[0]
        correlation_g = graycoprops(glcm, 'correlation')[0]
        energy_g      = graycoprops(glcm, 'energy')[0]
        homogeneity_g = graycoprops(glcm, 'homogeneity')[0]
    except Exception:
        contrast_g = correlation_g = energy_g = homogeneity_g = np.array([0.0])

    features = {
        "Cell_Label":            cell_label,
        "X":                     bbox_coords[0] if bbox_coords else 0,
        "Y":                     bbox_coords[1] if bbox_coords else 0,
        "Area":                  round(area, 2),
        "Perimeter":             round(perimeter, 2),
        "Major_Axis":            round(maj_ax, 2),
        "Minor_Axis":            round(min_ax, 2),
        "Compactness":           round(compactness, 4),
        "Eccentricity":          round(eccentricity, 4),
        "Solidity":              round(solidity, 4),
        "Aspect_Ratio":          round(aspect_ratio, 4),
        "Rectangularity":        round(rectangularity, 4),
        "CP_Area":               round(cp_area, 2),
        "CP_Perimeter":          round(cp_perim, 2),
        "CP_Major_Axis":         round(cp_maj, 2),
        "CP_Minor_Axis":         round(cp_min, 2),
        "CP_Compactness":        round(cp_comp, 4),
        "CP_Eccentricity":       round(cp_ecc, 4),
        "CP_Solidity":           round(cp_solid, 4),
        "CP_Ratio":              round(cp_ratio, 4),
        "GLCM_Contrast_Mean":    round(float(np.mean(contrast_g)), 6),
        "GLCM_Correlation_Mean": round(float(np.mean(correlation_g)), 6),
        "GLCM_Energy_Mean":      round(float(np.mean(energy_g)), 6),
        "GLCM_Homogeneity_Mean": round(float(np.mean(homogeneity_g)), 6),
        "Color_Mean_R":          round(c_mean_r, 4),
        "Color_Std_R":           round(c_std_r, 4),
        "Color_Skewness_R":      round(c_skew_r, 4),
        "Color_Kurtosis_R":      round(c_kurt_r, 4),
        "Color_Mean_G":          round(c_mean_g, 4),
        "Color_Std_G":           round(c_std_g, 4),
        "Color_Skewness_G":      round(c_skew_g, 4),
        "Color_Kurtosis_G":      round(c_kurt_g, 4),
        "Color_Mean_B":          round(c_mean_b, 4),
        "Color_Std_B":           round(c_std_b, 4),
        "Color_Skewness_B":      round(c_skew_b, 4),
        "Color_Kurtosis_B":      round(c_kurt_b, 4),
    }
    return features


def run_feature_extraction(extracted_cells, bounding_boxes, cell_masks, img_shape,
                           output_csv_path=None, border_margin=5,
                           min_area=200, max_area=5000, min_dim=15,
                           min_ar=0.6, max_ar=1.6):
    """
    extracted_cells: list crop sel RGB dari pipeline Main.py.
    Konversi ke BGR dilakukan di sini agar konsisten dengan notebook.
    """
    cell_imgs = [item[0] if isinstance(item, tuple) else item for item in extracted_cells]

    filtered_cells, filtered_boxes, filtered_masks, cell_labels, filter_stats = \
        quality_filter_cells(cell_imgs, bounding_boxes, cell_masks, img_shape,
                             border_margin=border_margin, min_area=min_area,
                             max_area=max_area, min_dim=min_dim,
                             min_ar=min_ar, max_ar=max_ar)

    all_features = []
    for idx in range(len(filtered_cells)):
        # Main.py menghasilkan RGB — konversi ke BGR agar konsisten dengan notebook
        cell_rgb = filtered_cells[idx]
        cell_bgr = cv.cvtColor(cell_rgb, cv.COLOR_RGB2BGR)

        feat = extract_all_features(cell_bgr, filtered_masks[idx],
                                    cell_label=cell_labels[idx],
                                    bbox_coords=filtered_boxes[idx])
        if feat is not None:
            all_features.append(feat)

    if len(all_features) == 0:
        df_features = pd.DataFrame()
    else:
        df_features = (pd.DataFrame(all_features)
                       .sort_values("Cell_Label")
                       .reset_index(drop=True))

    if output_csv_path is not None and not df_features.empty:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_features.to_csv(output_csv_path, index=False)

    return df_features, cell_labels, filter_stats
