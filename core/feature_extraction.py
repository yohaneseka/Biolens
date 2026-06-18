import cv2 as cv
import numpy as np
import pandas as pd
import os
import warnings
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

NORM_TARGET_SIZE = 96
NORM_PAD_FRAC    = 0.12

def _fill_holes(mask):
    h, w   = mask.shape
    canvas = mask.copy()
    border = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv.floodFill(canvas, border, (0, 0), 255)
    return cv.bitwise_or(mask, cv.bitwise_not(canvas))


def normalize_cell_crop(crop_bgr, cell_mask, target_size=NORM_TARGET_SIZE,
                        pad_frac=NORM_PAD_FRAC):
    h, w   = cell_mask.shape[:2]
    ys, xs = np.where(cell_mask == 255)
    if len(ys) == 0:
        crop_out = cv.resize(crop_bgr,  (target_size, target_size), interpolation=cv.INTER_LINEAR)
        mask_out = cv.resize(cell_mask, (target_size, target_size), interpolation=cv.INTER_NEAREST)
        return crop_out, mask_out, False
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh, bw = max(y1 - y0, 1), max(x1 - x0, 1)
    pad    = int(max(bh, bw) * pad_frac)
    sy0 = max(0, y0 - pad);  sy1 = min(h, y1 + pad + 1)
    sx0 = max(0, x0 - pad);  sx1 = min(w, x1 + pad + 1)
    crop_out = cv.resize(crop_bgr[sy0:sy1, sx0:sx1],
                         (target_size, target_size), interpolation=cv.INTER_LINEAR)
    mask_out = cv.resize(cell_mask[sy0:sy1, sx0:sx1],
                         (target_size, target_size), interpolation=cv.INTER_NEAREST)
    return crop_out, mask_out, True


def isolate_target_cell(crop_bgr):
    h, w  = crop_bgr.shape[:2]
    lab   = cv.cvtColor(crop_bgr, cv.COLOR_BGR2LAB)
    L_ch  = lab[:, :, 0]
    _, raw_mask = cv.threshold(L_ch, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(raw_mask, cv.MORPH_CLOSE, k)
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        closed, connectivity=8)
    best_label, best_dist = -1, float('inf')
    cx_c, cy_c = w / 2.0, h / 2.0
    for lbl in range(1, n_labels):
        if stats[lbl, cv.CC_STAT_AREA] < 200:
            continue
        dist = np.hypot(centroids[lbl][0] - cx_c, centroids[lbl][1] - cy_c)
        if dist < best_dist:
            best_dist, best_label = dist, lbl
    if best_label == -1:
        return np.zeros((h, w), dtype=np.uint8), False
    return np.where(labels == best_label, 255, 0).astype(np.uint8), True


def quality_filter_cells(cropped_cells, bounding_boxes, cell_masks, img_shape,
                         border_margin=5, min_area=200, max_area=5000,
                         min_dim=15, min_ar=0.6, max_ar=1.6):
    img_h, img_w = img_shape[:2]
    filtered_cells, filtered_boxes, filtered_masks, cell_labels = [], [], [], []
    stats = {"border": 0, "small": 0, "large": 0, "shape": 0, "dim": 0, "passed": 0}
    for idx in range(len(cropped_cells)):
        x, y, w, h = bounding_boxes[idx]
        if (x <= border_margin or y <= border_margin or
                x + w >= img_w - border_margin or y + h >= img_h - border_margin):
            stats["border"] += 1; continue
        if w * h < min_area: stats["small"] += 1; continue
        if w * h > max_area: stats["large"] += 1; continue
        ar = w / h if h > 0 else 0
        if ar < min_ar or ar > max_ar: stats["shape"] += 1; continue
        if w < min_dim or h < min_dim:  stats["dim"]   += 1; continue
        filtered_cells.append(cropped_cells[idx])
        filtered_boxes.append(bounding_boxes[idx])
        filtered_masks.append(cell_masks[idx])
        cell_labels.append(idx + 1)
        stats["passed"] += 1
    return filtered_cells, filtered_boxes, filtered_masks, cell_labels, stats


FEATURE_COLUMNS = [
    "Area", "Perimeter", "Major_Axis", "Minor_Axis", "Compactness",
    "Eccentricity", "Solidity", "Aspect_Ratio", "Rectangularity",
    "Convexity", "Circularity_Ratio", "Euler_Number",
    "CP_Area", "CP_Perimeter", "CP_Major_Axis", "CP_Minor_Axis",
    "CP_Compactness", "CP_Eccentricity", "CP_Solidity", "CP_Ratio",
    "Pallor_Contrast_R", "Pallor_Ratio_R",
    "GLCM_Contrast_Mean", "GLCM_Correlation_Mean",
    "GLCM_Energy_Mean", "GLCM_Homogeneity_Mean",
    "Color_Mean_R", "Color_Std_R", "Color_Skewness_R", "Color_Kurtosis_R",
    "Color_Mean_G", "Color_Std_G", "Color_Skewness_G", "Color_Kurtosis_G",
    "Color_Mean_B", "Color_Std_B", "Color_Skewness_B", "Color_Kurtosis_B",
]


def extract_all_features(cell_img_bgr, cell_mask_external, cell_label,
                         bbox_coords=None):
                             
    cell_mask_raw, found = isolate_target_cell(cell_img_bgr)
    if not found:
        cell_mask_raw = cell_mask_external if cell_mask_external is not None \
                        else np.zeros(cell_img_bgr.shape[:2], dtype=np.uint8)

    crop_bgr, cell_mask, _ = normalize_cell_crop(cell_img_bgr, cell_mask_raw)
    h, w = crop_bgr.shape[:2]

    lab  = cv.cvtColor(crop_bgr, cv.COLOR_BGR2LAB)
    L_ch = lab[:, :, 0]
    b_ch, g_ch, r_ch = cv.split(crop_bgr)

    area = perimeter = maj_ax = min_ax = 0.0
    compactness = eccentricity = solidity = aspect_ratio = rectangularity = 0.0
    convexity = circularity_ratio = 0.0

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
                eccentricity = (np.sqrt(1 - (min_ax**2 / maj_ax**2))
                                if maj_ax > min_ax else 0.0)
            except Exception:
                pass
        hull       = cv.convexHull(c)
        hull_area  = cv.contourArea(hull)
        hull_perim = cv.arcLength(hull, True)
        solidity        = area / hull_area if hull_area > 0 else 0.0
        convexity       = hull_perim / perimeter if perimeter > 0 else 0.0
        compactness     = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0.0
        circularity_ratio = area / (perimeter**2) if perimeter > 0 else 0.0

    mask_px = cell_mask == 255
    r_pix = r_ch[mask_px]; g_pix = g_ch[mask_px]; b_pix = b_ch[mask_px]
    if len(r_pix) == 0:
        r_pix = g_pix = b_pix = np.array([0], dtype=np.uint8)

    c_mean_r, c_std_r  = float(np.mean(r_pix)), float(np.std(r_pix))
    c_skew_r, c_kurt_r = float(skew(r_pix)),    float(kurtosis(r_pix))
    c_mean_g, c_std_g  = float(np.mean(g_pix)), float(np.std(g_pix))
    c_skew_g, c_kurt_g = float(skew(g_pix)),    float(kurtosis(g_pix))
    c_mean_b, c_std_b  = float(np.mean(b_pix)), float(np.std(b_pix))
    c_skew_b, c_kurt_b = float(skew(b_pix)),    float(kurtosis(b_pix))

    cp_mask = np.zeros((h, w), dtype=np.uint8)
    filled        = _fill_holes(cell_mask)
    lumen_mask    = cv.subtract(filled, cell_mask)
    lumen_area    = int(np.sum(lumen_mask == 255))
    cell_area_now = int(np.sum(cell_mask == 255))

    if cell_area_now > 0 and lumen_area / cell_area_now > 0.05:
        cp_mask = lumen_mask
    else:
        l_pix_in = L_ch[mask_px]
        if len(l_pix_in) > 0:
            thresh_cp = float(np.percentile(l_pix_in, 75))
            _, cp_raw = cv.threshold(L_ch, thresh_cp, 255, cv.THRESH_BINARY)
            cp_raw_m  = cv.bitwise_and(cp_raw, cp_raw, mask=cell_mask)
            cp_conts, _ = cv.findContours(
                cp_raw_m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if cp_conts:
                largest = max(cp_conts, key=cv.contourArea)
                cv.drawContours(cp_mask, [largest], -1, 255, -1)

    cp_contours, _ = cv.findContours(cp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    euler_number   = 1 - sum(1 for cc in cp_contours if cv.contourArea(cc) > 10)

    cp_area = cp_perim = cp_maj = cp_min = 0.0
    cp_comp = cp_ecc = cp_solid = cp_ratio = 0.0
    if cp_contours:
        c_cp     = max(cp_contours, key=cv.contourArea)
        cp_area  = cv.contourArea(c_cp)
        cp_perim = cv.arcLength(c_cp, True)
        cp_ratio = cp_area / area if area > 0 else 0.0
        if len(c_cp) >= 5:
            try:
                _, (cp_min, cp_maj), _ = cv.fitEllipse(c_cp)
                cp_ecc = (np.sqrt(1 - (cp_min**2 / cp_maj**2))
                          if cp_maj > cp_min else 0.0)
            except Exception:
                pass
        hull_cp      = cv.convexHull(c_cp)
        hull_cp_area = cv.contourArea(hull_cp)
        cp_solid = cp_area / hull_cp_area if hull_cp_area > 0 else 0.0
        cp_comp  = (cp_perim**2) / (4 * np.pi * cp_area) if cp_area > 0 else 0.0

    pallor_px       = cp_mask == 255
    rim_px          = mask_px & (cp_mask == 0)
    r_pallor        = float(r_ch[pallor_px].mean()) if pallor_px.any() else 0.0
    r_rim           = float(r_ch[rim_px].mean())    if rim_px.any()    else 0.0
    pallor_contrast = r_pallor - r_rim
    pallor_ratio_r  = r_pallor / r_rim if r_rim > 0 else 0.0

    gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
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

    return {
        "Cell_Label"            : cell_label,
        "X"                     : bbox_coords[0] if bbox_coords else 0,
        "Y"                     : bbox_coords[1] if bbox_coords else 0,
        "Area"                  : round(area, 2),
        "Perimeter"             : round(perimeter, 2),
        "Major_Axis"            : round(maj_ax, 2),
        "Minor_Axis"            : round(min_ax, 2),
        "Compactness"           : round(compactness, 4),
        "Eccentricity"          : round(eccentricity, 4),
        "Solidity"              : round(solidity, 4),
        "Aspect_Ratio"          : round(aspect_ratio, 4),
        "Rectangularity"        : round(rectangularity, 4),
        "Convexity"             : round(convexity, 4),
        "Circularity_Ratio"     : round(circularity_ratio, 4),
        "Euler_Number"          : euler_number,
        "CP_Area"               : round(cp_area, 2),
        "CP_Perimeter"          : round(cp_perim, 2),
        "CP_Major_Axis"         : round(cp_maj, 2),
        "CP_Minor_Axis"         : round(cp_min, 2),
        "CP_Compactness"        : round(cp_comp, 4),
        "CP_Eccentricity"       : round(cp_ecc, 4),
        "CP_Solidity"           : round(cp_solid, 4),
        "CP_Ratio"              : round(cp_ratio, 4),
        "Pallor_Contrast_R"     : round(pallor_contrast, 4),
        "Pallor_Ratio_R"        : round(pallor_ratio_r, 4),
        "GLCM_Contrast_Mean"    : round(float(np.mean(contrast_g)), 6),
        "GLCM_Correlation_Mean" : round(float(np.mean(correlation_g)), 6),
        "GLCM_Energy_Mean"      : round(float(np.mean(energy_g)), 6),
        "GLCM_Homogeneity_Mean" : round(float(np.mean(homogeneity_g)), 6),
        "Color_Mean_R"          : round(c_mean_r, 4),
        "Color_Std_R"           : round(c_std_r, 4),
        "Color_Skewness_R"      : round(c_skew_r, 4),
        "Color_Kurtosis_R"      : round(c_kurt_r, 4),
        "Color_Mean_G"          : round(c_mean_g, 4),
        "Color_Std_G"           : round(c_std_g, 4),
        "Color_Skewness_G"      : round(c_skew_g, 4),
        "Color_Kurtosis_G"      : round(c_kurt_g, 4),
        "Color_Mean_B"          : round(c_mean_b, 4),
        "Color_Std_B"           : round(c_std_b, 4),
        "Color_Skewness_B"      : round(c_skew_b, 4),
        "Color_Kurtosis_B"      : round(c_kurt_b, 4),
    }


def run_feature_extraction(extracted_cells, bounding_boxes, cell_masks, img_shape,
                           output_csv_path=None, border_margin=5,
                           min_area=200, max_area=5000, min_dim=15,
                           min_ar=0.6, max_ar=1.6):
    cell_imgs = [item[0] if isinstance(item, tuple) else item
                 for item in extracted_cells]

    filtered_cells, filtered_boxes, filtered_masks, cell_labels, filter_stats = \
        quality_filter_cells(cell_imgs, bounding_boxes, cell_masks, img_shape,
                             border_margin=border_margin, min_area=min_area,
                             max_area=max_area, min_dim=min_dim,
                             min_ar=min_ar, max_ar=max_ar)

    all_features = []
    for idx in range(len(filtered_cells)):
        cell_bgr = cv.cvtColor(filtered_cells[idx], cv.COLOR_RGB2BGR)
        feat = extract_all_features(cell_bgr, filtered_masks[idx],
                                    cell_label=cell_labels[idx],
                                    bbox_coords=filtered_boxes[idx])
        if feat is not None:
            all_features.append(feat)

    if not all_features:
        return pd.DataFrame(), cell_labels, filter_stats

    df_features = (pd.DataFrame(all_features)
                   .sort_values("Cell_Label")
                   .reset_index(drop=True))

    if output_csv_path is not None and not df_features.empty:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_features.to_csv(output_csv_path, index=False)

    return df_features, cell_labels, filter_stats
