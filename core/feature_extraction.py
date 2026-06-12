import cv2 as cv
import numpy as np
import pandas as pd
import os
import warnings
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

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


def extract_morphological_features(contour, mask):
    area      = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if len(contour) >= 5:
        try:
            (_, _), (MA, ma), _ = cv.fitEllipse(contour)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except Exception:
            major_axis = minor_axis = 0.0
    else:
        major_axis = minor_axis = 0.0

    compactness  = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0

    hull      = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0.0

    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    x, y, w, h = cv.boundingRect(contour)
    bbox_area      = w * h
    rectangularity = area / bbox_area if bbox_area > 0 else 0.0

    return {
        "Area": area, "Perimeter": perimeter,
        "Major_Axis": major_axis, "Minor_Axis": minor_axis,
        "Compactness": compactness, "Eccentricity": eccentricity,
        "Solidity": solidity, "Aspect_Ratio": aspect_ratio,
        "Rectangularity": rectangularity,
    }


def extract_central_pallor_features(cell_img_bgr, cell_mask):

    lab   = cv.cvtColor(cell_img_bgr, cv.COLOR_BGR2LAB)
    L_ch  = lab[:, :, 0]

    l_pix_in = L_ch[cell_mask == 255]
    cp_mask  = np.zeros(cell_mask.shape, dtype=np.uint8)
    if len(l_pix_in) > 0:
        thresh_cp = float(np.percentile(l_pix_in, 75))
        _, cp_raw = cv.threshold(L_ch, thresh_cp, 255, cv.THRESH_BINARY)
        cp_mask   = cv.bitwise_and(cp_raw, cp_raw, mask=cell_mask)

    cp_contours, _ = cv.findContours(cp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cp_contours) > 0:
        cp_cnt       = max(cp_contours, key=cv.contourArea)
        cp_area      = cv.contourArea(cp_cnt)
        cp_perimeter = cv.arcLength(cp_cnt, True)

        if len(cp_cnt) >= 5:
            try:
                (_, _), (MA, ma), _ = cv.fitEllipse(cp_cnt)
                cp_major = max(MA, ma)
                cp_minor = min(MA, ma)
            except Exception:
                cp_major = cp_minor = 0.0
        else:
            cp_major = cp_minor = 0.0

        cp_compactness  = (cp_perimeter ** 2) / (4 * np.pi * cp_area) if cp_area > 0 else 0.0
        cp_eccentricity = np.sqrt(1 - (cp_minor / cp_major) ** 2) if cp_major > 0 else 0.0
        cp_hull         = cv.convexHull(cp_cnt)
        cp_hull_area    = cv.contourArea(cp_hull)
        cp_solidity     = cp_area / cp_hull_area if cp_hull_area > 0 else 0.0

        cell_contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cell_area = cv.contourArea(max(cell_contours, key=cv.contourArea)) if cell_contours else 0
        cp_ratio  = cp_area / cell_area if cell_area > 0 else 0.0
    else:
        cp_area = cp_perimeter = cp_major = cp_minor = 0.0
        cp_compactness = cp_eccentricity = cp_solidity = cp_ratio = 0.0

    return {
        "CP_Area": cp_area, "CP_Perimeter": cp_perimeter,
        "CP_Major_Axis": cp_major, "CP_Minor_Axis": cp_minor,
        "CP_Compactness": cp_compactness, "CP_Eccentricity": cp_eccentricity,
        "CP_Solidity": cp_solidity, "CP_Ratio": cp_ratio,
    }


def extract_glcm_features(cell_img_bgr, cell_mask):
    gray = cv.cvtColor(cell_img_bgr, cv.COLOR_BGR2GRAY)

    distances = [1]
    angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        glcm  = graycomatrix(gray, distances=distances, angles=angles,
                             levels=256, symmetric=True, normed=True)
        contrast_g    = graycoprops(glcm, "contrast")[0]
        correlation_g = graycoprops(glcm, "correlation")[0]
        energy_g      = graycoprops(glcm, "energy")[0]
        homogeneity_g = graycoprops(glcm, "homogeneity")[0]

        features = {
            "GLCM_Contrast_Mean":    float(np.mean(contrast_g)),
            "GLCM_Correlation_Mean": float(np.mean(correlation_g)),
            "GLCM_Energy_Mean":      float(np.mean(energy_g)),
            "GLCM_Homogeneity_Mean": float(np.mean(homogeneity_g)),
        }
    except Exception:
        features = {
            "GLCM_Contrast_Mean": 0.0, "GLCM_Correlation_Mean": 0.0,
            "GLCM_Energy_Mean":   0.0, "GLCM_Homogeneity_Mean": 0.0,
        }

    return features


def extract_color_moment_features(cell_img_bgr, cell_mask):

    b_ch, g_ch, r_ch = cv.split(cell_img_bgr)
    mask_px = cell_mask == 255

    features = {}
    for ch_name, ch_arr in [("R", r_ch), ("G", g_ch), ("B", b_ch)]:
        pixels = ch_arr[mask_px]
        if len(pixels) > 0:
            features[f"Color_Mean_{ch_name}"]     = float(np.mean(pixels))
            features[f"Color_Std_{ch_name}"]      = float(np.std(pixels))
            features[f"Color_Skewness_{ch_name}"] = float(skew(pixels))
            features[f"Color_Kurtosis_{ch_name}"] = float(kurtosis(pixels))
        else:
            for moment in ["Mean", "Std", "Skewness", "Kurtosis"]:
                features[f"Color_{moment}_{ch_name}"] = 0.0

    return features


def extract_all_features(cell_img_bgr, cell_mask, cell_label, bbox_coords=None):

    contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour  = max(contours, key=cv.contourArea)
    features = {"Cell_Label": cell_label}

    if bbox_coords is not None:
        features["X"], features["Y"] = bbox_coords[0], bbox_coords[1]
    else:
        features["X"], features["Y"] = 0, 0

    features.update(extract_morphological_features(contour, cell_mask))
    features.update(extract_central_pallor_features(cell_img_bgr, cell_mask))
    features.update(extract_glcm_features(cell_img_bgr, cell_mask))
    features.update(extract_color_moment_features(cell_img_bgr, cell_mask))

    return features


def run_feature_extraction(extracted_cells, bounding_boxes, cell_masks, img_shape,
                           output_csv_path=None, border_margin=5,
                           min_area=200, max_area=5000, min_dim=15,
                           min_ar=0.6, max_ar=1.6):

    cell_imgs = [item[0] if isinstance(item, tuple) else item for item in extracted_cells]

    filtered_cells, filtered_boxes, filtered_masks, cell_labels, filter_stats = \
        quality_filter_cells(cell_imgs, bounding_boxes, cell_masks, img_shape,
                             border_margin=border_margin, min_area=min_area,
                             max_area=max_area, min_dim=min_dim,
                             min_ar=min_ar, max_ar=max_ar)

    all_features = []
    for idx in range(len(filtered_cells)):
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
