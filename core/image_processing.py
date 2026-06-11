import cv2 as cv
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.feature import peak_local_max

#Parameter Preprocessing
MEDIAN_KERNEL = 3
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
LOG_SIGMA = 1.5
LOG_KERNEL_SIZE = 5
LOG_ALPHA = 0.5

#Parameter Segmentasi
K = 6
N_TOP_CLUSTERS = 2
BG_LUMA_FLOOR = 30
MORPH_CLOSE_K = 5
MORPH_OPEN_K = 5
MIN_AREA = 300
ITERASI_BO = 4
FRS_RADII = [15, 20, 25, 30, 35]

#Tahapan Preprocessing
def apply_median_filter(img_rgb, kernel_size=MEDIAN_KERNEL):
    return cv.medianBlur(img_rgb, kernel_size)

def reinhard_normalization(Source, Target, epsilon=1e-6):
    src = cv.cvtColor(Source, cv.COLOR_RGB2LAB).astype(float)
    tgt = cv.cvtColor(Target, cv.COLOR_RGB2LAB).astype(float)
    result = []
    for i in range(3):
        src_channel = src[:, :, i]
        tgt_channel = tgt[:, :, i]

        src_mean, src_std = np.mean(src_channel), np.std(src_channel)
        tgt_mean, tgt_std = np.mean(tgt_channel), np.std(tgt_channel)

        normalized = (src_channel - src_mean) * (tgt_std / (src_std + epsilon)) + tgt_mean
        result.append(normalized)

    merged = np.clip(cv.merge(result), 0, 255).astype(np.uint8)
    return cv.cvtColor(merged, cv.COLOR_LAB2RGB)

def apply_clahe(img_rgb, clip_limit=CLAHE_CLIP, tile_grid_size=CLAHE_TILE):
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_cl = clahe.apply(L)
    lab_cl = cv.merge((L_cl, A, B))
    return cv.cvtColor(lab_cl, cv.COLOR_LAB2RGB)

def build_log_kernel(size=LOG_KERNEL_SIZE, sigma=LOG_SIGMA):
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1].astype(np.float64)
    r2 = x**2 + y**2
    kernel = -(1.0 / (np.pi * sigma**4)) * \
             (1 - r2 / (2 * sigma**2)) * \
             np.exp(-r2 / (2 * sigma**2))
    kernel -= kernel.mean()   
    return kernel
    
def apply_log_enhancement(img_clahe, sigma=LOG_SIGMA, kernel_size=LOG_KERNEL_SIZE, alpha=LOG_ALPHA):
    img_f = img_clahe.astype(np.float32) / 255.0
    log_k = build_log_kernel(size=kernel_size, sigma=sigma).astype(np.float32)
    sharpened = np.zeros_like(img_f)
    for i in range(3):
        response = cv.filter2D(img_f[:, :, i], cv.CV_32F, log_k)
        sharpened[:, :, i] = img_f[:, :, i] + alpha * response
    return np.clip(sharpened, 0.0, 1.0)

def preprocess_image(img_rgb, ref_img_rgb=None):
    # 1. Median Filter
    img = apply_median_filter(img_rgb)
    
    # 2. Reinhard Normalization (Jika ada gambar referensi)
    if ref_img_rgb is not None:
        img_norm = reinhard_normalization(img, ref_img_rgb)
    else:
        img_norm = img
        
    # 3. CLAHE
    img = apply_clahe(img_norm)
    
    # 4. LoG Enhancement
    img_f = apply_log_enhancement(img_clahe)
    return (img_f * 255).astype(np.uint8)

# Tahapan Segmentasi
def make_fg_mask(image_rgb, luma_floor=BG_LUMA_FLOOR):
    lab = cv.cvtColor(image_rgb, cv.COLOR_RGB2LAB)
    L   = lab[:, :, 0].astype(np.float32)
    L_u8 = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, otsu_mask = cv.threshold(L_u8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    floor_mask = (L > luma_floor)
    fg_2d = (otsu_mask == 255) & floor_mask
    return fg_2d.flatten()

def kmeans_segmentation(image_rgb, k=K, ref_img_rgb=None):
    processed_img = preprocess_image(image_rgb, ref_img_rgb=ref_img_rgb)
 
    h, w = processed_img.shape[:2]
    lab  = cv.cvtColor(processed_img, cv.COLOR_RGB2LAB).astype(np.float32)
    ab   = lab[:, :, 1:3].reshape(-1, 2)     # a* and b* channels only
 
    fg_mask = make_fg_mask(processed_img)     # foreground only
 
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(ab[fg_mask])                       # fit on foreground pixels
 
    labels_flat        = km.predict(ab).astype(np.int32)
    labels_2d          = labels_flat.reshape(h, w)
    labels_2d[~fg_mask.reshape(h, w)] = -1   # mask out background
 
    seg_images = []
    for i in range(k):
        seg = np.zeros_like(processed_img)
        seg[labels_2d == i] = processed_img[labels_2d == i]
        seg_images.append(seg)
 
    return labels_2d, km.cluster_centers_, seg_images

def build_binary_mask(labels_2d, selected, min_area=MIN_AREA):
    h, w = labels_2d.shape
 
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in selected:
        raw_mask[labels_2d == idx] = 255
 
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_OPEN_K,  MORPH_OPEN_K))
 
    mask = cv.morphologyEx(raw_mask, cv.MORPH_CLOSE, k_close)
    mask = cv.morphologyEx(mask,     cv.MORPH_OPEN,  k_open)
 
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for cnt in contours:
        if cv.contourArea(cnt) >= min_area:
            cv.drawContours(cleaned, [cnt], -1, 255, cv.FILLED)
 
    return raw_mask, cleaned

def remove_unwanted_cells_extended(clustered_images, selected_cluster, original_image):
    if not selected_cluster: raise ValueError("No clusters selected.")
    segmented_mask = clustered_images[selected_cluster[0]].copy()
    h, w = original_image.shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in selected_cluster:
        seg = clustered_images[idx]
        gray = cv.cvtColor(seg, cv.COLOR_RGB2GRAY)
        _, m = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        binary_mask = cv.bitwise_or(binary_mask, m)
 
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_OPEN_K,  MORPH_OPEN_K))
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, k_close)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN,  k_open)
 
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(binary_mask)
    for contour in contours:
        if cv.contourArea(contour) >= MIN_AREA:
            cv.drawContours(filtered_mask, [contour], -1, 255, thickness=cv.FILLED)
 
    rbc_only_image = cv.bitwise_and(original_image, original_image, mask=filtered_mask)
    for c in range(rbc_only_image.shape[2]):
        _, rbc_only_image[:, :, c] = cv.threshold(
            rbc_only_image[:, :, c], 15, 255, cv.THRESH_TOZERO)
 
    return rbc_only_image, filtered_mask, binary_mask

def detect_staining_level(image_rgb, fg_mask=None):
    lab = cv.cvtColor(image_rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    a_channel = lab[:, :, 1]
    if fg_mask is None:
        L    = lab[:, :, 0]
        L_u8 = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        _, fg = cv.threshold(L_u8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        fg_mask = fg
    fg_pixels = a_channel[fg_mask > 0]
    
    if len(fg_pixels) == 0:
        fg_pixels = a_channel.flatten()
 
    mean_a = float(np.mean(fg_pixels))
    std_a  = float(np.std(fg_pixels))
 
    if mean_a > 143 or (mean_a > 138 and std_a > 12):
        staining = "stained"
    else:
        staining = "unstained"
 
    return staining, mean_a, std_a

def fill_cell_holes(mask):
    filled = mask.copy()
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(filled, [cnt], -1, 255, thickness=cv.FILLED)
    return filled

def bounded_opening(mask, n_iter=ITERASI_BO):
    se     = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    result = mask.copy()
    for _ in range(n_iter):
        result = cv.morphologyEx(result, cv.MORPH_OPEN, se)
    return result

def bounded_opening_frs(binary_mask, num_openings=ITERASI_BO):
    staining, _, _ = detect_staining_level( cv.cvtColor(binary_mask, cv.COLOR_GRAY2RGB), fg_mask=binary_mask)
    if staining == "unstained":
        filled_mask = fill_cell_holes(binary_mask)
    else:
        filled_mask = binary_mask
 
    # Bounded Opening 
    opened_mask = bounded_opening(filled_mask, n_iter=num_openings)
 
    # Distance Transform 
    dist_transform = cv.distanceTransform(opened_mask, cv.DIST_L2, 5)
    dist_norm      = cv.normalize(dist_transform, None, 0.0, 1.0, cv.NORM_MINMAX)
 
    # Gradient for FRS votes 
    gx      = cv.Sobel(dist_norm, cv.CV_64F, 1, 0, ksize=3)
    gy      = cv.Sobel(dist_norm, cv.CV_64F, 0, 1, ksize=3)
    gmag    = np.sqrt(gx**2 + gy**2)
    orient  = np.arctan2(gy, gx)
    ys, xs  = np.where(opened_mask > 0)
 
    # FRS accumulation
    frs_maps = []
    for radius in FRS_RADII:
        smap = np.zeros_like(dist_norm)
        if len(ys):
            angles       = orient[ys, xs]
            grads        = gmag[ys, xs]
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            for sign in (1, -1):
                px = np.clip((xs + sign * radius * cos_a).astype(int), 0, smap.shape[1]-1)
                py = np.clip((ys + sign * radius * sin_a).astype(int), 0, smap.shape[0]-1)
                np.add.at(smap, (py, px), grads)
        frs_maps.append(smap)
 
    frs_combined = cv.normalize(
        np.mean(frs_maps, axis=0), None, 0.0, 1.0, cv.NORM_MINMAX)
    combined_map = 0.6 * dist_norm + 0.4 * frs_combined
 
    # deteksi kasar -> estimasi nilai adaptif minimum 
    rough_coords = peak_local_max(combined_map,
                                  min_distance=8,
                                  threshold_abs=0.08,
                                  exclude_border=False)
    rough_radii_vals = [dist_transform[y, x]
                        for y, x in rough_coords
                        if 0 <= x < dist_transform.shape[1]
                        and 0 <= y < dist_transform.shape[0]]
 
    if rough_radii_vals:
        rough_median_r    = float(np.median(rough_radii_vals))
        adaptive_min_dist = max(8, int(rough_median_r * 1.2))
    else:
        adaptive_min_dist = 15
 
    # membentuk ulang peaks dengan adaptive suppresion 
    coords  = peak_local_max(combined_map,
                             min_distance=adaptive_min_dist,
                             threshold_abs=0.10,
                             exclude_border=False)
    centers = [(int(x), int(y)) for y, x in coords]
 
    # Nilai kandidat radius dari DT  
    radii_list = [dist_transform[cy, cx]
                  for cx, cy in centers
                  if 0 <= cx < dist_transform.shape[1]
                  and 0 <= cy < dist_transform.shape[0]]
    candidate_radius = int(np.median(radii_list)) if radii_list else 15
    radius_std       = float(np.std(radii_list))  if radii_list else 0.0
 
    # Nilai Tengah
    center_map = np.zeros_like(opened_mask)
    for cx, cy in centers:
        cv.circle(center_map, (cx, cy), 3, 255, -1)
 
    return {
        'refined_mask':     opened_mask,
        'dist_transform':   dist_transform,
        'frs_map':          frs_combined,
        'combined_map':     combined_map,
        'centers':          centers,
        'center_map':       center_map,
        'candidate_radius': candidate_radius,
        'radius_std':       radius_std,
    }

def suppress_nearby_centers(local_centers, min_dist):
    if not local_centers:
        return []
    kept       = []
    suppressed = set()
    for i, c1 in enumerate(local_centers):
        if i in suppressed:
            continue
        kept.append(c1)
        for j, c2 in enumerate(local_centers):
            if j <= i or j in suppressed:
                continue
            if np.hypot(c1[0]-c2[0], c1[1]-c2[1]) < min_dist:
                suppressed.add(j)
    return kept

def estimate_k_from_area(contour_area, candidate_radius):
    single_cell_area = np.pi * (candidate_radius ** 2)
    return max(1, min(6, round(contour_area / single_cell_area)))

def separate_overlapping_rbc_with_gmm(bofrs_results, cells_image):
    centers_global = bofrs_results['centers']
    dist_transform = bofrs_results['dist_transform']
    refined_mask = bofrs_results['refined_mask']
    candidate_radius = bofrs_results['candidate_radius']
    
    MAX_REPLICATIONS = max(50, int(candidate_radius * 2))
    SUPPRESSION_DIST = candidate_radius * 0.8
 
    all_cropped_cells  = []
    all_bounding_boxes = []
    all_cell_masks     = []
 
    contours, _ = cv.findContours(refined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
 
    for idx, contour in enumerate(contours):
        x_off, y_off, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
 
        if w < 15 or h < 15 or area < MIN_AREA:
            continue
 
        # Local crops
        single_mask   = np.zeros_like(refined_mask)
        cv.drawContours(single_mask, [contour], -1, 255, thickness=cv.FILLED)
        cropped_mask  = single_mask[y_off:y_off+h, x_off:x_off+w]
        cropped_image = cells_image[y_off:y_off+h, x_off:x_off+w]
        cropped_dist  = dist_transform[y_off:y_off+h, x_off:x_off+w]
 
        # Collect + NMS on local FRS centres
        raw_local = [(cx - x_off, cy - y_off)
            for (cx, cy) in centers_global
            if x_off <= cx < x_off + w and y_off <= cy < y_off + h
        ]
        local_centers = suppress_nearby_centers(raw_local, SUPPRESSION_DIST)
 
        # No FRS centre -> area-based fallback
        if len(local_centers) == 0:
            k_fb = estimate_k_from_area(area, candidate_radius)
            if k_fb == 1:
                all_bounding_boxes.append((x_off, y_off, w, h))
                all_cropped_cells.append(cropped_image)
                all_cell_masks.append(cropped_mask)
                continue
            local_centers = [
                (int(w * (i+1) / (k_fb+1)), int(h * (i+1) / (k_fb+1)))
                for i in range(k_fb)
            ]
 
        # Single centre → no GMM needed
        if len(local_centers) == 1:
            all_bounding_boxes.append((x_off, y_off, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            continue
 
        #  Overlapping: Pixel Replication + GMM 
        k = len(local_centers)
 
        # Replikasi semua pixels foreground dengan nilai DT
        ys_fg, xs_fg = np.where(cropped_mask == 255)
        X_replicated = []
        for px, py in zip(xs_fg, ys_fg):
            reps = max(1, min(int(cropped_dist[py, px]), MAX_REPLICATIONS))
            X_replicated.extend([(px, py)] * reps)
 
        if len(X_replicated) < k * 10:
            all_bounding_boxes.append((x_off, y_off, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            continue
 
        try:
            # covariance_type='full', n_init=5
            means_init = np.array(local_centers, dtype=np.float64)
            gmm = GaussianMixture(
                n_components    = k,
                covariance_type = 'full',   
                max_iter        = 100,
                n_init          = 5,        
                means_init      = means_init,
                random_state    = 42,
                tol             = 1e-4,
            )
            gmm.fit(np.array(X_replicated, dtype=np.float64))
 
            ys_fg2, xs_fg2 = np.where(cropped_mask == 255)
            if len(ys_fg2) == 0:
                continue
            fg_pixels    = np.column_stack((xs_fg2, ys_fg2)).astype(np.float64)
            pixel_labels = gmm.predict(fg_pixels)
 
            labeled_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
            for (px2, py2), lbl in zip(fg_pixels.astype(int), pixel_labels):
                labeled_mask[py2, px2] = int(lbl) + 1
 
            for lbl in np.unique(labeled_mask):
                if lbl == 0:
                    continue
                cell_mask  = (labeled_mask == lbl).astype(np.uint8) * 255
                cell_image = cv.bitwise_and(cropped_image, cropped_image, mask=cell_mask)
                coords     = cv.findNonZero(cell_mask)
                if coords is not None and len(coords) > 50:
                    xc, yc, wc, hc = cv.boundingRect(coords)
                    all_bounding_boxes.append((xc+x_off, yc+y_off, wc, hc))
                    all_cropped_cells.append(cell_image[yc:yc+hc, xc:xc+wc])
                    all_cell_masks.append(cell_mask[yc:yc+hc, xc:xc+wc])
 
        except Exception:
            all_bounding_boxes.append((x_off, y_off, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
 
    return all_cropped_cells, all_bounding_boxes, all_cell_masks

def convert_hsv_circular(image_rgb, v_thresh=20):
    hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    v = hsv_image[:, :, 2]
    mask = v > v_thresh
    return hsv_image, mask

def sobel_edge_detect(image):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(255 * (sobel_edges / np.max(sobel_edges)))
    _, sobel_binary = cv.threshold(sobel_edges, 50, 255, cv.THRESH_BINARY)
    contours_sobel, _ = cv.findContours(sobel_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return sobel_edges, contours_sobel

def draw_bounding_boxes(image, contours):
    bbox_image = image.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return bbox_image

def extract_contours(image, edge_map):
    contours, _ = cv.findContours(edge_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(image)
    cv.drawContours(contour_mask, contours, -1, 255, thickness=cv.FILLED)
    return contours, contour_mask
