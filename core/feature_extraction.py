# ── Helper (tambahkan jika belum ada) ──────────────────────────────────────────
def _fill_holes(mask):
    h, w   = mask.shape
    canvas = mask.copy()
    border = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(canvas, border, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(canvas))

NORM_TARGET_SIZE = 96
NORM_PAD_FRAC    = 0.12

def ekstrak_fitur_satu_sel(crop_bgr, label=0):
    
    cell_mask, found = isolate_target_cell(crop_bgr)
    crop_bgr, cell_mask, norm_ok = normalize_cell_crop(
        crop_bgr, cell_mask, target_size=NORM_TARGET_SIZE, pad_frac=NORM_PAD_FRAC
    )
    h, w = crop_bgr.shape[:2]

    lab  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L_ch = lab[:, :, 0]
    b_ch, g_ch, r_ch = cv2.split(crop_bgr)

    area = perimeter = maj_ax = min_ax = 0
    compactness = eccentricity = solidity = aspect_ratio = 0
    rectangularity = convexity = circularity_ratio = euler_number = 0

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c         = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        _, _, wc, hc = cv2.boundingRect(c)
        rectangularity = area / (wc * hc) if wc * hc > 0 else 0
        if len(c) >= 5:
            (_, _), (min_ax, maj_ax), _ = cv2.fitEllipse(c)
            aspect_ratio = maj_ax / min_ax if min_ax > 0 else 0
            eccentricity = (
                np.sqrt(1 - (min_ax**2 / maj_ax**2))
                if maj_ax > min_ax else 0
            )
        hull       = cv2.convexHull(c)
        hull_area  = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(hull, True)
        solidity   = area / hull_area if hull_area > 0 else 0
        convexity  = hull_perim / perimeter if perimeter > 0 else 0
        compactness       = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
        circularity_ratio = area / (perimeter**2) if perimeter > 0 else 0

    mask_px = cell_mask == 255
    r_pix   = r_ch[mask_px]
    g_pix   = g_ch[mask_px]
    b_pix   = b_ch[mask_px]
    if len(r_pix) == 0:
        r_pix = g_pix = b_pix = np.array([0], dtype=np.uint8)

    c_mean_r, c_std_r  = float(np.mean(r_pix)), float(np.std(r_pix))
    c_skew_r, c_kurt_r = float(skew(r_pix)),    float(kurtosis(r_pix))
    c_mean_g, c_std_g  = float(np.mean(g_pix)), float(np.std(g_pix))
    c_skew_g, c_kurt_g = float(skew(g_pix)),    float(kurtosis(g_pix))
    c_mean_b, c_std_b  = float(np.mean(b_pix)), float(np.std(b_pix))
    c_skew_b, c_kurt_b = float(skew(b_pix)),    float(kurtosis(b_pix))

    cp_mask = np.zeros((h, w), dtype=np.uint8)

    filled     = _fill_holes(cell_mask)
    lumen_mask = cv2.subtract(filled, cell_mask)
    lumen_area = int(np.sum(lumen_mask == 255))
    cell_area_now = int(np.sum(cell_mask == 255))

    if cell_area_now > 0 and lumen_area / cell_area_now > 0.05:
        cp_mask = lumen_mask                                    
    else:
        l_pix_in = L_ch[mask_px]
        if len(l_pix_in) > 0:
            thresh_cp = float(np.percentile(l_pix_in, 75))
            _, cp_raw = cv2.threshold(L_ch, thresh_cp, 255, cv2.THRESH_BINARY)
            cp_raw_masked = cv2.bitwise_and(cp_raw, cp_raw, mask=cell_mask)
            cp_conts, _ = cv2.findContours(
                cp_raw_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cp_conts:
                largest = max(cp_conts, key=cv2.contourArea)
                cv2.drawContours(cp_mask, [largest], -1, 255, -1)

    cp_contours, _ = cv2.findContours(cp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    euler_number = 1 - sum(1 for cc in cp_contours if cv2.contourArea(cc) > 10)

    cp_area = cp_perim = cp_maj = cp_min = cp_comp = cp_ecc = cp_solid = cp_ratio = 0
    if cp_contours:
        c_cp     = max(cp_contours, key=cv2.contourArea)
        cp_area  = cv2.contourArea(c_cp)
        cp_perim = cv2.arcLength(c_cp, True)
        cp_ratio = cp_area / area if area > 0 else 0
        if len(c_cp) >= 5:
            _, (cp_min, cp_maj), _ = cv2.fitEllipse(c_cp)
            cp_ecc = (
                np.sqrt(1 - (cp_min**2 / cp_maj**2))
                if cp_maj > cp_min else 0
            )
        hull_cp      = cv2.convexHull(c_cp)
        hull_cp_area = cv2.contourArea(hull_cp)
        cp_solid = cp_area / hull_cp_area if hull_cp_area > 0 else 0
        cp_comp  = (cp_perim**2) / (4 * np.pi * cp_area) if cp_area > 0 else 0

    pallor_px       = cp_mask == 255
    rim_px          = mask_px & (cp_mask == 0)
    r_pallor        = float(r_ch[pallor_px].mean()) if pallor_px.any() else 0.0
    r_rim           = float(r_ch[rim_px].mean())    if rim_px.any()    else 0.0
    pallor_contrast = r_pallor - r_rim
    pallor_ratio_r  = r_pallor / r_rim if r_rim > 0 else 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray, distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )
    contrast_g    = graycoprops(glcm, 'contrast')[0]
    correlation_g = graycoprops(glcm, 'correlation')[0]
    energy_g      = graycoprops(glcm, 'energy')[0]
    homogeneity_g = graycoprops(glcm, 'homogeneity')[0]

    return {
        "Cell_Label"            : label,
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
    }, cp_mask, cell_mask
