import os
import time
from fpdf import FPDF

NAVY        = (4, 21, 98)     
NAVY_LIGHT  = (35, 56, 148)    
ACCENT_BLUE = (214, 222, 255)  
RED         = (211, 47, 47)   
RED_BG      = (253, 236, 236)
GREEN       = (43, 138, 62)   
GREEN_BG    = (235, 247, 237)
GRAY_TEXT   = (110, 116, 130)
GRAY_LINE   = (228, 230, 238)
WHITE       = (255, 255, 255)
DARK_TEXT   = (30, 33, 45)


class PDFWithHeaderFooter(FPDF):
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        poppins_path = os.path.join(self.base_dir, "add-on", "Poppins-Bold.ttf")
        inter_path = os.path.join(self.base_dir, "add-on", "Inter_18pt-SemiBold.ttf")
        inter_reg_path = os.path.join(self.base_dir, "add-on", "Inter_18pt-Regular.ttf")

        if os.path.exists(poppins_path) and os.path.exists(inter_path):
            self.add_font("Poppins", "", poppins_path, uni=True)
            self.add_font("Inter", "", inter_path, uni=True)
            self.font_title = "Poppins"
            self.font_body = "Inter"
            
            if os.path.exists(inter_reg_path):
                self.add_font("Inter-Reg", "", inter_reg_path, uni=True)
                self.font_reg = "Inter-Reg"
            else:
                self.font_reg = "Inter"
        else:
            print("⚠️ Peringatan: File font TTF tidak ditemukan di folder add-on. Menggunakan font Arial.")
            self.font_title = "Arial"
            self.font_body = "Arial"
            self.font_reg = "Arial"

        self.set_auto_page_break(auto=False)

    def rounded_box(self, x, y, w, h, r, style="F"):
        r = min(r, w / 2, h / 2)
        if "F" in style:
            self.rect(x + r, y, w - 2 * r, h, "F")
            self.rect(x, y + r, w, h - 2 * r, "F")
            self.ellipse(x, y, 2 * r, 2 * r, "F")
            self.ellipse(x + w - 2 * r, y, 2 * r, 2 * r, "F")
            self.ellipse(x, y + h - 2 * r, 2 * r, 2 * r, "F")
            self.ellipse(x + w - 2 * r, y + h - 2 * r, 2 * r, 2 * r, "F")

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 24, "F")

        logo_h = 13
        logo_y = (24 - logo_h) / 2
        x_cursor = 196  
        
        its_path = os.path.join(self.base_dir, "add-on", "ITS.png")
        if os.path.exists(its_path):
            x_cursor -= logo_h
            self.image(its_path, x=x_cursor, y=logo_y, h=logo_h)
            x_cursor -= 3

        biomed_path = os.path.join(self.base_dir, "add-on", "BIOMED.png")
        if os.path.exists(biomed_path):
            x_cursor -= logo_h
            self.image(biomed_path, x=x_cursor, y=logo_y, h=logo_h)
            x_cursor -= 3

        logo_path = os.path.join(self.base_dir, "add-on", "logo.png")
        if os.path.exists(logo_path):
            x_cursor -= logo_h
            self.image(logo_path, x=x_cursor, y=logo_y, h=logo_h)

        self.set_xy(14, 6)
        self.set_text_color(*WHITE)
        self.set_font(self.font_title, "", 15)
        self.cell(0, 6, "BIOLENS", ln=True)

        self.set_xy(14, 13)
        self.set_font(self.font_reg, "", 9)
        self.set_text_color(200, 207, 235)
        self.cell(0, 5, "Iron Deficiency Anemia Detection Report", ln=True)

        self.set_y(30)

    def footer(self):
        self.set_y(-16)
        self.set_draw_color(*GRAY_LINE)
        self.set_line_width(0.3)
        self.line(14, self.get_y(), 196, self.get_y())

        self.set_y(-12)
        self.set_font(self.font_reg, "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.cell(95, 6, "Biolens - Automated Microscopy Report", align="L")
        self.cell(95, 6, f"Page {self.page_no()}", align="R")

    def _stat_card(self, x, y, w, h, label, value, value_color=DARK_TEXT, bg=ACCENT_BLUE):
        self.set_fill_color(*bg)
        self.rounded_box(x, y, w, h, 2.5, style="F")

        self.set_xy(x + 4, y + 5)
        self.set_font(self.font_reg, "", 8.5)
        self.set_text_color(*GRAY_TEXT)
        self.cell(w - 8, 4, label, ln=True)

        self.set_xy(x + 4, y + 11)
        self.set_font(self.font_title, "", 17)
        self.set_text_color(*value_color)
        self.cell(w - 8, 9, str(value), ln=True)

    def _severity_badge(self, x, y, is_ida, ratio):
        w, h = 46, 9
        color = RED if is_ida else GREEN
        bg = RED_BG if is_ida else GREEN_BG
        label = f"IDA RISK - {ratio:.1f}%" if is_ida else "NORMAL RANGE"

        self.set_fill_color(*bg)
        self.rounded_box(x, y, w, h, h / 2, style="F")
        self.set_xy(x, y + 1.6)
        self.set_font(self.font_title, "", 9)
        self.set_text_color(*color)
        self.cell(w, 6, label, align="C")

    def _section_title(self, text, y=None):
        if y is not None:
            self.set_y(y)
        self.set_x(14)
        self.set_font(self.font_title, "", 11)
        self.set_text_color(*DARK_TEXT)
        self.cell(0, 6, text, ln=True)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.6)
        ly = self.get_y() + 1
        self.line(14, ly, 28, ly)
        self.ln(4)

    def generate_result(self, imagePath, detectPath, cells, mal, parPath, output_path, patient_name, top_features=None, df_features=None, feature_labels=None):
        self.add_page()
        normal_count = cells - mal
        severity_ratio = (mal / cells * 100) if cells > 0 else 0
        is_ida = mal > 0

        self.set_xy(14, 30)
        self.set_font(self.font_title, "", 13)
        self.set_text_color(*DARK_TEXT)
        self.cell(120, 7, patient_name.replace("_", " "), ln=False)

        self._severity_badge(150, 30, is_ida, severity_ratio)

        self.set_xy(14, 38)
        self.set_font(self.font_reg, "", 8.5)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 5, f"Report generated on {self.timestamp}", ln=True)

        card_y = 46
        card_w = (182 - 2 * 4) / 3
        self._stat_card(14, card_y, card_w, 20, "TOTAL RBC DETECTED", cells,
                         value_color=NAVY, bg=ACCENT_BLUE)
        self._stat_card(14 + card_w + 4, card_y, card_w, 20, "NORMAL CELLS", normal_count,
                         value_color=GREEN, bg=GREEN_BG)
        self._stat_card(14 + 2 * (card_w + 4), card_y, card_w, 20, "IDA CELLS", mal,
                         value_color=RED, bg=RED_BG)

        img_y = 72
        self._section_title("Detection Overview", y=img_y)
        img_top = self.get_y()

        img_w = 88
        img_h = 49.5
        if imagePath and os.path.exists(imagePath):
            self.set_draw_color(*GRAY_LINE)
            self.set_line_width(0.3)
            self.rect(14, img_top, img_w, img_h)
            self.image(imagePath, x=14, y=img_top, w=img_w, h=img_h)
            self.set_xy(14, img_top + img_h + 1)
            self.set_font(self.font_reg, "", 7.5)
            self.set_text_color(*GRAY_TEXT)
            self.cell(img_w, 4, "Raw sample image", align="C")

        if detectPath and os.path.exists(detectPath):
            x2 = 14 + img_w + 6
            self.rect(x2, img_top, img_w, img_h)
            self.image(detectPath, x=x2, y=img_top, w=img_w, h=img_h)
            self.set_xy(x2, img_top + img_h + 1)
            self.cell(img_w, 4, "Detection result", align="C")

        legend_y = img_top + img_h + 7
        self.set_xy(14, legend_y)
        self.set_fill_color(*GREEN)
        self.rect(14, legend_y + 1, 3, 3, "F")
        self.set_font(self.font_reg, "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.set_xy(18, legend_y)
        self.cell(45, 5, "Normal red blood cell")

        self.set_fill_color(*RED)
        self.rect(70, legend_y + 1, 3, 3, "F")
        self.set_xy(74, legend_y)
        self.cell(45, 5, "IDA-classified cell")

        feat_y = legend_y + 10
        self._section_title("Key Morphological Features Analyzed", y=feat_y)

        self.set_font(self.font_reg, "", 8.5)
        self.set_text_color(70, 75, 90)
        if top_features:
            chips_y = self.get_y()
            x_cur = 14
            chip_h = 6.5
            self.set_font(self.font_reg, "", 7.5)
            for feat in top_features:
                fw = self.get_string_width(feat) + 6
                if x_cur + fw > 196:
                    x_cur = 14
                    chips_y += chip_h + 2
                self.set_fill_color(*ACCENT_BLUE)
                self.set_text_color(*NAVY_LIGHT)
                self.set_xy(x_cur, chips_y)
                self.rounded_box(x_cur, chips_y, fw, chip_h, chip_h / 2, style="F")
                self.set_xy(x_cur, chips_y + 1.3)
                self.cell(fw, 4, feat, align="C")
                x_cur += fw + 2.5
            self.set_y(chips_y + chip_h + 4)

            self.set_x(14)
            self.set_font(self.font_reg, "", 8.5)
            self.set_text_color(*GRAY_TEXT)
            explanation = ("These metrics quantify variations in cell size (microcytosis), "
                           "cell shape (poikilocytosis), and central pallor widening -- "
                           "morphological hallmarks used by the model to flag Iron Deficiency Anemia.")
            self.multi_cell(182, 4.3, explanation)
        else:
            self.set_x(14)
            self.multi_cell(182, 4.3, "Morphological features such as cell area, perimeter, "
                                       "and shape factor were analyzed to determine cell normality.")

        grid_y = self.get_y() + 5
        self._section_title("Sample Extracted Cells", y=grid_y)
        grid_top = self.get_y()

        if parPath and os.path.exists(parPath) and os.path.isdir(parPath):
            image_files = sorted(os.listdir(parPath))[:8]
            cell_w = 21
            gap = 3.2
            n_cols = 8
            total_w = n_cols * cell_w + (n_cols - 1) * gap
            start_x = 14 + (182 - total_w) / 2
            for index, filename in enumerate(image_files):
                x = start_x + index * (cell_w + gap)
                y = grid_top
                file_path = os.path.join(parPath, filename)
                self.set_draw_color(*GRAY_LINE)
                self.set_line_width(0.2)
                self.rect(x, y, cell_w, cell_w)
                self.image(file_path, x=x + 0.4, y=y + 0.4, w=cell_w - 0.8, h=cell_w - 0.8)

            grid_bottom = grid_top + cell_w + 6
        else:
            grid_bottom = grid_top + 4

        box_y = grid_bottom + 10
        box_h = 26
        color = RED if is_ida else GREEN
        bg = RED_BG if is_ida else GREEN_BG

        self.set_fill_color(*bg)
        self.rounded_box(14, box_y, 182, box_h, 2.5, style="F")
        self.set_fill_color(*color)
        self.rounded_box(14, box_y, 2.5, box_h, 1, style="F")

        self.set_xy(21, box_y + 3)
        self.set_font(self.font_title, "", 10)
        self.set_text_color(*color)
        self.cell(0, 5, "CONCLUSION", ln=True)

        self.set_xy(21, box_y + 9)
        self.set_font(self.font_reg, "", 8.7)
        self.set_text_color(50, 54, 66)
        if is_ida:
            text = (f"Based on system detection, the patient exhibits a {severity_ratio:.1f}% "
                    f"ratio of abnormal cells ({mal} of {cells}), indicating potential Iron "
                    f"Deficiency Anemia. Further clinical evaluation is recommended.")
        else:
            text = ("The system detection indicates normal cells. No significant "
                    "morphological abnormalities associated with IDA were found.")
        self.set_x(21)
        self.multi_cell(168, 4.3, text)

        if df_features is not None and not df_features.empty and top_features:
            self._feature_analysis_page(df_features, top_features, feature_labels,
                                         parPath, is_ida)

        self.output(output_path)

    def _feature_analysis_page(self, df_features, top_features, feature_labels, parPath, is_ida):
        feature_labels = feature_labels or {}
        pred_col = "Predicted_Class" if "Predicted_Class" in df_features.columns else None

        self.add_page()
        self._section_title("Feature Analysis", y=30)
        self.set_x(14)
        self.set_font(self.font_reg, "", 8.5)
        self.set_text_color(*GRAY_TEXT)
        self.multi_cell(182, 4.3,
            "The table below compares the average value of each top feature between cells "
            "classified as Normal and IDA. Larger gaps indicate stronger influence on the "
            "model's decision.")
        self.ln(2)

        table_y = self.get_y() + 2
        row_h = 7.5
        col_feat = 62
        col_val = (182 - col_feat) / 3

        self.set_xy(14, table_y)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font(self.font_title, "", 8.5)
        self.cell(col_feat, row_h, "  Feature", fill=True, align="L")
        self.cell(col_val, row_h, "Normal avg", fill=True, align="C")
        self.cell(col_val, row_h, "IDA avg", fill=True, align="C")
        self.cell(col_val, row_h, "Difference", fill=True, align="C")
        self.ln(row_h)

        df_normal = df_features[df_features[pred_col] == 0] if pred_col else None
        df_ida = df_features[df_features[pred_col] == 1] if pred_col else None

        for i, feat in enumerate(top_features):
            if feat not in df_features.columns:
                continue
            y_row = self.get_y()
            bg = WHITE if i % 2 == 0 else (245, 246, 250)
            self.set_fill_color(*bg)
            self.rect(14, y_row, 182, row_h, "F")

            label = feature_labels.get(feat, feat.replace("_", " "))
            self.set_xy(14, y_row)
            self.set_font(self.font_reg, "", 8)
            self.set_text_color(*DARK_TEXT)
            self.cell(col_feat, row_h, "  " + label, align="L")

            mean_n = df_normal[feat].mean() if df_normal is not None and len(df_normal) else None
            mean_i = df_ida[feat].mean() if df_ida is not None and len(df_ida) else None

            self.set_text_color(*GREEN)
            self.cell(col_val, row_h, f"{mean_n:.3f}" if mean_n is not None else "-", align="C")
            self.set_text_color(*RED)
            self.cell(col_val, row_h, f"{mean_i:.3f}" if mean_i is not None else "-", align="C")

            if mean_n is not None and mean_i is not None:
                delta = mean_i - mean_n
                pct = (abs(delta) / abs(mean_n) * 100) if mean_n != 0 else 0
                self.set_text_color(*NAVY_LIGHT)
                self.cell(col_val, row_h, f"{delta:+.3f} ({pct:.0f}%)", align="C")
            else:
                self.cell(col_val, row_h, "-", align="C")
            self.ln(row_h)

        self.set_draw_color(*GRAY_LINE)
        self.set_line_width(0.2)
        self.line(14, self.get_y(), 196, self.get_y())

        sample_y = self.get_y() + 8
        self._section_title("Representative Cell Examples", y=sample_y)
        self.set_x(14)
        self.set_font(self.font_reg, "", 8.5)
        self.set_text_color(*GRAY_TEXT)
        self.multi_cell(182, 4.3,
            "Below are sample cells with their actual extracted feature values, "
            "shown side-by-side for each predicted class.")
        self.ln(2)

        card_top = self.get_y() + 2
        card_w = (182 - 6) / 2
        card_h = 64

        groups = []
        if df_normal is not None and len(df_normal):
            groups.append(("Normal", GREEN, GREEN_BG, df_normal.iloc[0]))
        if df_ida is not None and len(df_ida):
            groups.append(("IDA", RED, RED_BG, df_ida.iloc[0]))

        for idx, (label, color, bg, row) in enumerate(groups):
            x = 14 + idx * (card_w + 6)
            self.set_fill_color(*bg)
            self.rounded_box(x, card_top, card_w, card_h, 2.5, style="F")

            self.set_xy(x + 4, card_top + 4)
            self.set_font(self.font_title, "", 9)
            self.set_text_color(*color)
            cell_id = int(row.get("Cell_Label", idx + 1))
            self.cell(card_w - 8, 5, f"{label} - Cell #{cell_id}", ln=True)

            img_x = x + 4
            img_y = card_top + 11
            img_size = 32
            thumb_path = None
            if parPath and os.path.exists(parPath) and os.path.isdir(parPath):
                candidate = os.path.join(parPath, f"cell_{cell_id - 1}.png")
                if os.path.exists(candidate):
                    thumb_path = candidate
            if thumb_path:
                self.set_draw_color(*GRAY_LINE)
                self.rect(img_x, img_y, img_size, img_size)
                self.image(thumb_path, x=img_x + 0.4, y=img_y + 0.4,
                           w=img_size - 0.8, h=img_size - 0.8)

            text_x = img_x + img_size + 5
            text_w = card_w - img_size - 13

            self.set_xy(text_x, img_y)
            self.set_font(self.font_reg, "", 7.3)
            self.set_text_color(50, 54, 66)
            for feat in top_features[:4]:
                if feat not in row.index:
                    continue
                label_f = feature_labels.get(feat, feat.replace("_", " "))
                val = row[feat]
                self.set_x(text_x)
                self.cell(text_w, 4.6, f"{label_f}", ln=True)
                self.set_x(text_x)
                self.set_font(self.font_title, "", 8)
                self.set_text_color(*color)
                self.cell(text_w, 4.8, f"{val:.3f}", ln=True)
                self.set_font(self.font_reg, "", 7.3)
                self.set_text_color(50, 54, 66)

            remaining = top_features[4:8]
            if remaining:
                ry = img_y + img_size + 6
                col_w = (card_w - 8) / 2
                self.set_font(self.font_reg, "", 7)
                for ridx, feat in enumerate(remaining):
                    if feat not in row.index:
                        continue
                    rcol = ridx % 2
                    rrow = ridx // 2
                    rx = x + 4 + rcol * col_w
                    rry = ry + rrow * 9.5
                    label_f = feature_labels.get(feat, feat.replace("_", " "))
                    val = row[feat]
                    self.set_xy(rx, rry)
                    self.set_text_color(*GRAY_TEXT)
                    self.cell(col_w, 4, label_f, ln=True)
                    self.set_xy(rx, rry + 3.8)
                    self.set_font(self.font_title, "", 7.5)
                    self.set_text_color(*color)
                    self.cell(col_w, 4, f"{val:.3f}", ln=True)
                    self.set_font(self.font_reg, "", 7)
