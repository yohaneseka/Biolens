import os
import time
from fpdf import FPDF

class PDFWithHeaderFooter(FPDF):
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        poppins_path = os.path.join(self.base_dir, "add-on", "Poppins-Bold.ttf")
        inter_path = os.path.join(self.base_dir, "add-on", "Inter_18pt-SemiBold.ttf")
        
        if os.path.exists(poppins_path) and os.path.exists(inter_path):
            self.add_font("Poppins", "", poppins_path, uni=True)
            self.add_font("Inter", "", inter_path, uni=True)
            self.font_title = "Poppins"
            self.font_body = "Inter"
        else:
            print("⚠️ Peringatan: File font TTF tidak ditemukan di folder add-on. Menggunakan font Arial.")
            self.font_title = "Arial"
            self.font_body = "Arial"

    def header(self):
        self.set_font(self.font_title, "", 20)
        self.set_fill_color(4, 21, 98)
        self.set_xy(0, 16)
        self.cell(165, 3, fill=True)
        
        logo_path = os.path.join(self.base_dir, "add-on", "logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, x=170, y=10, w=30)
            
        self.ln(10)
        self.cell(0, 10, "Segmentation and Detection Report", ln=True, align="L")
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font(self.font_title, "", 15)
        self.set_text_color(100)
        self.cell(0, 10, "Biolens, 2026", align="L")
        self.set_xy(65, 281)
        self.set_fill_color(4, 21, 98)
        self.cell(170, 2, fill=True)

    def generate_result(self, imagePath, detectPath, cells, mal, parPath, output_path, patient_name, top_features=None):
        self.add_page()
        self.set_font(self.font_body, size=12)
        self.set_xy(18, 40)
        self.set_text_color(0)
        self.cell(170, 10, f"Patient Name / ID: {patient_name.replace('_', ' ')}", ln=True)
        
        self.set_xy(18, 50)
        self.set_text_color(120)
        self.cell(170, 10, f"Report generated on {self.timestamp}", ln=True)

        if imagePath and os.path.exists(imagePath): self.image(imagePath, x=18, y=70, w=88, h=49.5)
        if detectPath and os.path.exists(detectPath): self.image(detectPath, x=100, y=70, w=88, h=49.5)

        self.set_font_size(10)
        self.set_text_color(150)
        self.set_xy(18, 123)
        self.multi_cell(170, 5, "Green bounding boxes indicate normal red blood cells, while red bounding boxes indicate IDA cells.")

        self.set_text_color(0)
        self.set_xy(18, 135)
        self.set_font_size(12)
        
        severity_ratio = (mal / cells * 100) if cells > 0 else 0
        
        self.cell(170, 8, f"Total red blood cells detected: {cells}", border=1, ln=True)
        self.set_x(18)
        self.cell(170, 8, f"Infected/Abnormal cells detected: {mal}", border=1, ln=True)
        self.set_x(18)
        self.cell(170, 8, f"IDA Severity / Confidence Ratio: {severity_ratio:.2f}%", border=1, ln=True)

        self.ln(5)
        self.set_x(18)
        self.set_font(self.font_title, "", 12)
        self.cell(170, 8, "Key Morphological Features Analyzed:", ln=True)
        
        self.set_font(self.font_body, size=10)
        self.set_text_color(80)
        if top_features:
            features_text = ", ".join(top_features)
            explanation = f"The classification model relies on top extracted features: {features_text}. In Iron Deficiency Anemia (IDA), these metrics are crucial for identifying variations in cell size (microcytosis), cell shape (poikilocytosis), and central pallor widening."
        else:
            explanation = "Morphological features such as cell area, perimeter, and shape factor were analyzed to determine cell normality."
            
        self.set_x(18)
        self.multi_cell(170, 5, explanation)

        current_y = self.get_y() + 5 
        if parPath and os.path.exists(parPath) and os.path.isdir(parPath):
            image_files = os.listdir(parPath)[:8]
            for index, filename in enumerate(image_files):
                col = index % 4
                row = index // 4
                x = 40 + col * 32
                y = current_y + row * 32
                file_path = os.path.join(parPath, filename)
                self.image(file_path, x=x, y=y, w=30, h=30)

        box_y = current_y + 70 
        self.set_text_color(255)
        self.set_xy(18, box_y)
        
        style_title = "" if self.font_title == "Poppins" else "B"
        self.set_font(self.font_title, style_title, 11)
        
        if mal > 0:
            self.set_fill_color(220, 53, 69) 
            self.multi_cell(170, 6, f"CONCLUSION: Based on system detection, the patient exhibits a {severity_ratio:.1f}% ratio of abnormal cells, indicating potential IDA. Further clinical evaluation is recommended.", border=1, fill=True)
        else:
            self.set_fill_color(40, 167, 69) 
            self.multi_cell(170, 6, "CONCLUSION: The system detection indicates normal cells. No significant morphological abnormalities associated with IDA were found.", border=1, fill=True)

        self.output(output_path)
