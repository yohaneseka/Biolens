import sys
import os
import glob
import time
import shutil
import numpy as np
import cv2 as cv
import pandas as pd
import imageio
from PIL import Image

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton,
    QRadioButton, QStackedWidget, QVBoxLayout, QCheckBox, QSpinBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

from core.hardware import ESP32Controller, MagnificationSensor, CameraSystem
from core.image_processing import (
    convert_hsv_circular, kmeans_segmentation, remove_unwanted_cells_extended,
    bounded_opening_frs, separate_overlapping_rbc_with_gmm, sobel_edge_detect,
    draw_bounding_boxes, extract_contours
)
from core.feature_extraction import run_feature_extraction
from core.machine_learning import SVMDetector
from utils.report import PDFWithHeaderFooter
from core.dictionary import FEATURE_LABELS

class CaptureThread(QThread):
    capture_finished = pyqtSignal(bool)

    def __init__(self, camera, image_path):
        super().__init__()
        self.camera = camera
        self.image_path = image_path

    def run(self):
        try:
            self.camera.picam2.capture_file(self.image_path)
            time.sleep(0.5) 
            self.capture_finished.emit(True)
        except Exception as e:
            self.capture_finished.emit(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        uic.loadUi("Main_Program.ui", self)
        
        logo_biomed_path = os.path.join(self.base_dir, "add-on", "BIOMED.png")
        if hasattr(self, 'label_15') and os.path.exists(logo_biomed_path):
            self.label_15.setPixmap(QPixmap(logo_biomed_path).scaled(
                self.label_15.width(), self.label_15.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        logo_its_path = os.path.join(self.base_dir, "add-on", "ITS.png")
        if hasattr(self, 'label_16') and os.path.exists(logo_its_path):
            self.label_16.setPixmap(QPixmap(logo_its_path).scaled(
                self.label_16.width(), self.label_16.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        self.master_data_dir = os.path.join(self.base_dir, "DATA_PASIEN")
        os.makedirs(self.master_data_dir, exist_ok=True)

        self.motor = ESP32Controller()
        self.sensor = MagnificationSensor()
        self.camera = CameraSystem()
        self.ml_detector = SVMDetector(self.base_dir)

        self.stackedWidget = self.findChild(QStackedWidget, "stackedWidget")
        self.nameInput = self.findChild(QLineEdit, "nameInput")

        self.mainPage = self.findChild(QPushButton, "mainBtn")
        self.segmentPage = self.findChild(QPushButton, "rbcBtn")
        self.detectPage = self.findChild(QPushButton, "malBtn")
        self.aboutPage = self.findChild(QPushButton, "abtBtn")
        self.close_app = self.findChild(QPushButton, "closeBtn")

        self.distVal = self.findChild(QLabel, "distVal")
        self.imageSource = [self.findChild(QRadioButton, "camInput"), self.findChild(QRadioButton, "fileInput")]
        self.getButton = self.findChild(QPushButton, "getBtn")
        self.inputIm = self.findChild(QLabel, "rawImage")
        
        self.kmeansButton = self.findChild(QPushButton, "kmeansBtn") or self.findChild(QPushButton, "doSegBtn")
        self.clusterText = self.findChild(QLabel, "clustText")
        self.selectCluster = [self.findChild(QCheckBox, f"clust{i}") for i in range(1, 7)]
        self.clusterIm = [self.findChild(QLabel, f"clust{i}Im") for i in range(1, 7)]

        self.extractButton = self.findChild(QPushButton, "extBtn")
        self.extractedIm = self.findChild(QLabel, "cellsExtract")
        self.rbcValText = self.findChild(QLabel, "rbcText")
        self.sepOverlap = self.findChild(QPushButton, "overlapBtn")
        self.saveCells = self.findChild(QPushButton, "saveBtn")
        self.detectButton = self.findChild(QPushButton, "detectBtn")

        self.detectText = self.findChild(QLabel, "detectText")
        self.detectIm = self.findChild(QLabel, "detectIm")
        self.pdfGenButton = self.findChild(QPushButton, "pdfBtn")

        self.spinBox = self.findChild(QSpinBox, "spinBox")
        self.upBtn = self.findChild(QPushButton, "upBtn")
        self.downBtn = self.findChild(QPushButton, "downBtn")
        self.stopBtn = self.findChild(QPushButton, "stopBtn")

        if self.spinBox: self.spinBox.setRange(1, 99999)
        if self.upBtn: self.upBtn.clicked.connect(lambda checked=False: self.motor.send_command('U', self.spinBox.value()))
        if self.downBtn: self.downBtn.clicked.connect(lambda checked=False: self.motor.send_command('D', self.spinBox.value()))
        if self.stopBtn: self.stopBtn.clicked.connect(self.motor.stop)

        self.layout = QVBoxLayout()
        self.sensor_timer = QTimer()
        self.sensor_timer.timeout.connect(self.update_sensor_value)
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self._update_frame)
        self.camera_started = False

        if self.imageSource[0]: self.imageSource[0].toggled.connect(self.cameraInputToggled)
        if self.imageSource[1]: self.imageSource[1].toggled.connect(self.externalFileToggled)
        if self.getButton: self.getButton.clicked.connect(self.takeImage)
        if self.kmeansButton: self.kmeansButton.clicked.connect(self.kmeansProcess)
        if self.extractButton: self.extractButton.clicked.connect(self.extractCells)
        if self.sepOverlap: self.sepOverlap.clicked.connect(self.separateOverlap)
        if self.saveCells: self.saveCells.clicked.connect(self.saveExtractedCells)
        if self.detectButton: self.detectButton.clicked.connect(self.detectCells)
        if self.pdfGenButton: self.pdfGenButton.clicked.connect(self.generatePDF)

        # MENGGUNAKAN FUNGSI AMAN UNTUK MENU AGAR TIDAK FREEZE
        if self.mainPage: self.mainPage.clicked.connect(self.moveMainPage)
        if self.segmentPage: self.segmentPage.clicked.connect(self.moveSegmentPage)
        if self.detectPage: self.detectPage.clicked.connect(self.moveDetectPage)
        if self.aboutPage: self.aboutPage.clicked.connect(self.moveAboutPage)
        if self.close_app: self.close_app.clicked.connect(self.closeApp)

        self._initializing = True
        if self.imageSource[0]:
            self.imageSource[0].setChecked(True)
        self._initializing = False
        
        self.setStyles()
        print("=== CEK STATUS TOMBOL ===")
        for name, btn in [("mainPage", self.mainPage), 
                          ("aboutPage", self.aboutPage),
                          ("close_app", self.close_app),
                          ("segmentPage", self.segmentPage),
                          ("detectPage", self.detectPage)]:
            if btn:
                print(f"{name}: enabled={btn.isEnabled()}, parent={btn.parent()}, parent_enabled={btn.parent().isEnabled() if btn.parent() else 'N/A'}")
            else:
                print(f"{name}: NOT FOUND")

        if self.camera.using_picam and self.camera.qpicamera2:
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.layout.addWidget(self.camera.qpicamera2)
            if self.inputIm: self.inputIm.setLayout(self.layout)

    def moveMainPage(self):
        if self.stackedWidget: self.stackedWidget.setCurrentIndex(0)
    def moveSegmentPage(self):
        if self.stackedWidget: self.stackedWidget.setCurrentIndex(1)
    def moveDetectPage(self):
        if self.stackedWidget: self.stackedWidget.setCurrentIndex(3)
    def moveAboutPage(self):
        if self.stackedWidget: self.stackedWidget.setCurrentIndex(4)
    def closeApp(self):
        self.close()
    
    def update_sensor_value(self):
        distance = self.sensor.read_distance()
       
        if self.distVal:
            self.distVal.setText(
                f"Lens to Object Dist : {distance:.1f} mm"
            )

    def setStyles(self):
        btn_style = "QPushButton {border:1px; border-radius: 10px; color: rgb(0,0,0); background-color: rgb(214, 222, 255);} QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255);} QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);}"
        menu_style = "QPushButton {border:1px; color: rgb(225,225,225); background-color: rgb(17, 70, 143)} QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255);} QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);}"
        if self.mainPage: self.mainPage.setStyleSheet(menu_style)
        if self.segmentPage: self.segmentPage.setStyleSheet(btn_style)
        if self.detectPage: self.detectPage.setStyleSheet(btn_style)
        if self.aboutPage: self.aboutPage.setStyleSheet(menu_style)

    def _re_enable_navigation(self):
        nav_buttons = [self.mainPage, self.segmentPage, self.detectPage, 
                       self.aboutPage, self.close_app]
        for btn in nav_buttons:
            if btn:
                btn.setEnabled(True) 
    
    def _create_session_folders(self):
        self.current_patient = self.nameInput.text().strip().replace(" ", "_") if self.nameInput and self.nameInput.text().strip() != "" else "Anonim"
        session_folder_name = f"{self.current_patient}_{time.strftime('%Y%m%d_%H%M%S')}"
        session_path = os.path.join(self.master_data_dir, session_folder_name)
        
        self.current_raw_dir = os.path.join(session_path, "0_raw_image")
        self.current_clust_dir = os.path.join(session_path, "1_clustering_image")
        self.current_sep_dir = os.path.join(session_path, "2_separated_cells")
        self.current_res_dir = os.path.join(session_path, "3_results")
        for folder in [self.current_raw_dir, self.current_clust_dir, self.current_sep_dir, self.current_res_dir]: os.makedirs(folder, exist_ok=True)

    def cameraInputToggled(self, checked):
        print("CAMERA TOGGLED", checked)
        if checked:
            self.sensor_timer.start(500)
            if self.camera.using_picam:
                if self.camera.qpicamera2.parent() is None:
                    self.layout.addWidget(self.camera.qpicamera2)
                    if self.inputIm:
                        self.inputIm.setLayout(self.layout)
    
                if not self.camera_started:
                    self.camera.start_camera()
                    self.camera_started = True
    
            else:
                self.webcam_timer.start(30)
    
            self.inputIm.clear()
                
    def externalFileToggled(self, checked):
        if getattr(self, '_initializing', False):  # tambah baris ini
            return
        if checked:
            self.sensor_timer.stop()
            if self.distVal: self.distVal.setText("Camera is not active")
            self.webcam_timer.stop()
            if self.camera.using_picam and self.camera_started:
                try:
                    self.camera.picam2.stop()
                    self.camera_started = False
                except:
                    pass

            if self.camera.using_picam and self.camera.qpicamera2:
                self.layout.removeWidget(self.camera.qpicamera2)
                self.camera.qpicamera2.setParent(None)
            
            if self.inputIm: self.inputIm.clear()

    def _update_frame(self):
        frame_rgb = self.camera.get_opencv_frame()
        if frame_rgb is not None and self.inputIm:
            h, w, ch = frame_rgb.shape
            pixmap = QPixmap.fromImage(QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888))
            self.inputIm.setPixmap(pixmap.scaled(self.inputIm.width(), self.inputIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def takeImage(self):
        self._create_session_folders()
        self.imagePath = os.path.join(self.current_raw_dir, f"raw_{self.current_patient}.jpg")
        
        if self.imageSource[0].isChecked():
            self.is_primary_data = True  
            
            if self.getButton:
                self.getButton.setEnabled(False)
                self.getButton.setText("Capturing...")
            
            if self.camera.using_picam:
                self.capture_thread = CaptureThread(self.camera, self.imagePath)
                self.capture_thread.capture_finished.connect(self.on_capture_done)
                self.capture_thread.start()
            else:
                if self.camera.capture_image(self.imagePath): 
                    self.displayImage(self.imagePath)
                    if self.getButton:
                        self.getButton.setEnabled(True)
                        self.getButton.setText("Get Image")
        else:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            
            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()[0]
                
                if "primer" in selected_file.lower():
                    self.is_primary_data = True
                else:
                    self.is_primary_data = False
                
                shutil.copy(selected_file, self.imagePath)
                self.displayImage(self.imagePath) 

    def on_capture_done(self, success):
        if self.getButton:
            self.getButton.setEnabled(True)
            self.getButton.setText("Get Image")
            
        if success:
            print("Thread Kamera Selesai: Berhasil")
            time.sleep(0.3) 
            self.displayImage(self.imagePath) 
        else:
            print("Thread Kamera Selesai: GAGAL")
            if self.clusterText: self.clusterText.setText("Gagal mengambil gambar dari kamera!")

    def displayImage(self, imagePath):
        if hasattr(self, 'webcam_timer') and self.webcam_timer.isActive():
            self.webcam_timer.stop()
            
        if hasattr(self, 'camera') and getattr(self.camera, 'using_picam', False) and getattr(self.camera, 'qpicamera2', None):
            if self.camera.qpicamera2.parent():
                self.layout.removeWidget(self.camera.qpicamera2)
                self.camera.qpicamera2.setParent(None)
                
        if os.path.exists(imagePath) and self.inputIm:
            pixmap = QPixmap(imagePath)
            if pixmap.isNull():
                print(f"Gagal memuat Pixmap! File rusak: {imagePath}")
                return
                
            self.raw_image = cv.imread(imagePath)  
            self.raw_image_rgb = cv.cvtColor(self.raw_image, cv.COLOR_BGR2RGB)

            print(f"Menampilkan gambar ke QLabel: {imagePath}")
            self.inputIm.setPixmap(pixmap.scaled(self.inputIm.width(), self.inputIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.inputIm.setAlignment(Qt.AlignCenter)
            self.inputIm.repaint() 
            QApplication.processEvents() 
        else:
            print(f"File gambar TIDAK DITEMUKAN: {imagePath}")

    def kmeansProcess(self):
        if not hasattr(self, 'current_clust_dir'):
            if self.clusterText: self.clusterText.setText("Silakan Get Image dulu!")
            return
            
        self.stackedWidget.setCurrentIndex(1)
        if self.clusterText: self.clusterText.setText("Please wait, doing Reinhard Norm & k-means clustering...")
        QApplication.processEvents()

        ref_image_rgb = None
        
        if hasattr(self, 'is_primary_data') and not self.is_primary_data:
            ref_path = os.path.join(self.base_dir, "source", "Referensi.png")
            if os.path.exists(ref_path):
                ref_image_rgb = cv.cvtColor(cv.imread(ref_path), cv.COLOR_BGR2RGB)
            else:
                print(f"Gambar referensi tidak ditemukan di {ref_path}. Normalisasi Reinhard dilewati.")

        self.hsv_clean_image, _ = convert_hsv_circular(self.raw_image_rgb, v_thresh=20)

        from core.image_processing import preprocess_image
        self.preprocessed_image = preprocess_image(self.raw_image_rgb, ref_img_rgb=ref_image_rgb)
        
        _, _, self.segmented_images = kmeans_segmentation(self.preprocessed_image, k=6)

        for idx, segment_image in enumerate(self.segmented_images):
            clusterPath = os.path.join(self.current_clust_dir, f"cluster_{idx+1}.jpg")
            cv.imwrite(clusterPath, cv.cvtColor(segment_image, cv.COLOR_RGB2BGR))
            
            pixmap = QPixmap(clusterPath)
            self.clusterIm[idx].setPixmap(pixmap.scaled(self.clusterIm[idx].width(), self.clusterIm[idx].height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.clusterIm[idx].setAlignment(Qt.AlignCenter)
            
        if self.clusterText: self.clusterText.setText("K-Means & Normalization done.")
        self._re_enable_navigation()
        
    def extractCells(self):
        self.stackedWidget.setCurrentIndex(2)
        QApplication.processEvents()
        
        self.selected_cluster = [i for i, chk in enumerate(self.selectCluster) if chk.isChecked()]
        rgb_clean_image = cv.cvtColor(self.hsv_clean_image, cv.COLOR_HSV2RGB)
        
        self.rbc_only_image, self.filtered_mask, self.binary_mask = remove_unwanted_cells_extended(
            self.segmented_images, 
            self.selected_cluster, 
            self.preprocessed_image
        )
        
        gray_img = cv.cvtColor(self.rbc_only_image, cv.COLOR_RGB2GRAY)
        edge_map, contour_edge = sobel_edge_detect(gray_img)
        cells_detected = draw_bounding_boxes(self.rbc_only_image, contour_edge)
        
        contours, _ = extract_contours(gray_img, edge_map)
        self.extracted_cells, self.cell_masks_list, self.bounding_boxes_sep = [], [], []
        
        for contour in contours:
            c_mask = np.zeros(self.rbc_only_image.shape[:2], dtype=np.uint8)
            cv.drawContours(c_mask, [contour], -1, 255, -1)
            x, y, w, h = cv.boundingRect(contour)
            self.extracted_cells.append(cv.bitwise_and(self.rbc_only_image[y:y+h, x:x+w], self.rbc_only_image[y:y+h, x:x+w], mask=c_mask[y:y+h, x:x+w]))
            self.cell_masks_list.append(c_mask[y:y+h, x:x+w])
            self.bounding_boxes_sep.append((x, y, w, h))

        detectPath = os.path.join(self.current_res_dir, "detect_cells_initial.jpg")
        cv.imwrite(detectPath, cells_detected)
        if self.extractedIm: 
            self.extractedIm.setPixmap(QPixmap(detectPath).scaled(self.extractedIm.width(), self.extractedIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.extractedIm.setAlignment(Qt.AlignCenter)
        if self.rbcValText: 
            self.rbcValText.setText(f"{len(self.extracted_cells)} RBC detected. Click Separate Cells if overlapping.")
        self._re_enable_navigation()
        
    def separateOverlap(self):
        if self.rbcValText: self.rbcValText.setText("Separating overlapping cells using BO-FRS + GMM...")
        QApplication.processEvents()

        bofrs_results = bounded_opening_frs(self.filtered_mask, self.preprocessed_image, num_openings=4)
        self.extracted_cells, self.bounding_boxes_sep, self.cell_masks_list = separate_overlapping_rbc_with_gmm(bofrs_results, self.rbc_only_image)

        copy_rbc = self.rbc_only_image.copy()
        for idx, bbox in enumerate(self.bounding_boxes_sep, start=1):
            x, y, w, h = bbox
            cv.rectangle(copy_rbc, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
        sepPath = os.path.join(self.current_res_dir, "after_sep.jpg")
        cv.imwrite(sepPath, copy_rbc)
        if self.extractedIm: 
            self.extractedIm.setPixmap(QPixmap(sepPath).scaled(self.extractedIm.width(), self.extractedIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.extractedIm.setAlignment(Qt.AlignCenter)
        if self.rbcValText: self.rbcValText.setText(f"Separation completed! {len(self.extracted_cells)} individual cells detected.")
        self._re_enable_navigation()
        
    def saveExtractedCells(self):
        if not hasattr(self, 'extracted_cells') or len(self.extracted_cells) == 0:
            if self.rbcValText:
                self.rbcValText.setText("Tidak ada sel. Jalankan Extract dulu!")
            return
    
        if self.rbcValText:
            self.rbcValText.setText("Menyimpan sel & mengekstraksi fitur, harap tunggu...")
        QApplication.processEvents()
    
        self.cell_info = []
        for idx, (cell_img, bbox) in enumerate(zip(self.extracted_cells, self.bounding_boxes_sep)):
            save_path = os.path.join(self.current_sep_dir, f"cell_{idx}.png")
            cv.imwrite(save_path, cv.cvtColor(cell_img, cv.COLOR_RGB2BGR))
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": bbox})
    
        csv_path = os.path.join(self.current_res_dir, f"features_{self.current_patient}.csv")
        try:
            self.df_features, self.cell_labels, filter_stats = run_feature_extraction(
                extracted_cells=self.extracted_cells,
                bounding_boxes=self.bounding_boxes_sep,
                cell_masks=self.cell_masks_list,
                img_shape=self.rbc_only_image.shape,
                output_csv_path=None
            )
            if not self.df_features.empty:
                self.df_features.to_csv(csv_path, index=False)
                if self.rbcValText:
                    self.rbcValText.setText(
                        f"{len(self.extracted_cells)} sel tersimpan | "
                        f"{filter_stats['passed']} lolos QC | "
                        f"Ditolak: border={filter_stats['border']}, "
                        f"kecil={filter_stats['small']}, besar={filter_stats['large']}, "
                        f"bentuk={filter_stats['shape']} — "
                        f"Klik 'Feature Extraction' untuk deteksi."
                    )
            else:
                if self.rbcValText:
                    self.rbcValText.setText("Tidak ada sel yang lolos quality filter.")
        except Exception as e:
            if self.rbcValText:
                self.rbcValText.setText(f"Ekstraksi fitur gagal: {e}")
    
        self._re_enable_navigation()
        
    def detectCells(self):
        self.stackedWidget.setCurrentIndex(3)
        if self.detectText:
            self.detectText.setText("Menjalankan seleksi fitur & model SVM...")
        QApplication.processEvents()
    
        if not hasattr(self, 'df_features') or self.df_features.empty:
            if self.detectText:
                self.detectText.setText("Belum ada fitur. Jalankan Save dulu!")
            return
    
        if not hasattr(self, 'preprocessed_image'):
            if self.detectText:
                self.detectText.setText("Jalankan K-Means dulu sebelum deteksi.")
            return
    
        try:
            selected_features = self.ml_detector.metadata.get('features', [])
            if not selected_features:
                raise Exception("Metadata model tidak memiliki daftar 'features'.")
    
            missing = [f for f in selected_features if f not in self.df_features.columns]
            if missing:
                raise Exception(f"Fitur tidak ada di hasil ekstraksi: {missing}")
    
            X = self.df_features[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled    = self.ml_detector.scaler.transform(X)
            predictions = self.ml_detector.model.predict(X_scaled)
            self.predictions = predictions
    
            ida_count    = int((predictions == 1).sum())
            normal_count = int((predictions == 0).sum())
    
            result_img = cv.cvtColor(self.raw_image_rgb.copy(), cv.COLOR_RGB2BGR)
            for i, pred in enumerate(predictions):
                x = int(self.df_features.iloc[i]["X"])
                y = int(self.df_features.iloc[i]["Y"])
                bbox_match = next(
                    (b for b in self.bounding_boxes_sep if b[0] == x and b[1] == y), None
                )
                bx, by, bw, bh = bbox_match if bbox_match else (x, y, 40, 40)
    
                color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                label = "IDA" if pred == 1 else "Normal"
                cv.rectangle(result_img, (bx, by), (bx + bw, by + bh), color, 5)
                cv.putText(result_img, label, (bx, by - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
            self.detectResultPath = os.path.join(
                self.current_res_dir, f"detection_result_{self.current_patient}.jpg"
            )
            cv.imwrite(self.detectResultPath, result_img)
    
            self.total_cells = ida_count + normal_count
            self.ida_cells   = ida_count
    
            self.top10    = selected_features[:10]
            top10 = selected_features[:10]
            summary = (
                f"Deteksi Selesai!\n"
                f"Total sel: {self.total_cells} | IDA: {ida_count} | Normal: {normal_count}\n"
                f"Top-10 Fitur:\n" + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top10))
            )
            if self.detectText:
                self.detectText.setText(summary)
            if self.detectIm:
                self.detectIm.setPixmap(
                    QPixmap(self.detectResultPath).scaled(
                        self.detectIm.width(), self.detectIm.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
                self.detectIm.setAlignment(Qt.AlignCenter)
    
        except Exception as e:
            if self.detectText:
                self.detectText.setText(f"Deteksi gagal: {e}")
    
        self._re_enable_navigation()

    def generatePDF(self):
        pdf_path = os.path.join(self.current_res_dir, f"Report_{self.current_patient}.pdf")
        pdf = PDFWithHeaderFooter(self.base_dir)
        
        features_to_pass = getattr(self, 'top10_features', []) 

        df_with_pred = None
        if hasattr(self, 'df_features') and not self.df_features.empty and hasattr(self, 'predictions'):
            df_with_pred = self.df_features.copy()
            df_with_pred['Predicted_Class'] = self.predictions
            
        pdf.generate_result(
            imagePath=self.imagePath, 
            detectPath=self.detectResultPath,
            cells=self.total_cells, 
            mal=self.ida_cells, 
            parPath=self.current_sep_dir,
            output_path=pdf_path, 
            patient_name=self.current_patient,
            top_features=features_to_pass,
            df_features=df_with_pred,       
            feature_labels=FEATURE_LABELS,
        )
        if self.detectText: self.detectText.setText(f"Report generated in PDF format at {self.current_res_dir}")

    def closeEvent(self, event):
        self.motor.close()
        self.camera.close()
        if hasattr(self, 'webcam_timer') and self.webcam_timer.isActive(): self.webcam_timer.stop()
        if hasattr(self, 'sensor_timer') and self.sensor_timer.isActive(): self.sensor_timer.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
