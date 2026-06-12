import os
import cv2 as cv
import numpy as np
import joblib
from core.feature_extraction import run_feature_extraction

class SVMDetector:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model = None
        self.scaler = None
        self.metadata = None

        models_dir = os.path.join(self.base_dir, "source", "models")
        model_path    = os.path.join(models_dir, "model.pkl")
        scaler_path   = os.path.join(models_dir, "scaler.pkl")
        metadata_path = os.path.join(models_dir, "metadata.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
            self.model    = joblib.load(model_path)
            self.scaler   = joblib.load(scaler_path)
            self.metadata = joblib.load(metadata_path)
            print("Model SVM, Scaler, dan Metadata berhasil diload!")
        else:
            print(f"File .pkl tidak ditemukan di {models_dir}")

    def run_detection_pipeline(self, extracted_cells, bounding_boxes_sep, cell_masks_list,
                               raw_image_rgb, output_dir, patient_id):
        if self.model is None or self.scaler is None or self.metadata is None:
            raise Exception("Model SVM tidak ditemukan! Pastikan file pkl ada di folder source/models/")

        selected_features = self.metadata.get('features', [])
        if not selected_features:
            raise Exception("Metadata tidak memiliki daftar 'features'.")

        df_features, _, _ = run_feature_extraction(
            extracted_cells, bounding_boxes_sep, cell_masks_list, raw_image_rgb.shape
        )
        if df_features.empty:
            raise Exception("Feature extraction tidak menghasilkan data.")

        missing = [f for f in selected_features if f not in df_features.columns]
        if missing:
            raise Exception(f"Fitur hilang dari ekstraksi: {missing}")

        X = df_features[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled     = self.scaler.transform(X)
        predictions  = self.model.predict(X_scaled)

        ida_count    = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())

        # 4. Gambar hasil deteksi di atas gambar asli
        result_img = cv.cvtColor(preprocessed_image_rgb.copy(), cv.COLOR_RGB2BGR)
        for bbox, pred in zip(bounding_boxes_sep, predictions):
            x, y, w, h = bbox
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)
            label = "IDA" if pred == 1 else "Normal"
            cv.rectangle(result_img, (x, y), (x+w, y+h), color, 5)
            cv.putText(result_img, label, (x, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
        res_path = os.path.join(output_dir, f"detection_result_{patient_id}.jpg")
        cv.imwrite(res_path, result_img)
    
        top5 = selected_features[:5] if len(selected_features) >= 5 else selected_features
        return res_path, ida_count, normal_count, top5
