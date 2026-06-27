import os
import cv2 as cv
import numpy as np
import pandas as pd
import joblib
from core.feature_extraction import run_feature_extraction

class SVMDetector:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model    = None
        self.scaler   = None
        self.imputer  = None
        self.metadata = None
 
        models_dir    = os.path.join(self.base_dir, "source", "models")
        model_path    = os.path.join(models_dir, "model.pkl")
        scaler_path   = os.path.join(models_dir, "scaler.pkl")
        metadata_path = os.path.join(models_dir, "metadata.pkl")
        imputer_path  = os.path.join(models_dir, "imputer.pkl")
 
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
            self.model    = joblib.load(model_path)
            self.scaler   = joblib.load(scaler_path)
            self.metadata = joblib.load(metadata_path)
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
                print("Model SVM, Scaler, Imputer, dan Metadata berhasil diload!")
            else:
                print("⚠ imputer.pkl tidak ditemukan — NaN akan diisi median fallback")
        else:
            print(f"File .pkl tidak ditemukan di {models_dir}")
 
    def run_detection_pipeline(self, extracted_cells, bounding_boxes_sep,
                               cell_masks_list,
                               preprocessed_image_rgb,
                               output_dir, patient_id):
 
        if self.model is None or self.scaler is None or self.metadata is None:
            raise Exception("Model SVM tidak ditemukan! Pastikan file pkl ada di folder source/models/")
 
        selected_features = self.metadata.get('features', [])
        if not selected_features:
            raise Exception("Metadata tidak memiliki daftar 'features'.")
            
        df_features, _, _ = run_feature_extraction(
            extracted_cells, bounding_boxes_sep, cell_masks_list,
            preprocessed_image_rgb.shape
        )
        if df_features.empty:
            raise Exception("Feature extraction tidak menghasilkan data.")
 
        for col in FEATURE_COLUMNS_40:
            if col not in df_features.columns:
                df_features[col] = 0.0
 
        X_40 = df_features[FEATURE_COLUMNS_40].replace([np.inf, -np.inf], np.nan)
 
        if self.imputer is not None:
            X_40_imputed = pd.DataFrame(
                self.imputer.transform(X_40),
                columns=FEATURE_COLUMNS_40
            )
        else:
            # Fallback: fit imputer baru dari data sekarang
            from sklearn.impute import SimpleImputer
            fallback = SimpleImputer(strategy='median')
            X_40_imputed = pd.DataFrame(
                fallback.fit_transform(X_40),
                columns=FEATURE_COLUMNS_40
            )
 
        missing = [f for f in selected_features if f not in X_40_imputed.columns]
        if missing:
            raise Exception(f"Fitur hilang setelah imputasi: {missing}")
 
        X_20 = X_40_imputed[selected_features]
 
        X_scaled    = self.scaler.transform(X_20)
        predictions = self.model.predict(X_scaled)
 
        ida_count    = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())
 
        result_img = cv.cvtColor(preprocessed_image_rgb.copy(), cv.COLOR_RGB2BGR)
        n_counter  = 1
        i_counter  = 1
 
        for bbox, pred in zip(bounding_boxes_sep, predictions):
            x, y, w, h = bbox
            if pred == 1:
                color = (0, 0, 255)
                label = f"I{i_counter}"
                i_counter += 1
            else:
                color = (0, 255, 0)
                label = f"N{n_counter}"
                n_counter += 1
            cv.rectangle(result_img, (x, y), (x + w, y + h), color, 5)
            cv.putText(result_img, label, (x, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
        res_path = os.path.join(output_dir, f"detection_result_{patient_id}.jpg")
        cv.imwrite(res_path, result_img)
 
        top5 = selected_features[:5] if len(selected_features) >= 5 else selected_features
        return res_path, ida_count, normal_count, top5
