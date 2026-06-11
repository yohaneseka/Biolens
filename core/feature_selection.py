import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif

def select_features_mi(
    df,
    result_dir,
    patient_name,
    target_col="Cell_Label",          
    threshold_quantile=0.5,            
    random_state=42
):
    meta_cols = ["Cell_Label", "Cell_Label_Name", "Source_Image", "X", "Y", target_col]
    exclude_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_feat = df[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    y_feat = df[target_col]

    mi_scores = mutual_info_classif(X_feat, y_feat, random_state=random_state)
    mi_results = (
        pd.DataFrame({"Feature": feature_cols, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    mi_threshold = mi_results["MI_Score"].quantile(threshold_quantile)
    selected_features = mi_results[mi_results["MI_Score"] >= mi_threshold]["Feature"].tolist()

    os.makedirs(result_dir, exist_ok=True)
    excel_sel_path = os.path.join(result_dir, f"features_selected_{patient_name}.xlsx")
    excel_mi_path  = os.path.join(result_dir, f"mutual_info_{patient_name}.xlsx")

    keep_meta = [c for c in ["Cell_Label", "X", "Y"] if c in df.columns]
    df_selected = df[keep_meta + selected_features + [target_col] if target_col not in keep_meta else keep_meta + selected_features]
    df_selected.to_excel(excel_sel_path, index=False)
    mi_results.to_excel(excel_mi_path, index=False)

    top5 = mi_results.head(5)["Feature"].tolist()

    print(f"[MI] Threshold (quantile {threshold_quantile*100:.0f}%): {mi_threshold:.6f}")
    print(f"[MI] Fitur terpilih: {len(selected_features)} dari {len(feature_cols)}")
    print(f"[MI] Top-5: {top5}")
    return selected_features, mi_results, top5
