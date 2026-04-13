"""
Diagnostic script: Analyze YG model feature importance.
"""
import sys
import os
from pathlib import Path

# Project root is parent of scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add global Python site-packages for lightgbm
python311_path = Path(r"C:\Users\andre\AppData\Local\Programs\Python\Python311\Lib\site-packages")
if str(python311_path) not in sys.path:
    sys.path.insert(0, str(python311_path))

import numpy as np
import pandas as pd
import joblib

# Load config directly without importing llm_tool package (to avoid shap/ sklearn issues)
def load_config_directly():
    """Load only YG config values without triggering package imports."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', project_root / 'llm_tool' / 'config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

config = load_config_directly()
YG_MODEL_PATH = config.YG_MODEL_PATH


def diagnose_importance():
    print("=" * 60)
    print("DIAGNOSTIC: YG Model - Feature Importance Analysis")
    print("=" * 60)
    
    # Load the model directly
    model = joblib.load(YG_MODEL_PATH)
    
    print(f"\nModel type: {type(model)}")
    print(f"Model classes: {model.classes_}")
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Feature names from training
        feature_names = [
            "zone", "month", "quarter",
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            "vehicle_type", "service_mode",
        ]
        
        # Print importance
        print("\nFeature Importances:")
        print("-" * 40)
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        for _, row in importance_df.iterrows():
            print(f"{row['feature']:15s}: {row['importance']:.4f}")
        
        print("-" * 40)
        
        # Check if hour features have low importance
        hour_features = importance_df[importance_df["feature"].isin(["hour_sin", "hour_cos"])]
        total_importance = importance_df["importance"].sum()
        hour_importance = hour_features["importance"].sum()
        hour_pct = (hour_importance / total_importance * 100) if total_importance > 0 else 0
        
        print(f"\nHour features (hour_sin + hour_cos) importance: {hour_importance:.4f} ({hour_pct:.1f}% of total)")
        
        if hour_importance < 0.01:
            print("\nWARNING: Hour features have very low importance!")
            print("This may explain why model produces same predictions across hours.")
            return False
        else:
            print("\nHour features have reasonable importance.")
            return True
    else:
        print("\nModel does not have feature_importances_ attribute")
        print("Checking booster...")

        if hasattr(model, 'booster_'):
            booster = model.booster_
            if hasattr(booster, 'feature_importance'):
                imp = booster.feature_importance()
                print("\nBooster feature importance:")
                print(imp)
        
        return None

if __name__ == "__main__":
    success = diagnose_importance()
    print("\n" + "=" * 60)
    if success:
        print("RESULT: Hour features have SIGNIFICANT importance")
    elif success is False:
        print("RESULT: Hour features have LOW importance - ISSUE DETECTED!")
    else:
        print("RESULT: Could not determine feature importance")
    print("=" * 60)