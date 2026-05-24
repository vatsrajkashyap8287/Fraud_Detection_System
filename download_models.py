import gdown
import os

FILES = {
    "lgbm_model.pkl":     "13aM7g-xYMGOMRH_IGx6cVfOzIdJHJLxA",
    "shap_explainer.pkl": "1A34RIQlzKMiWxpv8YokvT5fSKMuvD2PC",
    "scaler.pkl":         "1mKFKXV3DXW7tc25uA72WC8IMvkfZl1G3",
    "feature_names.pkl":  "1Uq4GxkY1fdVry8md2zM56hZTrHTT3dRI",
    "test_results.csv":   "19fI7ji7Y8lhsb_VFKh9EmnYKMYc9OL0X",
}

for filename, file_id in FILES.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filename,
            quiet=False
        )
        print(f"{filename} downloaded!")
    else:
        print(f"{filename} already exists, skipping.")