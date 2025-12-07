## HAND GESTURE RECOGNITION DATASET ## 
# Download dataset from Kaggle
# Reassign gestures to new labels 
# Copy images to new label folders
# Delete raw (full) dataset

import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

raw_dataset_dir = "./hgr_dataset_raw"
labeled_dataset_dir = "./hgr_dataset_labeled"  

# Gestures to include
label_mapping = {
    "closedFist": 1,  
    "openPalm": 0, "semiOpenPalm": 0                                   
}


kaggle_dataset = "ritikagiridhar/2000-hand-gestures"

os.makedirs(raw_dataset_dir, exist_ok=True)
os.makedirs(labeled_dataset_dir, exist_ok=True)
for label in [0, 1]:
    os.makedirs(os.path.join(labeled_dataset_dir, str(label)), exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Downloading and unzipping raw dataset...")
api.dataset_download_files(kaggle_dataset, path=raw_dataset_dir, unzip=True)

for gesture, label in label_mapping.items():
    src_folder = os.path.join(raw_dataset_dir, "images", gesture)
    dst_folder = os.path.join(labeled_dataset_dir, str(label))
    
    if not os.path.exists(src_folder):
        print(f"Warning: folder {src_folder} not found, skipping")
        continue
    
    print(f"Copying images from {gesture} to label {label} folder...")
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        shutil.copy2(src_file, dst_file)

print("Done! Labeled dataset is ready at:", labeled_dataset_dir)

## Remove raw dataset
print(f"Removing raw dataset folder: {raw_dataset_dir}")
shutil.rmtree(raw_dataset_dir)
print("Raw folder removed.")