## ASL DATASET 2 ## 
# Download dataset from Kaggle
# Reassign letters to new labels 
# Copy images to new label folders

import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

raw_dataset_dir = "./data/asl_2_dataset_raw"
labeled_dataset_dir = "./data/asl_2_dataset_labeled"  

# Letters to include
label_mapping = {
    "A": 1, "E": 1, "M": 1, "N": 1, "O": 1, "S": 1, "T": 1,  
    "B": 0, "C": 0, "F": 0                                   
}


kaggle_dataset = "ashish8898/hand-gestures"

os.makedirs(raw_dataset_dir, exist_ok=True)
os.makedirs(labeled_dataset_dir, exist_ok=True)
for label in [0, 1]:
    os.makedirs(os.path.join(labeled_dataset_dir, str(label)), exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Downloading and unzipping raw dataset...")
api.dataset_download_files(kaggle_dataset, path=raw_dataset_dir, unzip=True)

for letter, label in label_mapping.items():
    train_src_folder = os.path.join(raw_dataset_dir, "train", letter)
    test_src_folder = os.path.join(raw_dataset_dir, "test", letter)
    dst_folder = os.path.join(labeled_dataset_dir, str(label))
    
    if not os.path.exists(train_src_folder):
        print(f"Warning: folder {train_src_folder} not found, skipping")
        continue
    
    print(f"Copying images from train {letter} to label {label} folder...")
    for filename in os.listdir(train_src_folder):
        src_file = os.path.join(train_src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        shutil.copy2(src_file, dst_file)

    if not os.path.exists(test_src_folder):
        print(f"Warning: folder {test_src_folder} not found, skipping")
        continue

    print(f"Copying images from test {letter} to label {label} folder...")
    for filename in os.listdir(test_src_folder):
        src_file = os.path.join(test_src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        shutil.copy2(src_file, dst_file)

    

print("Done! Labeled dataset is ready at:", labeled_dataset_dir)