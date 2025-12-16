# HaGRIDv2
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/palm.zip
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop.zip
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop_inverted.zip
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/fist.zip
unzip palm.zip
unzip stop.zip
unzip stop_inverted.zip
unzip fist.zip
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip annotations.zip

# Open/Closed Hands Dataset (1k images): https://universe.roboflow.com/hand-gestures-tytyx/hand-gestures-9iwxv/dataset/6/download
unzip "Hand Gestures.v6i.coco.zip"

# ASL and RPS datasets
python3 dataset_downloads/asl_1_dataset.py
python3 dataset_downloads/asl_2_dataset.py
python3 dataset_downloads/rps_dataset.py