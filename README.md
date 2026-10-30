# Automated Satellite Detection in Amateur Astronomical Images

This repository contains a complete research pipeline for detecting satellite streaks in amateur astronomical images using computer vision and YOLO-based object detection.

The project includes scripts for:

- preparing and converting ASTA datasets into YOLO format,
- generating synthetic training images,
- splitting datasets into train/val/test sets,
- training detection models,
- validating models on labeled test images,
- running detection on new images from folders.

The goal is to detect thin satellite trails (streaks) in night-sky images and produce bounding-box predictions for each visible object.

---

## Repository overview

This project is organized around a few main groups of files:

### 1) Dataset preparation and conversion

These scripts convert raw astronomical data into a YOLO-style dataset structure.

- [convert_ASTA_to_detection_format.py](convert_ASTA_to_detection_format.py)  
  Converts ASTA mask images into YOLO detection labels using bounding boxes around detected streaks.

- [split_asta_dataset.py](split_asta_dataset.py)  
  Finds matching image-mask pairs, converts masks to YOLO polygon/text labels, and splits them into train/val/test folders.

- [split_asta_datset_check.py](split_asta_datset_check.py)  
  Verifies that ASTA conversion and label format are correct.

### 2) Synthetic data generation

These scripts generate artificial night-sky images with stars and satellite trails to improve training.

- [generate_syn_dataset_script.py](generate_syn_dataset_script.py)  
  Main procedural synthetic dataset generator.

- [generate_synetitic_images.py](generate_synetitic_images.py)  
  Another synthetic image generator with similar purpose.

- [generate_synthetic_images_version2.py](generate_synthetic_images_version2.py)  
  Faster and more simple generator/

- [generate_syn_dataset_each_layer_explained.py](generate_syn_dataset_each_layer_explained.py)  
  Explains how the synthetic dataset is built layer by layer.

- [generate_crossing_satelite_img.py](generate_crossing_satelite_img.py)  
  Produces examples of crossing satellite streaks.

### 3) Model training and evaluation

These scripts train and validate detection models.

- [Models_training/detection_yolo_train.py](Models_training/detection_yolo_train.py)  
  Trains a YOLOv8 object detection model on astronomical data.

- [Models_training/unet_train.py](Models_training/unet_train.py)  
  Training script for a segmentation-style YOLO model.

- [Models_training/rtdert_train.py](Models_training/rtdert_train.py)  
  RT-DETR training workflow.

- [test_model.py](test_model.py)  
  Loads a trained model and evaluates it on a validation/test dataset.

### 4) Prediction and inference

These scripts run the trained models on new images.

- [Models_predict/predict.py](Models_predict/predict.py)  
  Basic YOLO prediction script.

- [Models_predict/predict_final.py](Models_predict/predict_final.py)  
  Main inference script used for final prediction runs on image folders.

- [Models_predict/predict_detection_sahi.py](Models_predict/predict_detection_sahi.py)  
  SAHI-based detection pipeline for larger images.

- [Models_predict/preprocessing_yolo.py](Models_predict/preprocessing_yolo.py)  
  Applies preprocessing before YOLO inference.

- [Models_predict/predict_satelite_endpoints.py](Models_predict/predict_satelite_endpoints.py)  
  Extracts satellite streak endpoints and angle estimates from detections.

- [Models_predict/hough.py](Models_predict/hough.py)  
  Hough-transform based analysis for detected streak-like structures.

- [Models_predict/opticalflow_sequence.py](Models_predict/opticalflow_sequence.py)  
  Sequence-based motion analysis experiments.

- [Models_predict/opticalflow_twophotos.py](Models_predict/opticalflow_twophotos.py)  
  Two-frame motion analysis experiments.

### 5) Main configuration files

These YAML files define dataset paths for YOLO training and validation.

- [asta.yaml](asta.yaml)  
  Configuration for the ASTA dataset.

- [synthetic.yaml](synthetic.yaml)  
  Configuration for synthetic-data training.

- [streaks.yaml](streaks.yaml)  
  Another dataset configuration used in streak experiments.

- [syn_streaks.yaml](syn_streaks.yaml)  
  Synthetic streak dataset configuration.

- [combined.yaml](combined.yaml)  
  Combined-dataset configuration.

- [test_dataset.yaml](test_dataset.yaml)  
  Test-set configuration used for evaluation.

---

## Typical workflow

A standard workflow in this repository looks like this:

1.Generate synthetic data.
   - Use [generate_syn_dataset_script.py](generate_syn_dataset_script.py) or [generate_synthetic_images_version2.py](generate_synthetic_images_version2.py).

4. Train the model.
   - Use [Models_training/detection_yolo_train.py](Models_training/detection_yolo_train.py).

5. Validate the trained model.
   - Run [test_model.py](test_model.py).

6. Run inference on new images.
   - Use [Models_predict/predict_final.py](Models_predict/predict_final.py) or similar scripts in [Models_predict](Models_predict).

---

## Expected outputs

When the pipeline runs, the project typically produces:

- train/val/test dataset folders with image and label pairs,
- YOLO model weights
- experiment folders (in folder named run)
- prediction visualizations
---

## Notes

- This project is built around research and experimentation, so some scripts are utilities, intermediate experiments, or older versions.
- Not every file in the repository is used in the final pipeline; several scripts are kept to compare methods or generate additional data.
- The training and dataset YAML files often point to absolute paths, so the repo may need local path adjustment depending on the machine or server where it is run.

---

## Quick start example

A typical training command may look like this:

```bash
python Models_training/detection_yolo_train.py
```

And a typical evaluation command:

```bash
python test_model.py
```

And a typical prediction run:

```bash
python Models_predict/predict_final.py
```

---
