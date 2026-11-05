# Bahraini Traditional Food Object Detection

A computer vision project that detects and identifies **10 Bahraini and local food categories** in images using a fine-tuned **YOLO26s** object detection model. The project includes dataset preparation in Roboflow, model training and evaluation in Google Colab, and a simple Streamlit application for predictions using an uploaded image or a camera.

## Project Overview

The goal is to recognize and locate traditional food items in photographs by drawing bounding boxes around detected objects and displaying their predicted class names and confidence scores. This project explores the complete object detection workflow, from labeling images to evaluating predictions and deploying a working prototype.

## Food Categories

The dataset contains these 10 classes:

1. Balaleet
2. Halwa
3. Harees
4. Lugaimat
5. Machboos
6. Nakaj
7. Saif
8. Saloona
9. Samboosa
10. Tekka Lemon

## Dataset

Images were prepared and annotated with bounding boxes using **Roboflow** and exported in YOLO format. The dataset version used for training contained **1,175 images**, including augmented training images.

| Split | Images | Share |
| --- | ---: | ---: |
| Training | 1,075 | 91.5% |
| Validation | 64 | 5.4% |
| Testing | 36 | 3.1% |
| **Total** | **1,175** | **100%** |

The training set is substantially larger than the validation and test sets. The test set contains only 36 images, so its results should be interpreted cautiously, especially for food classes with very few examples.

### Preprocessing

- Auto-orientation applied to images.
- Resized to **512 × 512** using fill with center crop.

### Data Augmentation

Roboflow was configured to generate up to **5 outputs per training example**, with:

- 90° clockwise and counterclockwise rotation.
- Blur up to 2 pixels.
- Motion blur (length 30 pixels, angle 0°).

Augmentation adds variation to the training data. However, transformed copies of the same image are not completely independent examples, and heavy rotation or blur may not always resemble real-world photos.

### Class Balance

The Roboflow class-count view showed uneven representation across the categories. **Samboosa (340)** and **Lugaimat (292)** had much higher displayed counts than classes such as **Machboos (31)**, **Balaleet (32)**, and **Harees (36)**. These counts are from the displayed class-summary view and should not be treated as a verified count of original images per class. Such imbalance may contribute to differences in detection performance.

## Model Training

- **Model:** Ultralytics YOLO26s (pretrained weights: `yolo26s.pt`).
- **Environment:** Google Colab, using Python and Ultralytics.
- **Dataset configuration:** Roboflow YOLO dataset (`data.yaml`).
- **Initial training:** 10 epochs.
- **Continued training:** a further 15 epochs, starting from a saved `last.pt` checkpoint.
- **Final evaluation checkpoint:** `best.pt` from the continued training run (`train-3` in the notebook).
- 
## Model Evaluation
<img width="3000" height="2250" alt="download" src="https://github.com/user-attachments/assets/c657b1dd-9c07-4701-94e4-49273432d9af" />

The trained detector was evaluated on the **36-image test split**, containing **109 annotated food instances**, using Ultralytics validation metrics.

| Metric | Test result | Meaning |
| --- | ---: | --- |
| Precision | **77.6%** | How often the reported detections were correct |
| Recall | **74.6%** | How many labeled objects were found |
| mAP@50 | **79.1%** | Average precision at an IoU threshold of 0.50 |
| mAP@50–95 | **70.7%** | Average precision across IoU thresholds from 0.50 to 0.95 |

**IoU (Intersection over Union)** measures how much a predicted bounding box overlaps its ground-truth box. Higher IoU means more accurate box placement. The metrics above are aggregate evaluation scores, not the model's accuracy on every individual image.
<img width="3000" height="2250" alt="download" src="https://github.com/user-attachments/assets/ebff2401-4415-45fe-b425-657f92da7cdc" />


### Per-Class Test Results

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| Balaleet | 80.2% | 100.0% | 99.5% | 99.5% |
| Halwa | 100.0% | 75.7% | 87.2% | 80.3% |
| Harees | 100.0% | 96.7% | 99.5% | 96.4% |
| Lugaimat | 100.0% | 86.2% | 96.5% | 85.1% |
| Machboos | 68.0% | 100.0% | 99.5% | 99.5% |
| Nakaj | 48.3% | 20.2% | 32.9% | 17.8% |
| Saif | 84.3% | 41.6% | 59.1% | 37.7% |
| Saloona | 87.3% | 100.0% | 99.5% | 96.2% |
| Samboosa | 92.0% | 92.1% | 97.1% | 76.6% |
| Tekka Lemon | 15.8% | 33.3% | 20.2% | 17.6% |

Some high class scores are based on only **2–5 test images**, so they should not be interpreted as evidence of perfect performance on new images.

### Error Analysis and Challenges

- **Nakaj** had low precision and recall, suggesting both incorrect predictions and missed objects.
- **Tekka Lemon** was the weakest class in the test results, with only 15.8% precision and 33.3% recall.
- **Saif** had moderate precision but low recall, meaning many actual instances were missed.
- The confusion matrix shows confusion between some food classes and unmatched detections or missed objects represented by the **background** row/column. Background is **not** an additional food class.
- Class imbalance, limited data diversity, and the small test set are possible explanations for unstable results, but would require further experiments to confirm.

These findings identify areas for future improvement. No additional targeted-data experiment or before/after model comparison is claimed here.

## Streamlit Application

The app allows users to:

1. Take a photo with a camera **or** upload a JPG, JPEG, or PNG image.
2. Click **Predict**.
3. View the detected foods with bounding boxes and confidence scores.

### Run Locally

Place the trained `best.pt` model in the same directory as `app.py`, then install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501` by default.

### Main Libraries

- `ultralytics` — YOLO model training and inference.
- `roboflow` — dataset export and management in the notebook.
- `streamlit` — user interface.
- `Pillow` and `numpy` — image loading and processing.

## Limitations and Future Improvements

- Collect more original photographs for underrepresented or difficult classes.
- Include more lighting conditions, viewpoints, serving styles, and complex backgrounds.
- Review bounding-box labels and examples where objects are confused.
- Increase the size of the untouched test set to obtain more reliable metrics.
- Compare new experiments against the current model using the same test set.

## Conclusion

This project demonstrates an end-to-end custom object detection pipeline for Bahraini traditional food. The YOLO26s detector achieved **77.6% precision, 74.6% recall, and 79.1% mAP@50** on the supplied test set. While several categories performed well, Nakaj, Saif, and Tekka Lemon were more challenging. The Streamlit prototype makes it possible to test the trained model through a simple image-upload or camera interface.
