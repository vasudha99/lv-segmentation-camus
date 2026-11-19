# Cardiac Ultrasound Segmentation using U-Net

**Python | TensorFlow/Keras | Deep Learning | Medical Image Segmentation**

This project applies a four-class U-Net to echocardiography images from the CAMUS dataset. It builds on my MSc Artificial Intelligence dissertation work at the University of West London. The notebook covers dataset inspection, patient-level splitting, preprocessing, training, and evaluation.

## Project Overview

The task is semantic segmentation: the model assigns a class to each pixel of an ultrasound image. The four labels are background (0), left ventricular (LV) cavity (1), myocardium (2), and left atrium (3). Predictions are compared with expert-annotated masks.

## Technologies

- **Python, NumPy and Pandas:** image arrays and dataset records
- **SimpleITK and OpenCV:** reading CAMUS medical images and resizing images and masks
- **Scikit-learn:** separating patients into training, validation, and test sets
- **TensorFlow/Keras:** U-Net implementation and training
- **Matplotlib:** image, prediction, and training-curve visualisations
- **Google Colab:** notebook execution

## Dataset

Source: [CAMUS — Cardiac Acquisitions for Multi-structure Ultrasound Segmentation](https://www.creatis.insa-lyon.fr/Challenge/camus/)

The experiment uses **1,800 annotated frames from 450 patients**, with two- and four-chamber views at end-diastole and end-systole. Each ultrasound image is paired with an expert-labelled mask. The CAMUS dataset must be downloaded separately; the original images are not included in this repository.

## Methodology

1. **Inspect the annotations.** Display a source ultrasound image, its labelled mask, and an overlay to check correspondence.
2. **Match images and masks.** Pair CAMUS `.mhd` images with the matching `_gt.mhd` annotations, keeping the patient identifier for each frame.
3. **Split by patient.** Use `GroupShuffleSplit` so that frames from a patient cannot appear in multiple subsets. The recorded split contains 315 training patients (1,260 frames), 67 validation patients (268 frames), and 68 test patients (272 frames).
4. **Preprocess the data.** Resize grayscale images to 256 × 256 and normalise each image to the range [0, 1]. Resize masks with nearest-neighbour interpolation to preserve their integer class labels.
5. **Build U-Net.** The encoder extracts features at progressively smaller resolutions; the decoder upsamples them, using skip connections to recover spatial detail. The final softmax layer produces four class probabilities per pixel.
6. **Train.** Use Adam with a learning rate of 0.0001, batch size four, and a maximum of 60 epochs. The loss combines sparse categorical cross-entropy with foreground soft Dice loss. Validation Dice selects the best checkpoint; early stopping and learning-rate reduction monitor validation performance.
7. **Evaluate.** Choose the most probable class for each test pixel, then calculate Dice and intersection over union (IoU) for each foreground structure. Average per-image scores within each class, then average the three class means. Empty ground-truth/prediction pairs are excluded from that class's average.
8. **Inspect predictions.** Compare ultrasound images, expert labels, predicted masks, and overlays; plot training and validation Dice and loss.

## Results

The **recorded outputs from the completed experiment**, evaluated on 272 frames from 68 held-out patients, report:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| LV cavity Dice | 0.9262 | 0.0443 |
| Myocardium Dice | 0.8565 | 0.0578 |
| Left atrium Dice | 0.8814 | 0.1262 |
| **Mean anatomical Dice** | **0.8880** | — |
| **Mean anatomical IoU** | **0.8077** | — |

The anatomical averages are **macro-averages across the three foreground classes**, not averages including background. The LV cavity has the highest mean Dice. The left atrium shows the greatest variation in Dice across test images. These are results recorded in the earlier completed notebook; the cleaned notebook has not been retrained.

## Project Files

- [`notebooks/01_CAMUS_UNet.ipynb`](notebooks/01_CAMUS_UNet.ipynb): Full four-class experiment, with comments explaining the purpose of each stage
- [`requirements.txt`](requirements.txt): Required Python libraries

## Running the Project

1. Download CAMUS from the dataset link above.
2. Open the notebook in Google Colab and connect Google Drive.
3. Place the CAMUS `training` directory at `MyDrive/ComputerVisionProject/Original Camus Training Dataset/training`, or update `DATA_DIR` in the setup cell.
4. Run the cells in order. Model weights, evaluation tables, prediction masks, and figures are written to the configured `improved_unet_results` folder.