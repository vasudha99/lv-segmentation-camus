# Left Ventricular Segmentation using U-Net

This project uses a U-Net convolutional neural network to segment the left ventricular cavity from echocardiography images. It predicts the LV endocardium as a binary mask and compares the output with expert annotations from the CAMUS dataset.

## Dataset

The [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) contains echocardiography images in two-chamber and four-chamber views, with annotations at end-diastole and end-systole. The project reads the original `.mhd` images and their paired `.raw` files from `data/camus/training/patientXXXX/`. Place the training dataset in that directory before running the notebooks. CAMUS is distributed for non-commercial scientific research and requires attribution.

## Method

Images are resized to 256 × 256 pixels and normalised to the range 0–1. Masks are resized using nearest-neighbour interpolation, and the LV endocardium is extracted as the segmentation target. The U-Net uses an encoder-decoder architecture with skip connections. Training combines binary cross-entropy and Dice loss.

The baseline notebook demonstrates training and visualisation. The patient-level notebook separates patients across training, validation and testing, trains the model and evaluates predictions using Dice and IoU.

## Results

The initial baseline notebook reported a Dice score of approximately 0.868 using an image-level split. This is not a patient-independent test result. The patient-level experiment must be run before reporting independently evaluated metrics.

## Running the project

Install dependencies using `pip install -r requirements.txt`, open the notebooks from the repository root, and place CAMUS training data in `data/camus/training/`. A GPU is recommended for training.

## Technologies

Python, TensorFlow/Keras, SimpleITK, OpenCV, NumPy, pandas, scikit-learn and Matplotlib.

## References

Leclerc et al. (2019), [Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography](https://doi.org/10.1109/TMI.2019.2900516).

Ronneberger et al. (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
