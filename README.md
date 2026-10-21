# Left Ventricular Segmentation using U-Net

This project segments the left ventricular cavity in echocardiography images using a U-Net convolutional neural network. It uses the CAMUS dataset, which contains cardiac ultrasound images and expert segmentation masks. The model learns to predict the LV endocardium mask from each ultrasound image.

The project includes a notebook for exploring CAMUS images and annotations, followed by a training notebook that preprocesses the images, builds a U-Net, trains it, and evaluates segmentation predictions. The data is divided by patient into training, validation and test sets so that patients in the test set are not included during training.

## Dataset

Download the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) and place its original patient folders inside `data/camus/training/`. Each patient folder should contain the original `.mhd` files and their associated `.raw` files. The code reads the files locally; Google Drive is not required. CAMUS data use is subject to the dataset's research and attribution terms.

## Run

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

Open `notebooks/01_explore_camus.ipynb` to inspect the images. Run `notebooks/02_unet_patient_level_split.ipynb` to train and evaluate the model. Start Jupyter from the repository root, or set the `CAMUS_TRAINING_DIR` environment variable to your dataset's training folder. Training results, checkpoints and prediction images are saved in `results/`, which is excluded from Git.

The model uses 256 × 256 grayscale images and binary masks for the LV endocardium. Performance is calculated using Dice and Intersection over Union (IoU). The training notebook evaluates segmentation performance using Dice score and Intersection over Union (IoU), and saves the evaluation results and prediction visualisations.

## Tools

Python, TensorFlow/Keras, SimpleITK, OpenCV, NumPy, pandas, scikit-learn and Matplotlib.

## References

Leclerc et al. (2019), [Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography](https://doi.org/10.1109/TMI.2019.2900516).

Ronneberger et al. (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
