# BWSI Build a CubeSat — Lunar Soil Strength Mapping

**2nd Place out of 50 Teams | Beaver Works Award Recipient**

A custom machine learning system developed during MIT Beaver Works Summer Institute (BWSI) to estimate lunar soil properties and assess potential landing and construction sites using NASA lunar imagery.

## Demonstration

[Watch the Model Demonstration](YOUR_VIDEO_LINK)

## Overview

The system processes lunar imagery through a custom PyTorch U-Net to predict three soil properties at the pixel level:

- Cohesion
- Internal friction angle
- Bulk density

These predictions are integrated with Terzaghi bearing-capacity equations to generate numerical soil-strength grids and RGB heatmaps highlighting relatively stronger and weaker terrain.

## Technical Implementation

### Dataset & Preprocessing
- Sourced high-resolution lunar imagery from NASA databases.
- Developed automated Python scripts to crop oversized images into training patches, overcoming local computational limitations.
- Implemented automatic RGB-to-grayscale conversion and preprocessing.
- Created a synthetic training dataset using rule-based approximations to generate soil-property labels.
- Built a reusable preprocessing pipeline capable of handling images of varying dimensions.

### Feature Engineering
- Used OpenCV to generate five grayscale-derived feature maps, including intensity, gradient magnitude, Laplacian, local variance, and high-frequency detail.
- Combined these maps into a five-channel tensor representing surface texture and structural patterns.

### Custom Neural Network
- Designed and implemented a custom U-Net-style encoder–decoder CNN in PyTorch.
- Used double-convolution blocks with 3×3 kernels to extract increasingly complex spatial features.
- Implemented an encoder with increasing feature channels (64 → 128 → 256 → 512).
- Used a 512-channel bottleneck to capture broader terrain context.
- Integrated transposed convolutions and skip connections to reconstruct high-resolution predictions while preserving spatial details.
- Produced three-channel, per-pixel predictions of cohesion, friction angle, and density.

### Training & Optimization
- Independently implemented the training process in Python using PyTorch.
- Conducted approximately two weeks of iterative training and optimization.
- Used an exploratory training strategy with higher learning rates to search different regions of the neural network's weight space.
- Progressively reduced learning rates to fine-tune promising weight configurations and compare their performance.
- Evaluated approximately 30 training candidates before selecting the best-performing model.
- Recorded a final MSE of approximately 1,600 against synthetic targets.

### Physics & Visualization
- Applied Terzaghi-based bearing-capacity calculations to predicted soil parameters.
- Generated numerical strength grids containing estimated bearing-capacity values.
- Produced individual cohesion, friction-angle, and density maps.
- Visualized relative terrain strength through RGB heatmaps, with blue representing weaker regions and red representing stronger regions.

## My Contributions

Personally developed the machine learning pipeline, including:

- Automated NASA imagery preprocessing and cropping scripts
- Synthetic dataset and training-label generation
- OpenCV feature extraction
- Custom PyTorch CNN architecture
- Python training pipeline and learning-rate optimization strategy
- Model evaluation and checkpoint selection
- Pixel-level soil-property prediction and visualization pipeline

## Results

Developed an end-to-end system that transforms NASA lunar imagery into estimated soil-property distributions and numerical terrain-strength maps.

**Project Visualizations**
- Original NASA lunar imagery
- Extracted feature maps
- Predicted soil-property maps
- Numerical strength grids
- Final RGB terrain-strength heatmaps
- Training loss graphs

## Technologies

Python · PyTorch · OpenCV · NumPy · CNNs · Image Processing · Geotechnical Physics

## Limitations

The model was trained on synthetic soil-property labels rather than direct lunar measurements. Predictions therefore represent estimated relative terrain characteristics, not experimentally validated lunar bearing-capacity measurements.

## Project Structure

- `ML/` — Custom CNN architecture, synthetic data generation, feature extraction, training, and model weights.
- `Data_management/` — Automated image tiling, resolution reduction, and grayscale preprocessing.
- `IMG_converter/` — Image conversion utilities.
- `Inference.py` — Model inference.
- `Watchman.py` — Project utility script.
- `docs/` — Usage instructions and supporting calculations.
- `results/` — Example predictions, terrain strength maps, and training results.
