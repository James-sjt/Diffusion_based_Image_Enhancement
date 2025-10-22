# Diffusion-Based Image Enhancement Model

This repository contains a **Diffusion-Based Image Enhancement Model** designed to improve the quality of low-contrast images by enhancing the edges of features. The model is capable of producing high-quality images that can be used as input for subsequent models such as classification or segmentation.

## Model Overview

- **Purpose**: The model uses a diffusion-based approach to enhance the features and edges of low-contrast images.
- **Key Application**: Image enhancement for downstream tasks like classification and segmentation.

---

## Setup and Deployment

Follow the steps below to set up, train, and deploy the model.

### 1. Download Dataset

To begin, you need to download the dataset used for training and evaluation. Run the following command:

```bash
python extractDataset.py
```
This script will automatically download the necessary dataset.

### 2. Train the VAE (Optional)

If you want to train the Variational Autoencoder (VAE) component of the model, run:

```bash
python VAE.py
```
If you don't need to train the VAE, you can skip this step, as the model will automatically load the pre-trained VAE parameters.

### 3. Verify the Performance of the VAE (Optional)

To evaluate the performance of the VAE and generate the saved model parameters, run:

```bash
python VAEInference.py
```
The trained parameters of the VAE will be saved as VAESSIM.pth.

### 4. Train the Latent Diffusion Model (LDM)

Next, train the Latent Diffusion Model (LDM), which is the core image enhancement component of the system:

```bash
python latentDiffusion.py
```

During training, you can monitor the loss on both the training and validation datasets using:

```bash
python plot.py
```
This script will generate plots to visualize the training and validation loss over time.

### 5. Inference with Latent Diffusion Model

After training the model, you can sample enhanced images using the following command:

```bash
python inferenceLDM.py
```
This will generate enhanced images from low-contrast inputs.

### 6. Construct Enhanced Image Dataset for Segmentation

Once you have the enhanced images, you can create a dataset for subsequent segmentation tasks by running:

```bash
python sampleEnhancedImg.py
```

This script will organize and store the enhanced images for later use.

### 7. Evaluate Model Performance

Finally, to evaluate the performance of the model in terms of image quality, you can compute metrics such as PSNR, SSIM, GMSD, and LPIPS by running:

```bash
python evaluationDenoise.py
```
This script will provide detailed performance metrics to assess the effectiveness of the enhanced images.

### Model Parameters
The model is designed to automatically load trained parameters if available. However, if you need to re-train any component, the following files will be generated:

VAE Parameters: Stored as VAESSIM.pth after VAE training.

Latent Diffusion Model Parameters: Saved during training at ./Diffusion_based_Image_Enhancement/ckpts_unet.

### Requirements

Python 3.8+

PyTorch 1.9+

HuggingFace's transformers library (for model deployment, if needed)

Other dependencies are listed in requirements.txt

Install required packages:
```bash
pip install -r requirements.txt
```

