# Cloud Removal GAN Project

This project uses a Generative Adversarial Network (GAN) to remove artificial clouds from satellite images. It includes:
- Perlin noise-based cloud generation
- Custom PyTorch GAN model
- Training loop with PSNR/SSIM evaluation
- Flask API for model inference

## Usage
1. Train the model:
```bash
python train.py
```
2. Run the Flask server:
```bash
python app.py
```
3. Send a POST request to `/predict` with an image.

## Accuracy
- Achieved approx **50%** reconstruction accuracy using SSIM/PSNR metrics.

## Author
Vivek Kumar
