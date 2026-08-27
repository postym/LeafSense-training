\# LeafSense AI ðŸƒ



An AI-powered leaf species identification system, trained to classify leaf specimens from 5 plant families using deep learning.



\## Overview

LeafSense uses a fine-tuned MobileNetV3-Large model to identify leaf species from photos, covering 10 leaf specimens across 5 plant families: \*\*Fabaceae, Moraceae, Euphorbiaceae, Arecaceae, and Malvaceae\*\*. The model is exported in multiple formats (PyTorch, ONNX, TFLite) for flexible deployment across platforms.



\## Dataset

\- 10 leaf specimens across 5 plant families

\- 900 raw images per specimen (500 train / 300 validation / 100 test) â†’ 9,000 raw images total

\- Each training/validation image was expanded into 7 augmented variations, growing the dataset to \*\*56,000+ images\*\*



\## Model Performance

\- \*\*Test accuracy: 80.29%\*\*

\- \*\*Weighted AUC: 0.9583\*\*

\- Formats available: PyTorch (`.pth`), ONNX (`.onnx`), TensorFlow Lite (`.tflite`)



\## Dataset \& Model Weights

The full leaf specimen dataset and trained model weights are hosted on Hugging Face (too large for GitHub):

ðŸ‘‰ \[postym/LeafSense\_Dataset on Hugging Face](https://huggingface.co/datasets/postym/LeafSense\_Dataset)



\## Project Structure

\- Training notebooks (`.ipynb`) â€” model training and experimentation

\- Evaluation results â€” confusion matrices, ROC curves, classification reports, threshold analysis

\- `model\_architecture.py` â€” model definition



\## Setup

1\. Clone this repo

2\. Create a virtual environment and install dependencies: `pip install -r requirements.txt`

3\. Download the dataset/models from the Hugging Face link above

4\. Run the training or inference notebooks



\## Results

See `training\_results.png`, `confusion\_matrix\_final\*.png`, and `overall\_metrics.png` for detailed per-class performance breakdowns.


