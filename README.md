# Image Captioning

An end-to-end image captioning model built from scratch using PyTorch.

## How it works
- CNN encoder (ResNet50) extracts image features
- LSTM decoder generates captions word by word
- Trained on Flickr8k dataset (8000 images, 40000 captions)

## Project structure
- `src/dataset.py` — data pipeline and vocabulary
- `src/model.py` — encoder/decoder architecture
- `src/train.py` — training loop
- `src/evaluate.py` — evaluation and caption generation
- `app.py` — Streamlit demo

## Results
Model trained for 10 epochs, final loss: 2.75

## Run locally
pip install -r requirements.txt
streamlit run app.py
