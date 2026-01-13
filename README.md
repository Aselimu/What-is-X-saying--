# AI-powered voice emotion analyzer for babies and cats.

## Overview

Analyzes vocalizations from babies and pets to understand their emotional states. It uses MFCC feature extraction with CNN classification, then generates natural language interpretations using an LLM.

## Features

- **Baby Cry Analysis** - Detect emotions from baby crying sounds
- **Cat Voice Analysis** - Understand cat meows and vocalizations
- **Deep Learning** - CNN-based audio classification
- **Natural Language Output** - Human-friendly interpretations via Qwen LLM

## Project Structure

```
├── baby/
│   ├── train_baby_cnn.ipynb       # Training notebook
│   ├── run_baby_local.ipynb       # Inference notebook
│   └── emotion_cnn_model_baby.h5  # Trained model
│
├── cat/
│   ├── train_cat_cnn.ipynb        # Training notebook
│   ├── run_cat_local.ipynb        # Inference notebook
│   └── mfcc_cnn_model_cat.h5      # Trained model
```

## Requirements

```
numpy
librosa
scikit-learn
tensorflow
transformers
torch
joblib

```

## Installation

```bash
git clone https://github.com/Aselimu/What-is-X-saying--.git
cd What-is-X-saying--
pip install numpy librosa scikit-learn tensorflow transformers torch joblib
```

## Usage

### Baby Analysis

1. Open `baby/run_baby_local.ipynb`
2. Set your audio file path:
   ```python
   AUDIO_FILE_PATH = "your_baby_audio.mp3"
   ```
3. Run all cells

### Cat Analysis

1. Open `cat/run_cat_local.ipynb`
2. Set your audio file path:
   ```python
   AUDIO_FILE_PATH = "your_cat_audio.mp3"
   ```
3. Run all cells

## How It Works

1. **Audio Input** - Load WAV or MP3 file
2. **MFCC Extraction** - Extract 40 mel-frequency cepstral coefficients
3. **CNN Prediction** - Classify emotion/state using trained model
4. **LLM Interpretation** - Generate natural language explanation

## Model Architecture

```
Input (MFCC Features)
    ↓
Conv1D → MaxPooling → Dropout
    ↓
Conv1D → MaxPooling
    ↓
Flatten → Dense → Dropout
    ↓
Output (Softmax)
```

## Training

Both models were trained with:
- **Epochs**: 100
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Test Split**: 20%

## Dataset 

**Baby -> [URL= {https://www.kaggle.com/datasets/mennaahmed23/baby-cry-sense-dataset}]** 

**Cat  -> "Private Datset [URL = {https://www.mdpi.com/2076-3417/8/10/1949}, DOI = {10.3390/app8101949}]"**


## Supported Formats

- WAV (.wav)
- MP3 (.mp3)

  
