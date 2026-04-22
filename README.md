# 🧠 Alzheimer’s / Dementia Voice Screening using Deep Learning

An AI-powered voice analysis system for early cognitive screening that classifies uploaded speech samples as:

- Healthy-like Speech
- Dementia-like Speech

This project uses speech biomarkers, MFCC audio features, and a Convolutional Neural Network (CNN) to analyze vocal patterns associated with neurological decline.

> ⚠️ This project is intended for academic and research purposes only. It is **not** a clinical diagnostic tool.

---

# 📌 Project Highlights

✅ Deep Learning based voice classification  
✅ Upload audio or record live voice  
✅ Waveform visualization  
✅ MFCC spectrogram visualization  
✅ Modern Streamlit dashboard UI  
✅ Local deployment supported  
✅ Real-time prediction system

---

# 🧠 Problem Statement

Alzheimer’s disease and related dementias often affect:

- speech fluency  
- pauses  
- articulation  
- memory-linked verbal behavior  
- vocal rhythm patterns

This project explores whether machine learning can detect such patterns from voice samples for early screening support.

---

# ⚙️ Tech Stack

## Frontend / App

- Python
- Streamlit

## Machine Learning

- TensorFlow / Keras
- CNN (Convolutional Neural Network)

## Audio Processing

- Librosa
- NumPy

## Visualization

- Matplotlib

---

# 🧬 Model Pipeline

## Input

Speech audio files:

- `.wav`
- `.mp3`
- `.flac`
- `.m4a`

## Feature Extraction

MFCC (Mel Frequency Cepstral Coefficients)

- 40 coefficients extracted
- padded/cropped to fixed shape

## Model

CNN Architecture:

- Conv2D layers
- MaxPooling
- Dense layers
- Sigmoid binary output

## Output

Probability score:

- Dementia-like speech
- Healthy-like speech

---

# 🖥️ Application Features

## User Interface

- Login page
- Signup page
- Prediction dashboard

## Prediction Module

- Upload audio file
- Record microphone input
- Real-time inference

## Visual Outputs

- Voice waveform
- MFCC feature map
- Confidence score
- Prediction probability

---

# 📂 Project Structure

```text
project-folder/
│── main.py
│── alz_cnn.keras
│── requirements.txt
│── README.md

Author
Piyush Nandre.
