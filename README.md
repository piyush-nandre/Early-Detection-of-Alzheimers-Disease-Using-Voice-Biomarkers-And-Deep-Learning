# 🧠 Alzheimer’s / Dementia Voice Screening using Deep Learning

An AI-powered voice analysis system for early cognitive screening that classifies uploaded speech samples as:

- Healthy-like Speech
- Dementia-like Speech

This project uses speech biomarkers, MFCC audio features, and a Convolutional Neural Network (CNN) to analyze vocal patterns associated with neurological decline.

> ⚠️ This project is intended for academic and research purposes only. It is **not** a clinical diagnostic tool.

 ## Dataset Notice

This project was developed using a controlled-access clinical dataset (DementiaBank / TalkBank) under applicable research-use terms.

Due to licensing, privacy, and access restrictions, the training data and trained weights are not redistributed in this repository.

Researchers should request dataset access directly from TalkBank/DementiaBank and train models independently.

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
<img width="1920" height="951" alt="1" src="https://github.com/user-attachments/assets/07c4654a-2362-4ba1-8466-15b5cb8f2067" />
<img width="1920" height="952" alt="2" src="https://github.com/user-attachments/assets/50d81bf1-8665-4f1f-92d2-11402042e255" />
<img width="1920" height="952" alt="3" src="https://github.com/user-attachments/assets/90c89880-7eae-499e-91c6-25c039ab85ed" />
<img width="1920" height="952" alt="4" src="https://github.com/user-attachments/assets/81add689-8a02-4b7d-956e-265f9b2aa548" />


---

# 📂 Project Structure

```text
project-folder/
│── main.py
│── requirements.txt
│── README.md
