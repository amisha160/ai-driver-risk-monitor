# 🚘 AI Driver Risk Monitor

An AI-powered smart driver behavior intelligence and adaptive risk 
profiling platform that analyzes multimodal driving data and predicts 
driver risk in real time through an interactive web dashboard.

## Overview

AI Driver Risk Monitor is built to improve road safety by combining 
multimodal inputs — sensor values (speed, acceleration, steering), 
driver condition text, and driver image analysis — within a single 
unified Logistic Regression algorithm. The system classifies driver 
behavior and generates adaptive risk scores through a modern dashboard.

>  Core Innovation: Multimodal driver behavior analysis 
> (image + video + text + sensor) unified under a single 
> machine learning algorithm — Logistic Regression.

## Key Features

-  Live Risk Meter – Real-time driver risk percentage visualization
-  Driver State Classification – Safe / Distracted / Drowsy / Aggressive
-  Smart Route Map – GPS location detection with destination routing
-  Driver Image Analysis – Alertness estimation from uploaded images
-  Voice Alerts – Spoken warnings for Distracted, Drowsy & Aggressive states
-  Trip History – Permanent session history saved locally
-  Downloadable Reports – Export driving history as CSV
-  AI Suggestions Engine – Personalized safety recommendations
-  Responsive Futuristic UI – Mobile-friendly cyber dashboard

## Tech Stack

**Frontend:** HTML5, CSS3, JavaScript  
**Backend:** Python, Flask  
**Machine Learning:** Scikit-learn (Logistic Regression), NumPy, Pandas, Pillow  

## Driver State Classification

| State | Risk Score | Description |
|  Safe | < 30% | Stable and attentive driving |
|  Distracted | 30–50% | Attention diverted from road |
|  Drowsy | 50–75% | Signs of fatigue detected |
|  Aggressive | > 75% | Dangerous driving pattern |

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.93 |
| Precision | 1.0 |
| Recall | 0.62 |
| F1 Score | 0.76 |

## Project Architecture
Multimodal Input (Sensor + Image + Text)
↓
Data Preprocessing & Feature Extraction
↓
Logistic Regression Model (Single Algorithm)
↓
Adaptive Risk Score (0–1 scale)
↓
Driver State Classification
↓
Flask Web App Dashboard

## Applications

- Driver fatigue and distraction monitoring
- Fleet safety analytics
- Smart transportation systems
- Insurance risk profiling
- Road accident prevention

## Future Scope

- Live webcam-based drowsiness detection
- Emergency SOS alert system
- Cloud database integration
- Android / iOS mobile application
- Real-time IoT sensor integration
- Accident-prone route intelligence

## Live Demo

🔗 https://ai-driver-risk-monitor.onrender.com
