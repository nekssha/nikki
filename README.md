# Driver Drowsiness Detection System

## 📌 Project Overview

The **Driver Drowsiness Detection System** is a computer vision–based project that detects whether a driver is sleepy by monitoring eye movements in real time.
If the system detects that the driver's eyes are closed for a certain period, it triggers an alarm sound to alert the driver and prevent possible accidents.

This project uses a trained deep learning model to classify eye states (open or closed) and uses a webcam for real-time detection.

---

## 🚀 Features

* Real-time face and eye detection using OpenCV
* Eye state classification (Open / Closed) using a CNN model
* Drowsiness score calculation
* Alarm alert when driver appears sleepy
* Saves a captured image when drowsiness is detected
* 20-second alarm cooldown to avoid continuous sound

---

## 🛠 Technologies Used

* Python
* OpenCV
* NumPy
* TensorFlow / Keras
* Pygame (for alarm sound)

---

## 📂 Project Structure

DAS/

│── haar cascade files/

│   ├── haarcascade_frontalface_alt.xml

│   ├── haarcascade_lefteye_2splits.xml

│   └── haarcascade_righteye_2splits.xml

│

│── models/

│   └── cnncat2.h5

│

│── alarm.wav

│── main.py

│── image.jpg

│── README.md

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

pip install opencv-python
pip install tensorflow
pip install keras
pip install pygame
pip install numpy

### 2️⃣ Run the Program

python main.py

### 3️⃣ Exit Program

Press **Q** to close the camera window.

---

## ⚙️ How It Works

1. The webcam captures the driver’s face.
2. Haar Cascade classifiers detect the face and eyes.
3. The CNN model predicts whether the eyes are open or closed.
4. A score increases when eyes remain closed.
5. If the score exceeds a threshold, an alarm sound is triggered.

---

## 🎯 Applications

* Driver safety systems
* Accident prevention
* Smart vehicle monitoring
* AI-based driver assistance systems

---

## 👩‍💻 Author

Nikshitha Reddy
