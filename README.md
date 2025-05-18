# 🧠 Labeled Object Detection System with Raspberry Pi

A real-time object detection system using the Raspberry Pi camera module and TensorFlow Lite, capable of identifying and labeling over 2000+ objects.

---

## 📌 Overview

This project leverages a lightweight object detection model (**EfficientDet Lite0**) running on **Raspberry Pi** with **PiCamera2**, using **TensorFlow Lite**. The system captures live video, detects objects, and overlays bounding boxes and labels on detected items in real time.

---

## 🚀 Features

- Uses the **Raspberry Pi Camera Module** for video input  
- Runs **EfficientDet Lite0** model via **TFLite Runtime**  
- Displays object labels with confidence scores  
- Optimized for real-time performance on edge devices  
- Includes **output demo videos** in the repository  

---

## 🧠 Technologies Used

- Raspberry Pi (with **PiCamera2** module)  
- **OpenCV** for image processing  
- **TensorFlow Lite Runtime** for inference  
- Python (NumPy, cv2)  

---

## 📁 Repository Contents

- `main.py` – Full object detection script  
- `output_video.mp4` – Sample output from real-time detection  
- `README.md` – Project documentation  

---

## ▶️ How to Run

1. Connect your Raspberry Pi Camera Module.
2. Clone this repository.
3. Ensure you have the following Python packages:
   ```bash
   pip install opencv-python numpy tflite-runtime
