# Weapons and Knives Detector with YOLOv8

This project implements a **real-time object detection system** for identifying weapons and knives using the **YOLOv8** model. It includes scripts for detecting objects in images, videos, and webcam feeds, along with optional preprocessing capabilities.

---

## 📋 Prerequisites

* Python 3.8+
* A compatible webcam (for `webcam-detection.py`)
* Test videos or images for detection

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set Up a Virtual Environment *(optional but recommended)*

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Model Weights

Place your trained model weights (e.g., `best.pt`) into:

```
runs/detect/Normal_Compressed/weights/
```

## 🚀 Usage

### 1. Detect Objects in Images

```bash
python detecting-images.py
```

* Make sure to update the script with the path to your input image.
* Results will be saved in the `Results/` directory.

---

### 2. Detect Objects in Videos

```bash
python video-detection.py
```

* Update the script with the path to your input video (e.g., `test_vid/vid1.mp4`).
* The output video will be saved as `detected_downloaded_video.mp4`.

---

### 3. Real-time Webcam Detection

```bash
python webcam-detection.py
```

* Ensure your webcam is connected.
* Press `q` to exit the webcam feed.
* The output will be saved as `webcam_detected_objects.avi`.

---

### 4. Preprocess Images *(Optional)*

```bash
python preprocessing-images.py
```