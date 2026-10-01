# VisionCount AI

An AI-powered object detection and counting system that uses **YOLO** and **computer vision** to detect objects in images and videos. The system identifies objects, draws bounding boxes, counts detected objects, and displays the results through an interactive **Streamlit** web interface.

## Features

* 🔍 AI-based object detection using YOLO
* 🖼️ Image object detection
* 🎥 Video object detection
* 📦 Bounding boxes and object labels
* 🔢 Automatic object counting
* 📊 Detection results through a web interface
* ⚡ Real-time processing of uploaded images and videos
* 🌐 Simple and interactive Streamlit interface

## Technologies Used

* **Python**
* **YOLO (Ultralytics)**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **Pillow**

## Project Structure

```text
VisionCount-AI/
│
├── app.py
├── detection.py
├── requirements.txt
├── .gitignore
├── pic.jpg
├── results/
└── README.md
```

## How It Works

The system follows these steps:

1. User uploads an image or video.
2. The YOLO model analyzes the input.
3. Objects are detected using deep learning.
4. Bounding boxes and object labels are generated.
5. Detected objects are counted.
6. The results are displayed through the Streamlit interface.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/VisionCount-AI.git
```

### 2. Open the Project

```bash
cd VisionCount-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Output

The system displays:

* Detected object names
* Number of detected objects
* Bounding boxes
* Processed image/video results

## AI Model

The project uses a pretrained **YOLO** model from Ultralytics for object detection. The model can recognize multiple common object categories without requiring the model to be trained from scratch.

## Future Improvements

* Object tracking with unique IDs
* Real-time webcam detection
* Improved object counting across video frames
* Detection confidence display
* Analytics and visualization dashboard
* Detection history and reports

## Author

**Hira Aslam**

BS Information Technology
University of Education, Lahore

## License

This project is developed for educational and internship purposes.
