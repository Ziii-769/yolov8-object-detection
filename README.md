# YOLOv8 Object Detection System

## 1. Project overview
This beginner-friendly AI engineering project uses the Ultralytics YOLOv8 model to detect common objects in images, videos, or a webcam stream. It uses pretrained model weights, so model training is not required.

## 2. Objectives
- Set up a Python computer-vision project.
- Load a pretrained YOLOv8 model.
- Detect and label objects in an input source.
- Display confidence scores in the terminal.
- Save annotated results to an output folder.

## 3. Project structure
```text
yolov8_object_detection_project/
├── input/
│   └── (put your test image here)
├── output/
│   └── (detection results are saved here)
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Requirements
- Python 3.10 or 3.11 is a good beginner-friendly choice; check the current Ultralytics installation requirements if using another version.
- Internet access the first time dependencies and model weights are downloaded.
- VS Code or another Python editor.
- A test image, video, or webcam.

## 5. Installation on Windows (PowerShell)
Open this project folder in VS Code, then open the terminal in that folder.

Create a virtual environment:
```powershell
py -m venv .venv
```

Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install the dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. How to run

### Detect objects in an image
Place an image in the `input` folder and name it `test.jpg`, then run:
```powershell
python main.py
```

Or provide an image path:
```powershell
python main.py --source "C:\path\to\your\image.jpg"
```

### Detect objects in a video
```powershell
python main.py --source "C:\path\to\your\video.mp4"
```

### Use a webcam
```powershell
python main.py --source 0
```
Press `Ctrl+C` in the terminal to stop webcam processing if needed.

### Change the confidence threshold
```powershell
python main.py --source "input/test.jpg" --confidence 0.50
```
A higher confidence threshold generally filters out less-certain detections.

## 7. How the program works
1. `argparse` reads the source path and settings from the terminal.
2. `YOLO("yolov8n.pt")` loads the pretrained YOLOv8 nano model. The weights may download on first use.
3. `model.predict()` runs object detection.
4. The program prints each detected object's label and confidence score.
5. YOLO saves an annotated image or video under `output/results/`.

## 8. Team roles (group of three)
- **Member 1 — Project setup and documentation:** create the folders, install Python packages, and maintain this README.
- **Member 2 — Model and detection code:** explain the YOLO model, review `main.py`, and test object detection.
- **Member 3 — Testing and presentation:** collect test images, run experiments, record results, and prepare the demonstration.

All three members should understand the whole project and be able to explain the code.

## 9. Testing checklist
- [ ] Dependencies install without errors.
- [ ] The model loads successfully.
- [ ] An image with common objects is processed.
- [ ] Detected labels and confidence scores appear in the terminal.
- [ ] An annotated result is saved under `output/results/`.
- [ ] The team records which objects were detected and any mistakes.

## 10. Limitations and future improvements
The pretrained model can only recognize classes it was trained on. Detection quality depends on lighting, image quality, object size, and the model's training data. Future improvements could include a simple user interface, counting objects, comparing confidence thresholds, or training a custom model on a carefully labelled dataset.

## 11. Conclusion
This project demonstrates a basic AI engineering workflow: setting up a Python environment, using a pretrained computer-vision model, processing input data, reporting predictions, and saving results.
