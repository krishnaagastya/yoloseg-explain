# YOLO Segmentation Visualization Tool
This will work only with Yolov8-Segmentation models currently.

## Project Structure
```
yolosegm-explain/
│
├── src/
│   └── yoloseq_visualizer.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── test_data/
    └── books-1.jpg
```

## README.md
```markdown
# YOLO Segmentation Visualization Tool

## Overview
This tool provides advanced visualization and explanation capabilities for YOLO segmentation models.

## Prerequisites
- Python 3.8+
- CUDA (optional, but recommended for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolosegm-explain.git
cd yolosegm-explain
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO Model:
Model will be downloaded at run-time as needed.

## Usage

### Basic Visualization
```bash
python src/yolo_seg_working.py --image path/to/image.jpg --target_class "refrigerator" --model yolov8s-seg.pt --mode explain
```

### Command Line Arguments
- `--image`: Path to input image (Required)
- `--target_class`: Specific class to visualize (Optional)
- `--model`: YOLO model path (Default: yolov8s-seg.pt)
- `--mode`: Visualization mode ('all' or 'explain', Default: 'all')

## Troubleshooting
- Ensure CUDA is properly installed for GPU acceleration
- Check that your YOLO model is compatible with the script. Only SEGMENTATION MODELS will work now.
- Verify image path and class names

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
```

## requirements.txt
```
ultralytics==8.0.210
matplotlib>=3.7.1
torch>=1.13.1
torchvision>=0.14.1
numpy>=1.24.3
opencv-python>=4.7.0.72
argparse>=1.4.0
```
