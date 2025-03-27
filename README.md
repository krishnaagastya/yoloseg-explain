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
git clone https://github.com/yourusername/yoloseg-explain.git
cd yoloseg-explain
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
python src/yoloseg_visualizer.py --image path/to/image.jpg --target_class "refrigerator" --model yolov8s-seg.pt --mode explain
```

### Command Line Arguments
- `--image`: Path to input image (Required)
- `--target_class`: Specific class to visualize (Optional)
- `--model`: YOLO model path (Default: yolov8s-seg.pt)
- `--mode`: Visualization mode ('all' or 'explain', Default: 'all')

## Troubleshooting
- Ensure CUDA is properly installed for GPU acceleration
- Check that your YOLO model is compatible with the script. Only SEGMENTATION MODELS can be analyzed.
- Verify image path and class names

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
```

# Test Image
![Test Image](test_data/books-1.jpg "Test Image")
<img src="test_data/books-1.jpg" alt="Test Image" width="500"/>

# Results on Test image
![Results on Test Image](test_data/books-1.jpg "Results on Test Image")
