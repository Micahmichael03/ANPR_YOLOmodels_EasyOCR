# ANPR_YOLOmodels_EasyOCR
The ANPR_YOLOv8_EasyOCR project is an Automated Number Plate Recognition (ANPR) system built using YOLOv8 for object detection and EasyOCR for license plate text extraction. Here's a detailed breakdown of its components:

```markdown
# ANPR_YOLOv8_EasyOCR

This repository contains a project for Automatic Number Plate Recognition (ANPR) using YOLOv8 and EasyOCR. The system detects vehicles and their license plates in video footage, processes the license plates, and records the results in a CSV file.

## Project Structure

```
ANPR_YOLOv8_EasyOCR/
├── add_missing_data.py
├── main.py
├── util.py
├── visualize.py
├── csv_file_record/
├── models/
│   ├── yolov8n.pt
│   ├── license_plate_detector.pt
├── sort/
├── videos/
│   ├── tc1.mp4
├── results/
│   ├── results.csv
├── requirements.txt
└── README.md
```

### Files and Directories

- `add_missing_data.py`: Script to add missing data (not detailed here).
- `main.py`: Main script for running the ANPR system.
- `util.py`: Utility functions for reading and writing data, processing license plates, etc.
- `visualize.py`: Script for visualizing the results.
- `csv_file_record/`: Directory to store CSV files.
- `models/`: Directory containing the YOLO models.
- `sort/`: Directory containing the SORT tracker implementation.
- `videos/`: Directory containing video files for testing.
- `results/`: Directory to store the results.

## Requirements

The required packages are listed in `requirements.txt`.

```
ultralytics==8.0.114
pandas==2.0.2
opencv-python==4.7.0.72
numpy==1.24.3
scipy==1.10.1
easyocr==1.7.0
filterpy==1.4.5
```

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Running the main script

To run the ANPR system, use the `main.py` script. Ensure that you have the YOLO models and the video file in the correct directories.

```bash
python main.py
```

For example:

```python
cap = cv2.VideoCapture(r'./videos/your_video.mp4')
```

### Visualizing Results

To visualize the results, use the `visualize.py` script. This will generate an output video with the detected vehicles and license plates.

```bash
python visualize.py
```

## Example Video

You can use the following video link for testing: [Pexels Traffic Flow Video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)

## Repository for SORT

The SORT tracker implementation can be found here: [SORT GitHub Repository](https://github.com/abewley/sort)

## Acknowledgements

This project uses YOLOv8 from Ultralytics and EasyOCR for Optical Character Recognition.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
