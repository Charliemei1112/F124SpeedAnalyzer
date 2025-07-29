# F1 24 Speed Analyzer

> **Extracting Speed Data from F124 Game Recording by extracting frames and analyzing the speed digits through self-trained PyTorch CNN.**

---


### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd F124
   ```

2. **Create and activate virtual environment** *(optional but highly recommended)*
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Analyze the Video**
   ```bash
   python main.py <video_path>
   ```
   
   **Example:**
   ```bash
   python main.py RedBull.mp4
   ```

5. **Expected Output**
   ```
   [17:38:17] Starting analysis of video: RedBull.mp4
   [17:38:17] Starting video frame extraction and speed analysis...
   [17:38:17] Resolution: 1920x1200
   [17:38:17] Processing Frames::  97%|████████████████████████████████████▉ | 2686/2764 [00:09<00:00, 293.03it/s]
   [17:38:26] Time taken: 9.546 seconds
   [17:38:26] Successfully extracted 2686 speed data points
   [17:38:26] Calculating performance metrics...
   [17:38:27] ANALYSIS COMPLETE!
   [17:38:27] ==================================================
   [17:38:27] Video: RedBull.mp4
   [17:38:27] Frames processed: 2686
   [17:38:27] Average speed: 215.2 km/h
   [17:38:27] Max speed: 329 km/h
   [17:38:27] Min speed: 69 km/h
   ```

---

##  Output Files

After successful analysis, the following files will be generated:

| File | Description |
|------|-------------|
| `speed.json` | Raw speed data points |
| `speed.png` | Speed visualization chart |
| `speed.dat` | Speed data in text format |
| `metrics.json` | Statistical analysis |
| `results.json` | Comprehensive summary file |

---

## 🔄 Workflow

### 1. Frame Extraction
The video input will be processed into frames via OpenCV Library.

### 2. Speed Region Extraction
For each frame, the speedmometer region will be cropped as the input for model prediction

### 3. Digit Preprocessing
The speedmeter number will be divided into three digits, with each digits prepocessed with grayscale and thresholding techniques so that the the input image will be 26*40 binary format (grayscale 255 or grayscale 0).

### 4. Digit Prediction through Trained Model
The input will be transformed into 26*40 2D array for model prediction, the output will be digit number 0-9, and an extra digit called *, which is reserved as blank space for two-digit and one-digit number.

### 5. Speed Data Generation
With the prediction for each digit, we group three digits back together into speed data and append all of then into array for perfromance analysis.

---

## Prediction Model

The training model is generated with 8000 digit images as data input, with 80%-10%-10% training, validating, testing ratio. The model is trained with simple Pytorch CNN and the accuracy is 99.9%.

---

## ⚡ Performance

- **Accuracy:** >= 99.9%
- **Processing Speed:** 300 frames per second

### Performance Benchmarks (1080p 30fps)

| Video Length | Data Points | Process Time |
|--------------|-------------|-------------|
| 90 seconds (Qualifying) | ~3,000 | ~10 seconds |
| 30 minutes (Sprint) | ~160,000 | 3~4 minutes |
| 90 minutes (Race) | ~500,000 | ~10 minutes |

*Test performed on Macbook M4 Pro (24GB RAM)*

## Limitations & Improvement
At this point, there is still limitation of the program. I will post potential solution for the improvement and keep them updated as I solved them.

#### **1. Video Quality Limitation**
- **Issue**: The Programm only support 1920*1200 resolution. The input digit image is not 26*40 pixels at different resolutions.
- **Solution**: Resize input images into 26*40 pixels regardless of input size.

#### **2. Speed Region Configuration**
- **Issue**: The speed meter value does not necessary locate at the same (default) region on the screen. The resolution and user settings can change where the speed data locate on the screen.
- **Solution**: Add constant values for different resolutions (1080p, 1440p, 4k), create simple user interface to ask user input for speed region.

#### **3. Processing Speed**
- **Issue**: The program can only process 300 frames per second, ideally I want 3000 frames per second so that I can analyze the entire race in one minute.
- **Solution**: Optimize memory usage (currently 500MB) and simply code to reduce logic operations. Maybe consider parallel processing when iterating the frames

#### **4. Training Model**
- **Issue**: The current model is too complicated. There are 3 convolution layers trained with 50 epoches. The accuracy reaches 99.9% under 5 epoches.
- **Solution**: Consider simplyfying the CNN model while maintain the accuracy, thus reducing the parameters and increase processing speed.
