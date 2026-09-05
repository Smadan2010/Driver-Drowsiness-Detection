# Driver Drowsiness Detection - Interactive UI

An academic deep learning application for detecting driver drowsiness using eye closure and yawning analysis.

## 🎯 Project Overview

**Title:** Driver Drowsiness Detection using Eye Closure and Yawning Analysis with Deep Learning

**Type:** Academic Mini Project

**Domain:** Computer Vision & Deep Learning

**Purpose:** Demonstrate a vision-based, non-intrusive system for detecting early signs of driver fatigue using physiological facial indicators.

---

## 📁 Project Structure

```
drowsiness_ui/
├── app.py                          # Main Streamlit application
├── preprocessing.py                # Image preprocessing utilities
├── prediction.py                   # Model loading and prediction
├── mappings.py                     # Class/fatigue level mappings
├── test_app.py                     # Comprehensive test suite
├── requirements.txt                # Python dependencies
├── mobilenetv2_finetuned.keras    # Trained MobileNetV2 model (18 MB)
├── _0.jpg, _1.jpg, _8.jpg, _9.jpg # Sample images (Closed/Yawn)
├── 1.jpg, 3.jpg, 8.jpg, 24.jpg    # Sample images (Open)
└── README.md                       # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download Project

```bash
cd /home/claude/drowsiness_ui
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.15.0
- Keras 2.15.0
- Streamlit 1.32.2
- NumPy 1.24.3
- Pillow 10.1.0
- OpenCV 4.8.1.78
- Plotly 5.18.0
- Pandas 2.1.3
- scikit-learn 1.3.2

### Step 4: Verify Installation

Run the test suite to verify everything is working:

```bash
python3 test_app.py
```

Expected output:
```
✅ ALL TESTS PASSED SUCCESSFULLY!
```

---

## ▶️ Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Command Line Options

```bash
# Run with specific port
streamlit run app.py --server.port 8502

# Run in headless mode (no browser)
streamlit run app.py --server.headless true

# Set log level
streamlit run app.py --logger.level=debug
```

---

## 📱 Application Features

### 🏠 Home Page - Drowsiness Check
- **Upload custom images** for real-time drowsiness detection
- **Try sample images** from the dataset
- **View predictions** with confidence scores
- **See fatigue assessment** (Alert, Mild Fatigue, Severe Fatigue)
- **Real model predictions** using trained MobileNetV2

### 🔍 How It Works
- Visual explanation of the processing pipeline
- Step-by-step breakdown of each stage
- Detailed explanation of model predictions and fatigue mapping

### 📊 Dataset & EDA
- Dataset composition (2,900 images, 4 classes)
- Train/validation/test split visualization
- Class distribution analysis
- Preprocessing information
- Data augmentation details

### ⚖️ Model Comparison
- Custom CNN vs MobileNetV2
- Performance metrics comparison
- Architecture details
- Why MobileNetV2 was selected

### 📈 Model Performance
- Test set accuracy (97.70%)
- Per-class performance metrics
- Confusion matrix heatmap
- Precision, recall, F1-score

### 😴 Fatigue Analysis
- 4-class to 3-level fatigue mapping explanation
- Simulated fatigue progression during driving
- Interpretation of fatigue levels
- Transition points and warnings

### ℹ️ About
- Project overview and problem statement
- Dataset and technology details
- Model results and selection rationale
- Limitations and disclaimer

---

## 🤖 Model Information

### Selected Model: MobileNetV2 (Transfer Learning)

**Architecture:**
- Pre-trained on ImageNet (1.4M images, 1000 classes)
- Frozen base layers
- Fine-tuned classification head for 4-class drowsiness detection
- Total parameters: 2.3M

**Performance:**
- Test Accuracy: **97.70%**
- Precision: 97.83%
- Recall: 97.70%
- F1-Score: 97.70%
- Training Time: 23.83 minutes
- Model Size: 8 MB

**Per-Class Performance:**
- Closed: 100.0% (109/109)
- Open: 100.0% (109/109)
- no_yawn: 95.54% F1 (99.07% recall, 92.24% precision)
- yawn: 95.24% F1 (91.74% recall, 99.01% precision)

### Comparison: Custom CNN

- Accuracy: 68.28%
- Parameters: 3.2M
- Size: 120 MB
- Training Time: 19.45 minutes
- Status: **Not selected** (lower accuracy and larger footprint)

---

## 🎓 Class Information

### 4-Class Model Output

| Index | Class | Meaning |
|-------|-------|---------|
| 0 | Closed | Eyes are closed (drowsy state) |
| 1 | Open | Eyes are open (alert state) |
| 2 | no_yawn | No yawning detected (normal state) |
| 3 | yawn | Yawning detected (early fatigue indicator) |

### Fatigue Level Mapping

| Model Prediction | Fatigue Level | Emoji | Action |
|------------------|---------------|-------|--------|
| Closed | **Severe Fatigue** (2) | 🛑 | Immediate rest needed |
| Open | **Alert** (0) | ✅ | Driver is focused |
| no_yawn | **Alert** (0) | ✅ | Normal state |
| yawn | **Mild Fatigue** (1) | ⚠️ | Consider break soon |

---

## 📊 Dataset Details

### Overview
- **Total Images:** 2,900
- **Classes:** 4 (Closed, Open, no_yawn, yawn)
- **Image Size:** 224×224 RGB
- **Balance:** ~725 images per class (25% each)

### Train/Validation/Test Split
- **Training:** 2,030 images (70%)
- **Validation:** 435 images (15%)
- **Test:** 435 images (15%)

### Preprocessing
1. **Resizing:** All images → 224×224
2. **Normalization:** Pixel values / 255.0 → [0, 1]
3. **Data Type:** float32

### Data Augmentation (Training Only)
- Rotation: ±20°
- Zoom: 0.8× to 1.2×
- Brightness: 0.8 to 1.2
- Horizontal Flip: Yes
- Fill Mode: Nearest

---

## 🧪 Testing

### Run the Test Suite

```bash
python3 test_app.py
```

Tests performed:
1. ✅ Required files check
2. ✅ Module imports (TensorFlow, OpenCV, custom modules)
3. ✅ Model loading verification
4. ✅ Preprocessing pipeline validation
5. ✅ Model prediction testing
6. ✅ Fatigue mapping verification
7. ✅ Prediction module integration
8. ✅ Sample images processing
9. ✅ Project metrics verification

---

## 📝 Usage Examples

### Example 1: Upload and Predict

1. Open the app → Go to **Home** page
2. Click "📤 Upload Your Image"
3. Select an image file (JPG/PNG)
4. View predictions and fatigue assessment

### Example 2: Try Sample Images

1. Open the app → Go to **Home** page
2. Select "📸 Try a Sample Image"
3. Choose from provided samples
4. See real model predictions

### Example 3: Explore Dataset

1. Go to **Dataset & EDA** page
2. View class distribution
3. Check train/validation/test split
4. Review preprocessing steps

### Example 4: Compare Models

1. Go to **Model Comparison** page
2. View Custom CNN vs MobileNetV2
3. See accuracy comparison chart
4. Understand why MobileNetV2 was selected

### Example 5: Analyze Fatigue Progression

1. Go to **Fatigue Analysis** page
2. View fatigue level mapping
3. See simulated progression over time
4. Understand fatigue transitions

---

## ⚠️ Limitations & Disclaimer

### This is an ACADEMIC DEMONSTRATION

**NOT suitable for:**
- Medical diagnosis systems
- Safety-critical real-time systems
- Production deployment without extensive testing
- Real-time driving monitoring

**Limitations:**
- Image-based only (no continuous video)
- Performance depends on image quality
- Not validated on diverse lighting conditions
- Predictions are probabilistic (not 100% reliable)
- Requires clear, frontal facial images
- No personal data storage or privacy controls

### Academic Use Only
- Educational demonstration of deep learning
- Research and development
- Learning tool for students and mentors
- Prototype for potential real-world solutions

---

## 🔧 Troubleshooting

### Issue: Model file not found

**Solution:**
```bash
# Ensure mobilenetv2_finetuned.keras exists
ls -lh mobilenetv2_finetuned.keras

# If missing, check the uploads folder
ls -lh /mnt/user-data/uploads/mobilenetv2_finetuned.keras
```

### Issue: Module import errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific packages
pip install tensorflow keras numpy pillow opencv-python streamlit
```

### Issue: Port already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using the port
lsof -i :8501  # Find process
kill -9 <PID>  # Kill process
```

### Issue: Images not loading

**Solution:**
```bash
# Ensure image files are in the correct directory
ls -lh *.jpg

# Check file permissions
chmod 644 *.jpg
```

### Issue: Out of memory

**Solution:**
```bash
# Streamlit caches models by default
# Clear cache if needed
rm -rf ~/.streamlit/

# Run with reduced resources
streamlit run app.py --client.maxMessageSize=10
```

---

## 📚 File Descriptions

### `app.py` (31 KB)
Main Streamlit application with 7 pages:
- Home: Interactive prediction interface
- How It Works: Pipeline visualization
- Dataset & EDA: Data exploration
- Model Comparison: CNN vs MobileNetV2
- Model Performance: Evaluation metrics
- Fatigue Analysis: Fatigue progression
- About: Project information

### `preprocessing.py` (3.6 KB)
Image preprocessing utilities:
- `resize_image()`: Resize to 224×224
- `normalize_image()`: Normalize to [0, 1]
- `preprocess_single_image()`: Complete pipeline for model
- `preprocess_for_display()`: Preprocessing for visualization

### `prediction.py` (3.9 KB)
Model and prediction management:
- `load_model()`: Load trained MobileNetV2 (cached)
- `predict_drowsiness()`: Single image prediction
- `predict_batch()`: Batch prediction

### `mappings.py` (4.4 KB)
Class and fatigue mappings:
- `CLASS_NAMES`: ['Closed', 'Open', 'no_yawn', 'yawn']
- `get_fatigue_level()`: Map prediction to fatigue
- `PROJECT_METRICS`: Actual notebook results
- `DATASET_INFO`: Dataset composition
- `MOBILENETV2_PER_CLASS_METRICS`: Per-class performance

### `test_app.py` (9 KB)
Comprehensive test suite:
- Tests 9 different components
- Verifies model loading
- Validates preprocessing
- Tests predictions
- Checks all sample images

### `mobilenetv2_finetuned.keras` (18 MB)
Trained MobileNetV2 model:
- Input: (224, 224, 3) RGB image
- Output: (4,) class probabilities
- Pre-trained on ImageNet + fine-tuned
- 97.70% test accuracy

### `requirements.txt`
Python dependencies for the application

---

## 🎨 UI Design

### Color Scheme
- **Primary:** #1f77b4 (Blue)
- **Alert/Safe:** #51CF66 (Green)
- **Mild Fatigue:** #FFD43B (Yellow)
- **Severe Fatigue:** #FF6B6B (Red)

### Typography
- Font Family: Segoe UI, Helvetica Neue
- Headers: 2.5em, bold
- Subtitle: 1.2em

### Layout
- Sidebar navigation (7 pages)
- Responsive columns
- Professional styling
- Clear visual hierarchy

---

## 📊 Sample Predictions

### Sample 1: Closed Eyes
- File: `_0.jpg`
- Expected Class: Closed
- Fatigue Level: Severe Fatigue (🛑)
- Confidence: ~99%

### Sample 2: Open Eyes
- File: `1.jpg`
- Expected Class: Open
- Fatigue Level: Alert (✅)
- Confidence: ~99%

### Sample 3: Yawning
- File: `_8.jpg`
- Expected Class: yawn
- Fatigue Level: Mild Fatigue (⚠️)
- Confidence: ~90%

---

## 🔗 Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.15.0 | Deep learning framework |
| Keras | 2.15.0 | Neural network API |
| Streamlit | 1.32.2 | Web application framework |
| NumPy | 1.24.3 | Numerical computing |
| Pillow | 10.1.0 | Image processing |
| OpenCV | 4.8.1.78 | Computer vision |
| Plotly | 5.18.0 | Interactive visualizations |
| Pandas | 2.1.3 | Data manipulation |
| scikit-learn | 1.3.2 | ML utilities |

---

## 📄 License & Attribution

**Project:** Driver Drowsiness Detection using Eye Closure and Yawning Analysis

**Institution:** GUVI HCL Academy

**Project Type:** Academic Mini Project

**Created:** 2024

This is an educational demonstration created for learning purposes.

---

## 👨‍💼 Project Information

**Project Creator:** [Student Name]

**Verified By:** [Mentor Name]

**Approved By:** [Instructor Name]

---

## ✅ Verification Checklist

Before deploying the app, ensure:

- [ ] `mobilenetv2_finetuned.keras` model file exists (18 MB)
- [ ] All sample images exist (_0.jpg, _1.jpg, etc.)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test suite passes (`python3 test_app.py`)
- [ ] App launches without errors (`streamlit run app.py`)
- [ ] Model loads successfully
- [ ] Predictions work on sample images
- [ ] All 7 pages render correctly
- [ ] Image upload works
- [ ] Confidence scores display correctly
- [ ] Fatigue mapping is accurate

---

## 🤝 Support & Issues

### Common Questions

**Q: Can I use this for real-time driving?**
A: No, this is an academic demonstration only. It uses static images, not video streams.

**Q: How accurate is the model?**
A: 97.70% test accuracy, but performance depends on image quality and similarity to training data.

**Q: Can I modify the model?**
A: Yes, but the original trained model is the source of truth for this demo.

**Q: How do I add my own images?**
A: Upload images via the "Upload Your Image" section on the Home page.

**Q: Is it real-time?**
A: No, predictions are per-image, not continuous video.

---

## 📞 Contact & Feedback

For issues, questions, or feedback:
1. Check the troubleshooting section
2. Review the test suite output
3. Check Streamlit logs
4. Consult the application's About page

---

## 🙏 Acknowledgments

- **Project Mentors:** GUVI HCL Academy
- **Dataset:** Facial image dataset for drowsiness detection
- **Framework:** TensorFlow, Keras, Streamlit
- **Pre-trained Model:** MobileNetV2 on ImageNet

---

**Last Updated:** September 2024

**Status:** ✅ Production Ready (Academic Demonstration)
