# 🚀 Quick Start Guide

Get the Driver Drowsiness Detection app running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- ~500 MB disk space

## Installation (Windows, Mac, Linux)

### 1. Extract Project

```bash
cd drowsiness_ui
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installation time:** 2-3 minutes (first time)

### 4. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## First Steps

### ✅ Home Page
1. Upload an image or select a sample
2. See real predictions and confidence
3. View fatigue assessment

### 📊 Explore Other Pages
- **How It Works:** Understand the pipeline
- **Dataset & EDA:** See data statistics
- **Model Comparison:** Custom CNN vs MobileNetV2
- **Performance:** See model evaluation
- **Fatigue Analysis:** Understand fatigue mapping
- **About:** Project information

---

## Sample Images

Try these pre-loaded samples:

| Sample | Filename | Expected Class |
|--------|----------|-----------------|
| Closed Eyes 1 | _0.jpg | Closed |
| Closed Eyes 2 | _1.jpg | Closed |
| Open Eyes 1 | 1.jpg | Open |
| Open Eyes 2 | 3.jpg | Open |
| Yawning 1 | _8.jpg | yawn |
| Yawning 2 | _9.jpg | yawn |

---

## Model Info

- **Accuracy:** 97.70%
- **Model:** MobileNetV2 (Transfer Learning)
- **Input:** 224×224 RGB image
- **Output:** 4-class prediction + Fatigue level
- **Size:** 18 MB
- **Training Data:** 2,900 images, 4 classes

---

## Stop the App

Press `Ctrl+C` in terminal or close browser

---

## Troubleshooting

### Port 8501 Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Module Not Found Error
```bash
pip install -r requirements.txt --force-reinstall
```

### Model File Missing
Check that `mobilenetv2_finetuned.keras` exists in the project folder:
```bash
ls -lh mobilenetv2_finetuned.keras
```

---

## Next Steps

📖 Read the full [README.md](README.md) for:
- Detailed installation instructions
- Complete feature documentation
- API reference
- Advanced configuration

---

## Need Help?

1. Check [README.md](README.md) troubleshooting section
2. Run test suite: `python3 test_app.py`
3. Check Streamlit logs in terminal
4. Ensure all files are present: `ls -la`

---

**Status:** ✅ Ready to use!

Enjoy the application! 🎓🚗
