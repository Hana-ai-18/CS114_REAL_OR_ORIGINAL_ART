# 🎨 AI Art Detector — ML Project
## Phân biệt tranh thật vs tranh AI — Chỉ dùng thuật toán ML

---

## 📁 Cấu trúc project

```
ai_art_detector/
├── 01_data_preprocessing.py   # Xử lý dữ liệu, EDA, tạo train/val/test split
├── 02_feature_engineering.py  # Extract features theo 3 pipeline
├── 03_train_models.py         # Train 9 models × 3 pipelines, so sánh kết quả
├── 04_web_app.py              # Flask web app upload ảnh & predict
├── requirements.txt
└── README.md
```

---

## 🔧 Cài đặt

```bash
pip install -r requirements.txt
```

---

## 📂 Chuẩn bị dataset

Tải dataset từ Kaggle:
https://www.kaggle.com/datasets/ravidussilva/real-ai-art

Tổ chức thư mục như sau:
```
dataset/
├── real_art/      ← ảnh tranh thật
│   ├── img001.jpg
│   └── ...
└── ai_art/        ← ảnh tranh AI
    ├── img001.jpg
    └── ...
```

---

## 🚀 Chạy từng bước

### Bước 1 — Xử lý dữ liệu
```bash
python 01_data_preprocessing.py
```
Output:
- `processed/split_manifest.json` — danh sách train/val/test
- `plots/01_dataset_overview.png`
- `plots/01_pixel_distribution.png`
- `plots/01_split_distribution.png`

---

### Bước 2 — Feature Engineering
```bash
python 02_feature_engineering.py
```
Output:
- `processed/features/pipeline_A.pkl` — Fractal + Wavelet + GLCM
- `processed/features/pipeline_B.pkl` — FFT + Noise + LBP
- `processed/features/pipeline_C.pkl` — Full ensemble (A + B + HOG + Color + Moments)
- `plots/02_feature_extraction.png` — Visualize từng feature
- `plots/02_feature_variance_*.png` — Cohen's d effect size

---

### Bước 3 — Train & So sánh Models
```bash
python 03_train_models.py
```
Output:
- `models/` — 27 models (9 models × 3 pipelines)
- `models/best_model.pkl` — best model
- `processed/results.json` — kết quả đầy đủ
- `plots/03_model_comparison.png` — so sánh tổng thể
- `plots/03_confusion_matrices.png` — confusion matrix
- `plots/03_best_model_detail.png` — chi tiết best model
- `plots/03_pipeline_ranking.png` — so sánh pipeline

---

### Bước 4 — Web App
```bash
python 04_web_app.py
```
Mở browser: http://localhost:5000

---

## 🧠 3 Pipeline Feature Extraction

### Pipeline A — Hướng truyền thống (đề xuất ban đầu)
| Feature | Dims | Ý nghĩa |
|---------|------|---------|
| Fractal Dimension (Box-counting) | 3 | Độ phức tạp fractal |
| Wavelet DWT (db4, 3 levels) | ~120 | Phân tích đa phân giải |
| GLCM | 24 | Texture co-occurrence |
| **Total** | **~150** | |

### Pipeline B — Hướng noise/frequency ⚡ (THƯỜNG TỐT HƠN)
| Feature | Dims | Ý nghĩa |
|---------|------|---------|
| FFT Radial Spectrum | 25 | Power spectrum theo tần số |
| Noise Pattern Analysis | 31 | Residual noise + auto-correlation |
| LBP Histogram | 29 | Local texture patterns |
| **Total** | **~85** | |

> **LÝ DO Pipeline B tốt hơn:**
> - AI generators (Stable Diffusion, DALL-E, Midjourney) tạo FFT spectrum mượt bất thường → thiếu high-frequency noise ngẫu nhiên của tranh thật
> - Noise pattern của AI có tính regularity cao (lặp lại theo grid artifacts)
> - LBP bắt được micro-texture mà AI lặp lại do convolution trong decoder

### Pipeline C — Full Ensemble (BEST)
Pipeline A + B + HOG + Color Histogram + Statistical Moments
**~290 features**

---

## 🤖 9 Models được train

| Model | Cần Scale | Đặc điểm |
|-------|-----------|-----------|
| Random Forest | ❌ | Robust, tốt baseline |
| Extra Trees | ❌ | Nhanh hơn RF |
| Gradient Boosting | ❌ | Sequential, mạnh |
| AdaBoost | ❌ | Classic boosting |
| SVM RBF | ✅ | Rất mạnh sau scale |
| SVM Linear | ✅ | Nhanh, high-dim |
| Logistic Regression | ✅ | Interpretable |
| KNN | ✅ | Non-parametric |
| Naive Bayes | ✅ | Nhanh, baseline |

---

## 📊 Kết quả dự kiến

Dựa trên papers tương tự với dataset AI art:

| Pipeline | Model | Expected Accuracy | Expected AUC |
|----------|-------|------------------|--------------|
| C | SVM RBF | ~88-93% | ~0.93-0.97 |
| C | Random Forest | ~85-91% | ~0.91-0.95 |
| B | SVM RBF | ~84-90% | ~0.90-0.94 |
| A | Random Forest | ~79-85% | ~0.85-0.90 |

---

## 📐 Thuật toán sử dụng (pure ML, không có DL)

- **Fractal Dimension**: Box-counting algorithm
- **Wavelet Transform**: Discrete Wavelet Transform (PyWavelets)
- **GLCM**: Gray-Level Co-occurrence Matrix (scikit-image)
- **FFT**: Fast Fourier Transform + Radial Spectrum (numpy)
- **Noise Analysis**: Gaussian residual + Auto-correlation
- **LBP**: Local Binary Pattern (scikit-image)
- **HOG**: Histogram of Oriented Gradients (scikit-image)
- **Classifiers**: All from scikit-learn

---

## 🌐 Web App Features

- Upload ảnh bất kỳ (JPG/PNG/WEBP)
- Hiển thị confidence score
- Visualize: FFT spectrum, LBP histogram, noise map, wavelet energy
- Gauge chart confidence
- Responsive dark theme UI