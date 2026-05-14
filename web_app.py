"""
04_web_app.py
=============
Flask web app cho AI Art Detector.
Upload ảnh → extract features → predict với best model
Hiển thị: kết quả, confidence, feature visualization, explanation
"""

import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import cv2
import joblib
import pywt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import stats

from flask import Flask, request, jsonify, render_template_string
import traceback

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = Path("models/best_model.pkl")
META_PATH  = Path("models/best_model_meta.json")
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print("[INIT] Loading model...")
try:
    MODEL = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        META = json.load(f)
    print(f"[INIT] Loaded: {META['model_name']} ({META['pipeline']})")
    print(f"[INIT] AUC={META['auc']:.4f} | Acc={META['accuracy']:.4f}")
except Exception as e:
    print(f"[WARN] Cannot load trained model: {e}")
    MODEL = None
    META  = {"model_name": "Not trained", "pipeline": "N/A", "accuracy": 0, "auc": 0}


# ─────────────────────────────────────────
# FEATURE EXTRACTION (mirror từ 02_feature_engineering.py)
# ─────────────────────────────────────────

def fractal_dimension(gray, threshold=128):
    binary = (gray > threshold).astype(np.uint8)
    scales, counts = [], []
    max_box = min(gray.shape) // 2
    for box_size in range(2, max_box, max(1, max_box // 20)):
        h, w = binary.shape
        h_t = (h // box_size) * box_size
        w_t = (w // box_size) * box_size
        cropped = binary[:h_t, :w_t]
        reshaped = cropped.reshape(h_t//box_size, box_size, w_t//box_size, box_size)
        count = np.count_nonzero(reshaped.sum(axis=(1,3)))
        if count > 0:
            scales.append(box_size); counts.append(count)
    if len(scales) < 2: return 0.0
    slope, *_ = np.polyfit(np.log(1/np.array(scales,float)), np.log(np.array(counts,float)), 1)
    return float(slope)


def extract_all_features(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Extract full Pipeline C features"""
    feats = []

    # Fractal
    h, w = gray.shape
    feats.extend([fractal_dimension(gray), fractal_dimension(gray[:, :w//2]),
                  fractal_dimension(gray[:, w//2:])])

    # Wavelet
    img_f = gray.astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(img_f, 'db4', level=3)
    for lc in coeffs:
        for band in (lc if isinstance(lc, tuple) else (lc,)):
            bf = band.flatten()
            hist, _ = np.histogram(bf, bins=32, density=True)
            hp = hist[hist>0]
            feats.extend([np.sum(bf**2), np.mean(np.abs(bf)), np.std(bf),
                          -np.sum(hp*np.log2(hp+1e-12)), float(stats.kurtosis(bf))])

    # GLCM
    gq = np.clip(gray // 16, 0, 15).astype(np.uint8)
    glcm = graycomatrix(gq, [1,3,5], [0,np.pi/4,np.pi/2,3*np.pi/4], levels=16, symmetric=True, normed=True)
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        v = graycoprops(glcm, prop).flatten()
        feats.extend([v.mean(), v.std(), v.max(), v.min()])

    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(np.int32)
    max_r = min(cy, cx)
    n_bins = 20
    edges = np.linspace(0, max_r, n_bins+1).astype(int)
    rp = np.zeros(n_bins, np.float32)
    for i in range(n_bins):
        mask = (R >= edges[i]) & (R < edges[i+1])
        if mask.sum() > 0: rp[i] = mag[mask].mean()
    rp /= (rp.sum()+1e-8)
    slope, intercept = np.polyfit(np.arange(n_bins), rp, 1)
    feats.extend(rp.tolist())
    feats.extend([slope, intercept,
                  rp[n_bins//2:].sum()/(rp[:n_bins//2].sum()+1e-8),
                  rp.std(), rp.max()])

    # Noise
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5,5), 1.5)
    noise = gray.astype(np.float32) - blurred
    feats.extend([np.mean(np.abs(noise)), np.std(noise),
                  float(stats.kurtosis(noise.flatten())), float(stats.skew(noise.flatten()))])
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats.extend([np.mean(np.abs(lap)), np.std(lap)])
    nn = noise / (noise.std()+1e-8)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(nn))**2).real
    ac /= (ac.max()+1e-8)
    feats.extend(ac[h//2-2:h//2+3, w//2-2:w//2+3].flatten().tolist())

    # LBP
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    feats.extend(hist_lbp.tolist())
    hp = hist_lbp[hist_lbp>0]
    feats.extend([-np.sum(hp*np.log2(hp+1e-12)), np.std(hist_lbp), np.max(hist_lbp)])

    # HOG stats
    from skimage.feature import hog
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(16,16),
                    cells_per_block=(2,2), visualize=False, feature_vector=True)
    feats.extend([hog_feats.mean(), hog_feats.std(), hog_feats.max(), hog_feats.min(),
                  np.median(hog_feats), float(stats.skew(hog_feats)),
                  float(stats.kurtosis(hog_feats)),
                  np.percentile(hog_feats, 25), np.percentile(hog_feats, 75)])

    # Color
    for ch in range(3):
        cd = rgb[:,:,ch].flatten().astype(np.float32)
        hist_c, _ = np.histogram(cd, bins=32, range=(0,256), density=True)
        hp2 = hist_c[hist_c>0]
        feats.extend([cd.mean(), cd.std(), float(stats.skew(cd)),
                      float(stats.kurtosis(cd)), hist_c.max(),
                      -np.sum(hp2*np.log2(hp2+1e-12))])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for ch in range(3):
        cd = hsv[:,:,ch].flatten()
        feats.extend([cd.mean(), cd.std()])

    # Moments
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)
    edges_canny = cv2.Canny(gray, 50, 150)
    feats.extend(hu_log.tolist())
    feats.extend([float(edges_canny.mean()/255.0)])

    from scipy import ndimage
    ls = ndimage.generic_filter(gray.astype(float), np.std, size=9)
    feats.extend([float(ls.mean()), float(ls.max())])

    arr = np.array(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0, posinf=1e6, neginf=-1e6)


# ─────────────────────────────────────────
# VISUALIZATION HELPER
# ─────────────────────────────────────────

def make_analysis_figure(rgb: np.ndarray, gray: np.ndarray,
                          prediction: int, confidence: float,
                          features: np.ndarray) -> str:
    """Tạo figure phân tích, trả về base64 PNG"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0a0a15')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)
    TC = '#e0e0e0'
    RESULT_COLOR = '#ff6b6b' if prediction == 1 else '#00d4aa'
    label_text = "🤖 AI Generated" if prediction == 1 else "🎨 Real Art"

    fig.text(0.5, 0.97, f"Analysis Result: {label_text}  |  Confidence: {confidence:.1%}",
             ha='center', fontsize=14, color=RESULT_COLOR, fontweight='bold')

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Input Image", color=TC, fontsize=10)
    for s in ax.spines.values(): s.set_edgecolor(RESULT_COLOR); s.set_linewidth(2)

    # Grayscale
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(gray, cmap='gray'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Grayscale", color=TC, fontsize=10)
    for s in ax.spines.values(): s.set_edgecolor('#555'); s.set_linewidth(1)

    # FFT spectrum
    f = np.fft.fft2(gray.astype(float))
    fft_vis = np.log1p(np.abs(np.fft.fftshift(f)))
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(fft_vis, cmap='hot'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("FFT Spectrum", color=TC, fontsize=10)
    for s in ax.spines.values(): s.set_edgecolor('#555'); s.set_linewidth(1)

    # Noise map
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5,5), 1.5)
    noise_vis = np.abs(gray.astype(np.float32) - blurred)
    noise_vis = (noise_vis / noise_vis.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(noise_vis, cmap='viridis'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Noise Pattern", color=TC, fontsize=10)
    for s in ax.spines.values(): s.set_edgecolor('#555'); s.set_linewidth(1)

    # LBP histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#1a1a2e')
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    ax.bar(range(26), hist_lbp, color=RESULT_COLOR, alpha=0.8, width=0.8)
    ax.set_title("LBP Histogram", color=TC, fontsize=10)
    ax.tick_params(colors='#888', labelsize=7)
    ax.spines[:].set_color('#333355')

    # FFT radial profile
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#1a1a2e')
    h_img, w_img = gray.shape
    cy, cx = h_img//2, w_img//2
    Y_g, X_g = np.ogrid[:h_img, :w_img]
    R_g = np.sqrt((X_g-cx)**2 + (Y_g-cy)**2).astype(int)
    max_r = min(cy, cx); n_bins = 20
    edges = np.linspace(0, max_r, n_bins+1).astype(int)
    rp = np.zeros(n_bins, float)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(float)))))
    for i in range(n_bins):
        mask = (R_g >= edges[i]) & (R_g < edges[i+1])
        if mask.sum() > 0: rp[i] = mag[mask].mean()
    rp /= (rp.sum()+1e-8)
    ax.fill_between(range(n_bins), rp, alpha=0.4, color=RESULT_COLOR)
    ax.plot(rp, color=RESULT_COLOR, linewidth=2)
    ax.set_title("FFT Radial Profile", color=TC, fontsize=10)
    ax.tick_params(colors='#888', labelsize=7)
    ax.spines[:].set_color('#333355')

    # Wavelet energy
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('#1a1a2e')
    coeffs = pywt.wavedec2(gray.astype(float)/255., 'db4', level=3)
    energies, names = [], []
    for li, lc in enumerate(coeffs):
        for band, nm in zip((lc if isinstance(lc, tuple) else (lc,)),
                             ['cA','cH','cV','cD'] if isinstance(lc, tuple) else ['cA']):
            energies.append(np.sum(band**2))
            names.append(f"L{li}-{nm}")
    energies = np.array(energies[:9]); energies /= (energies.sum()+1e-8)
    colors_e = plt.cm.plasma(np.linspace(0.2, 0.9, len(energies)))
    ax.bar(range(len(energies)), energies, color=colors_e, width=0.7)
    ax.set_xticks(range(len(names[:9])))
    ax.set_xticklabels(names[:9], rotation=30, fontsize=6, color='#888')
    ax.set_title("Wavelet Sub-band Energy", color=TC, fontsize=10)
    ax.tick_params(colors='#888', labelsize=7)
    ax.spines[:].set_color('#333355')

    # Confidence gauge
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor('#1a1a2e')
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#333355', linewidth=8, solid_capstyle='round')
    conf_theta = np.pi - confidence * np.pi
    theta_fill = np.linspace(np.pi, conf_theta, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=RESULT_COLOR,
            linewidth=8, solid_capstyle='round')
    ax.plot([0, np.cos(conf_theta)*0.85], [0, np.sin(conf_theta)*0.85],
            color='white', linewidth=3, solid_capstyle='round')
    ax.text(0, -0.2, f"{confidence:.1%}", ha='center', fontsize=18,
            color=RESULT_COLOR, fontweight='bold')
    ax.text(0, -0.45, "Confidence", ha='center', fontsize=9, color=TC)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color('#333355')
    ax.set_title(label_text, color=RESULT_COLOR, fontsize=10, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Art Detector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg: #080812;
    --surface: #12121f;
    --surface2: #1a1a2e;
    --accent-real: #00d4aa;
    --accent-ai: #ff6b6b;
    --accent-gold: #ffd700;
    --text: #e2e2e2;
    --muted: #777;
    --border: #2a2a45;
    --radius: 12px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
    background-image: radial-gradient(ellipse at 20% 50%, rgba(0,212,170,0.04) 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 20%, rgba(255,107,107,0.04) 0%, transparent 50%);
  }

  header {
    padding: 28px 40px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
    background: rgba(18,18,31,0.8);
    backdrop-filter: blur(10px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { font-size: 28px; }
  header h1 {
    font-size: 20px; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent-real), var(--accent-gold));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  header p { font-size: 12px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
  .model-badge {
    margin-left: auto;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent-gold);
  }

  main { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }

  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    background: var(--surface);
    position: relative; overflow: hidden;
  }
  .upload-zone::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,212,170,0.03), rgba(255,107,107,0.03));
    opacity: 0; transition: opacity 0.3s;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent-real);
    box-shadow: 0 0 30px rgba(0,212,170,0.12);
  }
  .upload-zone:hover::before { opacity: 1; }
  .upload-icon { font-size: 52px; margin-bottom: 16px; }
  .upload-zone h2 { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
  .upload-zone p { color: var(--muted); font-size: 14px; }
  .upload-zone input { display: none; }

  .preview-wrap { margin-top: 20px; display: none; }
  .preview-wrap img { max-width: 100%; max-height: 320px; border-radius: var(--radius); object-fit: contain; }

  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, var(--accent-real), #00a88a);
    color: #000; font-weight: 700; font-size: 15px;
    padding: 14px 36px; border-radius: 10px; border: none;
    cursor: pointer; margin-top: 20px; transition: all 0.25s;
    font-family: 'Space Grotesk', sans-serif;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,212,170,0.35); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .result-card {
    margin-top: 36px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    display: none;
  }
  .result-header {
    padding: 24px 28px;
    display: flex; align-items: center; gap: 20px;
  }
  .result-icon { font-size: 44px; }
  .result-label { font-size: 26px; font-weight: 700; }
  .result-sub { font-size: 13px; color: var(--muted); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
  .result-real { border-top: 3px solid var(--accent-real); }
  .result-real .result-label { color: var(--accent-real); }
  .result-ai   { border-top: 3px solid var(--accent-ai); }
  .result-ai   .result-label { color: var(--accent-ai); }

  .metrics-row {
    display: flex; gap: 0;
    border-top: 1px solid var(--border);
  }
  .metric-box {
    flex: 1; padding: 20px; text-align: center;
    border-right: 1px solid var(--border);
  }
  .metric-box:last-child { border-right: none; }
  .metric-val { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .metric-key { font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

  .analysis-img {
    width: 100%; padding: 20px;
    border-top: 1px solid var(--border);
  }
  .analysis-img img { width: 100%; border-radius: var(--radius); }

  .loading { display: none; text-align: center; padding: 40px; }
  .spinner {
    width: 40px; height: 40px; border: 3px solid var(--border);
    border-top-color: var(--accent-real);
    border-radius: 50%; animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .info-cards {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 40px;
  }
  .info-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px;
  }
  .info-card h3 { font-size: 13px; color: var(--accent-gold); text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 10px; }
  .info-card p { font-size: 13px; color: var(--muted); line-height: 1.7; }
  .tag { display: inline-block; background: var(--surface2); border: 1px solid var(--border);
         border-radius: 6px; padding: 2px 8px; font-size: 11px; font-family: 'JetBrains Mono', monospace;
         color: var(--accent-real); margin: 2px; }

  .error-box {
    background: rgba(255,107,107,0.08); border: 1px solid var(--accent-ai);
    border-radius: var(--radius); padding: 16px; color: var(--accent-ai);
    font-size: 13px; margin-top: 16px; display: none;
  }

  footer { text-align: center; padding: 30px; color: var(--muted); font-size: 12px;
            font-family: 'JetBrains Mono', monospace; border-top: 1px solid var(--border); margin-top: 60px; }
</style>
</head>
<body>

<header>
  <span class="logo">🖼️</span>
  <div>
    <h1>AI Art Detector</h1>
    <p>Phân biệt tranh thật và tranh do AI tạo ra — ML only</p>
  </div>
  <div class="model-badge">
    {{ meta.model_name }} | {{ meta.pipeline }} | AUC {{ "%.3f"|format(meta.auc) }}
  </div>
</header>

<main>
  <div class="upload-zone" id="uploadZone">
    <div class="upload-icon">🎨</div>
    <h2>Upload một bức tranh</h2>
    <p>Kéo thả hoặc click để chọn ảnh • JPG, PNG, WEBP (tối đa 16MB)</p>
    <input type="file" id="fileInput" accept="image/*">
    <div class="preview-wrap" id="previewWrap">
      <img id="previewImg" src="" alt="preview">
    </div>
    <button class="btn" id="analyzeBtn" disabled onclick="analyze()">
      🔬 Phân tích
    </button>
  </div>

  <div class="error-box" id="errorBox"></div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p style="color: var(--muted); font-family: 'JetBrains Mono'; font-size: 13px;">Đang extract features & predict...</p>
  </div>

  <div class="result-card" id="resultCard">
    <div class="result-header" id="resultHeader">
      <span class="result-icon" id="resultIcon"></span>
      <div>
        <div class="result-label" id="resultLabel"></div>
        <div class="result-sub" id="resultSub"></div>
      </div>
    </div>
    <div class="metrics-row">
      <div class="metric-box">
        <div class="metric-val" id="mConf" style="color: var(--accent-gold)"></div>
        <div class="metric-key">Confidence</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" id="mProb0" style="color: var(--accent-real)"></div>
        <div class="metric-key">P(Real Art)</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" id="mProb1" style="color: var(--accent-ai)"></div>
        <div class="metric-key">P(AI Art)</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" id="mFeats" style="color: #aa88ff"></div>
        <div class="metric-key">Features</div>
      </div>
    </div>
    <div class="analysis-img">
      <img id="analysisImg" src="" alt="analysis">
    </div>
  </div>

  <div class="info-cards">
    <div class="info-card">
      <h3>⚙️ Pipeline Features</h3>
      <p>
        <span class="tag">Fractal Dim</span>
        <span class="tag">Wavelet DWT</span>
        <span class="tag">GLCM</span>
        <span class="tag">FFT Spectrum</span>
        <span class="tag">Noise Pattern</span>
        <span class="tag">LBP</span>
        <span class="tag">HOG</span>
        <span class="tag">Color Hist</span>
        <span class="tag">Hu Moments</span>
      </p>
    </div>
    <div class="info-card">
      <h3>🤖 Tại sao AI art khác biệt?</h3>
      <p>AI generators tạo ra noise pattern đều đặn bất thường, FFT spectrum mượt hơn tranh thật, LBP distribution thiếu tính ngẫu nhiên tự nhiên của brush stroke.</p>
    </div>
    <div class="info-card">
      <h3>📊 Model Performance</h3>
      <p>
        Model: <strong style="color: var(--accent-gold)">{{ meta.model_name }}</strong><br>
        Accuracy: <strong style="color: var(--accent-real)">{{ "%.2f%%"|format(meta.accuracy*100) }}</strong><br>
        AUC-ROC: <strong style="color: var(--accent-real)">{{ "%.4f"|format(meta.auc) }}</strong>
      </p>
    </div>
  </div>
</main>

<footer>AI Art Detector — ML Project | Fractal + Wavelet + GLCM + FFT + Noise + LBP</footer>

<script>
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg = document.getElementById('previewImg');
  const analyzeBtn = document.getElementById('analyzeBtn');
  let selectedFile = null;

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewWrap.style.display = 'block';
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;
    analyzeBtn.disabled = true;
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();

      document.getElementById('loading').style.display = 'none';
      analyzeBtn.disabled = false;

      if (!resp.ok || data.error) {
        const eb = document.getElementById('errorBox');
        eb.textContent = '⚠️ ' + (data.error || 'Có lỗi xảy ra');
        eb.style.display = 'block';
        return;
      }

      // Fill result
      const card = document.getElementById('resultCard');
      const isAI = data.prediction === 1;
      card.className = 'result-card ' + (isAI ? 'result-ai' : 'result-real');
      document.getElementById('resultIcon').textContent = isAI ? '🤖' : '🎨';
      document.getElementById('resultLabel').textContent = isAI ? 'AI Generated Art' : 'Real / Human Art';
      document.getElementById('resultSub').textContent =
        `Confidence: ${(data.confidence*100).toFixed(1)}% | Features: ${data.n_features}`;
      document.getElementById('mConf').textContent  = (data.confidence*100).toFixed(1) + '%';
      document.getElementById('mProb0').textContent = (data.prob_real*100).toFixed(1) + '%';
      document.getElementById('mProb1').textContent = (data.prob_ai*100).toFixed(1) + '%';
      document.getElementById('mFeats').textContent = data.n_features;
      document.getElementById('analysisImg').src = 'data:image/png;base64,' + data.figure;
      card.style.display = 'block';
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch(err) {
      document.getElementById('loading').style.display = 'none';
      analyzeBtn.disabled = false;
      const eb = document.getElementById('errorBox');
      eb.textContent = '⚠️ Network error: ' + err.message;
      eb.style.display = 'block';
    }
  }
</script>
</body>
</html>
"""


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, meta=META)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có file ảnh trong request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Tên file trống"}), 400

    try:
        # Đọc ảnh
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        rgb  = np.array(img)
        gray = np.array(img.convert("L"))

        # Extract features
        features = extract_all_features(rgb, gray)
        n_features = len(features)

        # Predict
        if MODEL is None:
            return jsonify({"error": "Model chưa được train. Chạy 03_train_models.py trước."}), 500

        X = features.reshape(1, -1)
        prediction = int(MODEL.predict(X)[0])
        try:
            probs = MODEL.predict_proba(X)[0]
            prob_real = float(probs[0])
            prob_ai   = float(probs[1])
        except Exception:
            prob_real = 1.0 - prediction
            prob_ai   = float(prediction)

        confidence = max(prob_real, prob_ai)

        # Generate analysis figure
        fig_b64 = make_analysis_figure(rgb, gray, prediction, confidence, features)

        return jsonify({
            "prediction" : prediction,
            "label"      : "AI Art" if prediction == 1 else "Real Art",
            "confidence" : round(confidence, 4),
            "prob_real"  : round(prob_real, 4),
            "prob_ai"    : round(prob_ai, 4),
            "n_features" : n_features,
            "figure"     : fig_b64,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status"     : "ok",
        "model"      : META.get("model_name"),
        "pipeline"   : META.get("pipeline"),
        "model_loaded": MODEL is not None,
    })


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 AI Art Detector Web App")
    print("="*55)
    print(f"  Model  : {META.get('model_name', 'N/A')}")
    print(f"  Pipeline: {META.get('pipeline', 'N/A')}")
    print(f"  URL    : http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)