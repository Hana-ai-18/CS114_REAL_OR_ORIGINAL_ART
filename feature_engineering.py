"""
02_feature_engineering.py
==========================
Feature extraction với nhiều hướng tiếp cận:

  PIPELINE A — Hướng đề xuất ban đầu:
    Fractal Dimension + Wavelet (DWT) + GLCM

  PIPELINE B — Hướng noise/frequency (thường tốt hơn với AI art):
    FFT Spectrum Features + Noise Analysis + LBP

  PIPELINE C — Hướng tổng hợp đầy đủ:
    A + B + HOG + Color Histogram + Statistical Moments

Lý do Pipeline B/C thường OUTPERFORM A với AI-generated images:
  - AI generators tạo ra patterns noise rất đặc trưng trong frequency domain
  - LBP bắt được texture micro-patterns mà AI lặp lại
  - FFT spectrum của AI art thường mượt hơn (thiếu high-freq noise tự nhiên)
"""

import json
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel
import pywt
from tqdm import tqdm


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    "manifest_path" : "processed/split_manifest.json",
    "features_dir"  : "processed/features",
    "image_size"    : (224, 224),
    # GLCM
    "glcm_distances": [1, 3, 5],
    "glcm_angles"   : [0, np.pi/4, np.pi/2, 3*np.pi/4],
    # Wavelet
    "wavelet_name"  : "db4",
    "wavelet_levels": 3,
    # LBP
    "lbp_radius"    : 3,
    "lbp_n_points"  : 24,
    # FFT
    "fft_bins"      : 20,           # số bin trong FFT radial profile
    # HOG
    "hog_orientations": 9,
    "hog_pixels_per_cell": (16, 16),
    "hog_cells_per_block": (2, 2),
}


# ═══════════════════════════════════════════════════════════
# NHÓM 1 — PIPELINE A: FRACTAL + WAVELET + GLCM
# ═══════════════════════════════════════════════════════════

def fractal_dimension(gray: np.ndarray, threshold: int = 128) -> float:
    """
    Ước lượng fractal dimension bằng Box-counting method.
    Tranh thật thường có FD cao hơn (phức tạp hơn) so với AI art.
    """
    binary = (gray > threshold).astype(np.uint8)
    scales, counts = [], []
    max_box = min(gray.shape) // 2
    for box_size in range(2, max_box, max(1, max_box // 20)):
        h, w = binary.shape
        h_trim = (h // box_size) * box_size
        w_trim = (w // box_size) * box_size
        cropped = binary[:h_trim, :w_trim]
        reshaped = cropped.reshape(h_trim // box_size, box_size,
                                   w_trim // box_size, box_size)
        box_sum = reshaped.sum(axis=(1, 3))
        count = np.count_nonzero(box_sum)
        if count > 0:
            scales.append(box_size)
            counts.append(count)

    if len(scales) < 2:
        return 0.0

    log_s = np.log(1.0 / np.array(scales, dtype=float))
    log_c = np.log(np.array(counts, dtype=float))
    slope, *_ = np.polyfit(log_s, log_c, 1)
    return float(slope)


def extract_fractal_features(gray: np.ndarray) -> np.ndarray:
    """
    3 features: FD toàn ảnh, FD góc trái, FD góc phải
    """
    h, w = gray.shape
    fd_full  = fractal_dimension(gray)
    fd_left  = fractal_dimension(gray[:, :w//2])
    fd_right = fractal_dimension(gray[:, w//2:])
    return np.array([fd_full, fd_left, fd_right], dtype=np.float32)


def extract_wavelet_features(gray: np.ndarray) -> np.ndarray:
    """
    Phân tích DWT đa mức, trích xuất energy + entropy của các sub-band.
    AI art thường có phân phối energy sub-band khác với tranh thật.
    """
    feats = []
    img = gray.astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(img, CONFIG["wavelet_name"], level=CONFIG["wavelet_levels"])

    for level_idx, level_coeffs in enumerate(coeffs):
        if isinstance(level_coeffs, tuple):
            sub_bands = level_coeffs  # (cH, cV, cD)
        else:
            sub_bands = (level_coeffs,)  # cA (approximation)

        for band in sub_bands:
            band_flat = band.flatten()
            energy  = float(np.sum(band_flat ** 2))
            mean    = float(np.mean(np.abs(band_flat)))
            std     = float(np.std(band_flat))
            # Entropy của histogram normalized
            hist, _ = np.histogram(band_flat, bins=32, density=True)
            hist_pos = hist[hist > 0]
            entropy  = float(-np.sum(hist_pos * np.log2(hist_pos + 1e-12)))
            kurtosis = float(stats.kurtosis(band_flat))
            feats.extend([energy, mean, std, entropy, kurtosis])

    return np.array(feats, dtype=np.float32)


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix features.
    Tính contrast, dissimilarity, homogeneity, energy, correlation,
    ASM trên nhiều distance × angle → bắt được texture structure.
    """
    gray_q = (gray // 16).astype(np.uint8)   # quantize: 256 → 16 levels
    gray_q = np.clip(gray_q, 0, 15)

    glcm = graycomatrix(
        gray_q,
        distances=CONFIG["glcm_distances"],
        angles=CONFIG["glcm_angles"],
        levels=16,
        symmetric=True,
        normed=True
    )

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()   # shape (n_dist, n_angle)
        feats.extend([vals.mean(), vals.std(), vals.max(), vals.min()])

    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# NHÓM 2 — PIPELINE B: FFT + NOISE + LBP
# ═══════════════════════════════════════════════════════════

def extract_fft_features(gray: np.ndarray) -> np.ndarray:
    """
    FFT Spectrum Analysis — LÝ DO QUAN TRỌNG:
    - AI-generated images có FFT spectrum 'mượt' bất thường
    - Tranh thật có high-frequency noise ngẫu nhiên tự nhiên
    - Radial profile của magnitude spectrum phân biệt rõ hai lớp
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.int32)

    max_r = min(cy, cx)
    n_bins = CONFIG["fft_bins"]
    bin_edges = np.linspace(0, max_r, n_bins + 1).astype(int)
    radial_profile = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i+1])
        if mask.sum() > 0:
            radial_profile[i] = float(mag[mask].mean())

    # Normalize
    radial_profile /= (radial_profile.sum() + 1e-8)

    # Thêm thống kê bổ sung
    slope, intercept = np.polyfit(np.arange(n_bins), radial_profile, 1)
    high_freq_ratio  = radial_profile[n_bins//2:].sum() / (radial_profile[:n_bins//2].sum() + 1e-8)

    extra = np.array([slope, intercept, high_freq_ratio,
                      radial_profile.std(), radial_profile.max()], dtype=np.float32)
    return np.concatenate([radial_profile, extra])


def extract_noise_features(gray: np.ndarray) -> np.ndarray:
    """
    Noise Pattern Analysis:
    - Estimate noise bằng cách trừ ảnh đã blur khỏi original
    - AI images thường có noise pattern regularity cao (lặp lại)
    - Phân tích auto-correlation của noise map
    """
    img_f = gray.astype(np.float32)

    # Gaussian residual noise
    blurred = cv2.GaussianBlur(img_f, (5, 5), 1.5)
    noise   = img_f - blurred

    # Laplacian residual
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Thống kê noise
    noise_mean = float(np.mean(np.abs(noise)))
    noise_std  = float(np.std(noise))
    noise_kurt = float(stats.kurtosis(noise.flatten()))
    noise_skew = float(stats.skew(noise.flatten()))

    # Auto-correlation của noise (center 5x5)
    noise_norm = noise / (noise.std() + 1e-8)
    autocorr_full = np.fft.ifft2(np.abs(np.fft.fft2(noise_norm))**2).real
    autocorr_full /= (autocorr_full.max() + 1e-8)
    h, w = autocorr_full.shape
    center_patch = autocorr_full[h//2-2:h//2+3, w//2-2:w//2+3].flatten()

    # Laplacian stats
    lap_mean = float(np.mean(np.abs(laplacian)))
    lap_std  = float(np.std(laplacian))

    feats = np.array([noise_mean, noise_std, noise_kurt, noise_skew,
                      lap_mean, lap_std], dtype=np.float32)
    return np.concatenate([feats, center_patch.astype(np.float32)])


def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern — bắt được texture micro-pattern.
    AI art có LBP histogram khác biệt do generation artifacts.
    """
    lbp = local_binary_pattern(
        gray,
        P=CONFIG["lbp_n_points"],
        R=CONFIG["lbp_radius"],
        method='uniform'
    )
    n_bins = CONFIG["lbp_n_points"] + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)

    # Thêm thống kê
    lbp_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-12)))
    lbp_std     = float(np.std(hist))
    lbp_max     = float(np.max(hist))

    extra = np.array([lbp_entropy, lbp_std, lbp_max], dtype=np.float32)
    return np.concatenate([hist.astype(np.float32), extra])


# ═══════════════════════════════════════════════════════════
# NHÓM 3 — BỔ SUNG CHO PIPELINE C
# ═══════════════════════════════════════════════════════════

def extract_hog_features(gray: np.ndarray) -> np.ndarray:
    """
    Histogram of Oriented Gradients — bắt cấu trúc cạnh và hình dạng.
    Tóm tắt thống kê thay vì trả về full HOG vector (quá dài).
    """
    from skimage.feature import hog
    hog_feats = hog(
        gray,
        orientations=CONFIG["hog_orientations"],
        pixels_per_cell=CONFIG["hog_pixels_per_cell"],
        cells_per_block=CONFIG["hog_cells_per_block"],
        visualize=False,
        feature_vector=True
    )
    # Tóm gọn thành thống kê thay vì ~1000 dims
    feats = np.array([
        hog_feats.mean(), hog_feats.std(), hog_feats.max(),
        hog_feats.min(), np.median(hog_feats),
        float(stats.skew(hog_feats)), float(stats.kurtosis(hog_feats)),
        np.percentile(hog_feats, 25), np.percentile(hog_feats, 75),
    ], dtype=np.float32)
    return feats


def extract_color_features(rgb: np.ndarray) -> np.ndarray:
    """
    Color histogram + statistical moments cho mỗi channel RGB + HSV.
    """
    feats = []
    # RGB stats
    for ch in range(3):
        ch_data = rgb[:, :, ch].flatten().astype(np.float32)
        hist, _ = np.histogram(ch_data, bins=32, range=(0, 256), density=True)
        feats.extend([
            ch_data.mean(), ch_data.std(),
            float(stats.skew(ch_data)), float(stats.kurtosis(ch_data)),
            hist.max(), float(-np.sum(hist[hist>0] * np.log2(hist[hist>0] + 1e-12)))
        ])

    # HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for ch in range(3):
        ch_data = hsv[:, :, ch].flatten()
        feats.extend([ch_data.mean(), ch_data.std()])

    return np.array(feats, dtype=np.float32)


def extract_statistical_moments(gray: np.ndarray) -> np.ndarray:
    """
    Hu moments + Zernike-like moments + edge density
    """
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    # Contrast via local std
    local_std = ndimage.generic_filter(gray.astype(float), np.std, size=9)
    local_mean = float(local_std.mean())
    local_max  = float(local_std.max())

    return np.concatenate([
        hu_log.astype(np.float32),
        np.array([edge_density, local_mean, local_max], dtype=np.float32)
    ])


# ═══════════════════════════════════════════════════════════
# PIPELINE WRAPPER
# ═══════════════════════════════════════════════════════════

def load_image(path: str, size: tuple = (224, 224)):
    """Load ảnh, trả về (rgb_array, gray_array)"""
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    rgb  = np.array(img)
    gray = np.array(img.convert("L"))
    return rgb, gray


PIPELINE_EXTRACTORS = {
    "pipeline_A": lambda rgb, gray: np.concatenate([
        extract_fractal_features(gray),   # 3
        extract_wavelet_features(gray),    # ~120
        extract_glcm_features(gray),       # 24
    ]),
    "pipeline_B": lambda rgb, gray: np.concatenate([
        extract_fft_features(gray),        # 25
        extract_noise_features(gray),      # 31
        extract_lbp_features(gray),        # 29
    ]),
    "pipeline_C": lambda rgb, gray: np.concatenate([
        extract_fractal_features(gray),    # 3
        extract_wavelet_features(gray),    # ~120
        extract_glcm_features(gray),       # 24
        extract_fft_features(gray),        # 25
        extract_noise_features(gray),      # 31
        extract_lbp_features(gray),        # 29
        extract_hog_features(gray),        # 9
        extract_color_features(rgb),       # 36
        extract_statistical_moments(gray), # 10
    ]),
}


def extract_features_for_split(split_items: list, pipeline: str, desc: str = "") -> tuple:
    """
    Extract features cho một danh sách (path, label).
    Returns: (X: np.ndarray, y: np.ndarray)
    """
    extractor = PIPELINE_EXTRACTORS[pipeline]
    X_list, y_list = [], []

    for path, label in tqdm(split_items, desc=desc, ncols=80):
        try:
            rgb, gray = load_image(path)
            feat = extractor(rgb, gray)
            if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                X_list.append(feat)
                y_list.append(label)
        except Exception as e:
            pass  # bỏ qua ảnh lỗi

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ═══════════════════════════════════════════════════════════
# VISUALIZE FEATURES
# ═══════════════════════════════════════════════════════════

def visualize_feature_extraction(sample_real: str, sample_ai: str,
                                  save_path: str = "plots/02_feature_extraction.png"):
    """
    Visualize từng bước feature extraction trên 1 ảnh real và 1 ảnh AI.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rgb_r, gray_r = load_image(sample_real)
    rgb_a, gray_a = load_image(sample_ai)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0a0a15')
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.35)

    TC = '#e0e0e0'
    REAL_C = '#00d4aa'
    AI_C   = '#ff6b6b'

    fig.text(0.5, 0.97, "🔬 Feature Extraction Visualization — Real vs AI Art",
             ha='center', fontsize=16, color='white', fontweight='bold', fontfamily='monospace')

    def add_image_axis(gs_loc, img, title, cmap=None, border_color=REAL_C):
        ax = fig.add_subplot(gs_loc)
        ax.set_facecolor('#1a1a2e')
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=TC, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color); spine.set_linewidth(1.5)
        return ax

    def add_plot_axis(gs_loc, title):
        ax = fig.add_subplot(gs_loc)
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color=TC, fontsize=8)
        ax.tick_params(colors='#888', labelsize=7)
        ax.spines[:].set_color('#333355')
        return ax

    # ── Row 0: Ảnh gốc & grayscale ──────────
    add_image_axis(gs[0, 0], rgb_r,  "Real Art (RGB)",       border_color=REAL_C)
    add_image_axis(gs[0, 1], gray_r, "Real (Grayscale)",  cmap='gray', border_color=REAL_C)
    add_image_axis(gs[0, 2], rgb_a,  "AI Art (RGB)",        border_color=AI_C)
    add_image_axis(gs[0, 3], gray_a, "AI (Grayscale)",    cmap='gray', border_color=AI_C)

    # LBP visualization
    lbp_r = local_binary_pattern(gray_r, P=24, R=3, method='uniform')
    lbp_a = local_binary_pattern(gray_a, P=24, R=3, method='uniform')
    add_image_axis(gs[0, 4], lbp_r, "Real LBP map", cmap='inferno', border_color=REAL_C)
    add_image_axis(gs[0, 5], lbp_a, "AI LBP map",   cmap='inferno', border_color=AI_C)

    # ── Row 1: FFT magnitude spectrum ────────
    def fft_mag(gray):
        f = np.fft.fft2(gray.astype(float))
        return np.log1p(np.abs(np.fft.fftshift(f)))

    fft_r = fft_mag(gray_r)
    fft_a = fft_mag(gray_a)
    add_image_axis(gs[1, 0], fft_r, "Real FFT Spectrum", cmap='hot', border_color=REAL_C)
    add_image_axis(gs[1, 1], fft_a, "AI FFT Spectrum",   cmap='hot', border_color=AI_C)

    # Radial profile comparison
    ax_fft = add_plot_axis(gs[1, 2:4], "FFT Radial Profile (Log)")
    for gray, label, color in [(gray_r, 'Real', REAL_C), (gray_a, 'AI', AI_C)]:
        feat = extract_fft_features(gray)[:CONFIG["fft_bins"]]
        ax_fft.plot(feat, color=color, label=label, linewidth=2)
    ax_fft.set_xlabel('Frequency bin', color='#888', fontsize=7)
    ax_fft.set_ylabel('Norm. power', color='#888', fontsize=7)
    ax_fft.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=7)

    # Noise map
    def noise_map(gray):
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5,5), 1.5)
        noise = np.abs(gray.astype(np.float32) - blurred)
        return (noise / noise.max() * 255).astype(np.uint8)

    add_image_axis(gs[1, 4], noise_map(gray_r), "Real Noise Map", cmap='viridis', border_color=REAL_C)
    add_image_axis(gs[1, 5], noise_map(gray_a), "AI Noise Map",   cmap='viridis', border_color=AI_C)

    # ── Row 2: Wavelet sub-bands ─────────────
    for col_off, (gray, cls, bc) in enumerate([(gray_r, 'Real', REAL_C), (gray_a, 'AI', AI_C)]):
        coeffs = pywt.wavedec2(gray.astype(np.float32)/255., 'db4', level=2)
        cA, (cH, cV, cD) = coeffs[0], coeffs[1]
        for sub_idx, (sub, name) in enumerate([(cA,'Approx'), (cH,'Horiz'), (cV,'Vert')]):
            ax = add_image_axis(gs[2, col_off*3 + sub_idx],
                                f"{cls} Wavelet-{name}", cmap='RdBu_r', border_color=bc)
            ax.imshow(sub, cmap='RdBu_r', vmin=sub.min(), vmax=sub.max())

    # ── Row 3: LBP histogram + GLCM energy ──
    ax_lbp = add_plot_axis(gs[3, 0:2], "LBP Histogram")
    for gray, label, color in [(gray_r, 'Real', REAL_C), (gray_a, 'AI', AI_C)]:
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        ax_lbp.bar(np.arange(26), hist, alpha=0.6, label=label, color=color, width=0.8)
    ax_lbp.set_xlabel('LBP code', color='#888', fontsize=7)
    ax_lbp.set_ylabel('Density', color='#888', fontsize=7)
    ax_lbp.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=7)

    # Wavelet energy bars
    ax_wav = add_plot_axis(gs[3, 2:4], "Wavelet Sub-band Energy")
    band_names = ['L3-cA', 'L3-cH', 'L3-cV', 'L3-cD', 'L2-cH', 'L2-cV', 'L2-cD', 'L1-cH', 'L1-cV', 'L1-cD']
    for gray, label, color in [(gray_r, 'Real', REAL_C), (gray_a, 'AI', AI_C)]:
        coeffs = pywt.wavedec2(gray.astype(np.float32)/255., 'db4', level=3)
        energies = []
        for c in coeffs:
            if isinstance(c, tuple):
                for sub in c: energies.append(np.sum(sub**2))
            else:
                energies.append(np.sum(c**2))
        energies = np.array(energies[:10])
        energies /= energies.sum() + 1e-8
        x = np.arange(len(energies))
        ax_wav.bar(x + (0.2 if label=='AI' else -0.2), energies,
                   width=0.35, alpha=0.75, label=label, color=color)
    ax_wav.set_xticks(range(len(band_names)))
    ax_wav.set_xticklabels(band_names, rotation=30, fontsize=6, color='#888')
    ax_wav.set_ylabel('Norm. Energy', color='#888', fontsize=7)
    ax_wav.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=7)

    # GLCM
    ax_glcm = add_plot_axis(gs[3, 4:], "GLCM Properties")
    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    x = np.arange(len(props))
    for gray, label, color in [(gray_r, 'Real', REAL_C), (gray_a, 'AI', AI_C)]:
        gq = (gray // 16).astype(np.uint8); gq = np.clip(gq, 0, 15)
        glcm = graycomatrix(gq, [1], [0, np.pi/2], levels=16, symmetric=True, normed=True)
        vals = [graycoprops(glcm, p).mean() for p in props]
        ax_glcm.bar(x + (0.2 if label=='AI' else -0.2), vals,
                    width=0.35, alpha=0.75, label=label, color=color)
    ax_glcm.set_xticks(x)
    ax_glcm.set_xticklabels(props, rotation=15, fontsize=7, color='#888')
    ax_glcm.set_ylabel('Value', color='#888', fontsize=7)
    ax_glcm.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=7)

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def visualize_feature_importance_preview(X: np.ndarray, y: np.ndarray,
                                          pipeline: str,
                                          save_path: str = None):
    """Sơ bộ feature variance giữa 2 class để thấy khả năng phân biệt"""
    if save_path is None:
        save_path = f"plots/02_feature_variance_{pipeline}.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X0 = X[y == 0]
    X1 = X[y == 1]

    # Tính effect size (Cohen's d) cho từng feature
    pooled_std = np.sqrt((X0.var(axis=0) + X1.var(axis=0)) / 2 + 1e-8)
    cohen_d = np.abs(X0.mean(axis=0) - X1.mean(axis=0)) / pooled_std

    top_k = min(30, len(cohen_d))
    top_idx = np.argsort(cohen_d)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    bars = ax.barh(range(top_k), cohen_d[top_idx],
                   color=[plt.cm.plasma(v / (cohen_d[top_idx].max() + 1e-8))
                          for v in cohen_d[top_idx]])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"feat_{i}" for i in top_idx], fontsize=8, color='#ccc')
    ax.set_xlabel("Cohen's d  (effect size)", color='#ccc')
    ax.set_title(f"Top {top_k} Discriminative Features — {pipeline}",
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#888')
    ax.spines[:].set_color('#333355')
    ax.axvline(0.5, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.52, top_k - 1, "Medium effect", color='yellow', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BƯỚC 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Load manifest
    with open(CONFIG["manifest_path"]) as f:
        manifest = json.load(f)

    splits = {
        sn: [(item["path"], item["label"]) for item in items]
        for sn, items in manifest.items()
    }

    out_dir = Path(CONFIG["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Visualize 1 mẫu real và 1 mẫu AI
    real_items = [(p, l) for p, l in splits["train"] if l == 0]
    ai_items   = [(p, l) for p, l in splits["train"] if l == 1]

    if real_items and ai_items:
        print("\n[VIZ] Visualize feature extraction...")
        visualize_feature_extraction(real_items[0][0], ai_items[0][0])
    else:
        print("[WARN] Không đủ mẫu để visualize.")

    # Extract features cho từng pipeline
    for pipeline_name in PIPELINE_EXTRACTORS:
        print(f"\n{'─'*50}")
        print(f"  Pipeline: {pipeline_name.upper()}")
        print(f"{'─'*50}")

        pipeline_data = {}
        for split_name in ["train", "val", "test"]:
            t0 = time.time()
            X, y = extract_features_for_split(
                splits[split_name], pipeline_name,
                desc=f"  {split_name:5s}"
            )
            elapsed = time.time() - t0
            print(f"  {split_name}: X={X.shape}, y={y.shape} | {elapsed:.1f}s")
            pipeline_data[split_name] = {"X": X, "y": y}

        # Lưu features
        save_path = out_dir / f"{pipeline_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(pipeline_data, f)
        print(f"  [SAVE] {save_path}")

        # Visualize effect size
        visualize_feature_importance_preview(
            pipeline_data["train"]["X"],
            pipeline_data["train"]["y"],
            pipeline_name
        )

    print("\n✅ Hoàn thành bước 2!")
    print("   → Chạy tiếp: python 03_train_models.py")


if __name__ == "__main__":
    main()