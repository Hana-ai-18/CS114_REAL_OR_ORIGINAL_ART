"""
01_data_preprocessing.py
========================
Xử lý và chuẩn bị dữ liệu từ dataset Real-AI-Art (Kaggle)
Dataset structure:
    dataset/
        real_art/    -> tranh thật
        ai_art/      -> tranh AI
"""

import os
import sys
import shutil
import random
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image, ImageStat
from sklearn.model_selection import train_test_split
import cv2
from collections import defaultdict

# ─────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────
CONFIG = {
    "dataset_root"  : "dataset",           # thư mục gốc chứa dataset
    "output_root"   : "processed",         # thư mục output
    "image_size"    : (224, 224),          # resize về kích thước chuẩn
    "test_size"     : 0.15,
    "val_size"      : 0.15,
    "random_seed"   : 42,
    "min_file_size" : 5_000,               # bỏ ảnh < 5KB (có thể bị corrupt)
    "supported_ext" : {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "class_names"   : ["real_art", "ai_art"],
    "label_map"     : {"real_art": 0, "ai_art": 1},
}

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])


# ─────────────────────────────────────────
# 1. QUÉT VÀ KIỂM TRA DATASET
# ─────────────────────────────────────────
def scan_dataset(root: str) -> dict:
    """Quét toàn bộ dataset, trả về dict {class: [paths]}"""
    root = Path(root)
    data = defaultdict(list)

    for cls in CONFIG["class_names"]:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"[WARN] Không tìm thấy thư mục: {cls_dir}")
            continue
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in CONFIG["supported_ext"]:
                if f.stat().st_size >= CONFIG["min_file_size"]:
                    data[cls].append(str(f))
                else:
                    print(f"[SKIP] File quá nhỏ (có thể lỗi): {f.name}")

    return dict(data)


def verify_images(paths: list) -> tuple:
    """Kiểm tra ảnh có đọc được không, trả về (valid_paths, corrupt_paths)"""
    valid, corrupt = [], []
    for p in paths:
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except Exception:
            corrupt.append(p)
    return valid, corrupt


def get_image_stats(paths: list, sample_size: int = 200) -> dict:
    """Thống kê kích thước, mode, tỉ lệ aspect ratio của ảnh"""
    sample = random.sample(paths, min(sample_size, len(paths)))
    widths, heights, modes, ratios = [], [], [], []

    for p in sample:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                modes.append(img.mode)
                ratios.append(w / h)
        except Exception:
            pass

    return {
        "count"         : len(paths),
        "mean_width"    : np.mean(widths),
        "mean_height"   : np.mean(heights),
        "std_width"     : np.std(widths),
        "std_height"    : np.std(heights),
        "min_wh"        : (min(widths), min(heights)),
        "max_wh"        : (max(widths), max(heights)),
        "modes"         : dict(zip(*np.unique(modes, return_counts=True))),
        "mean_ratio"    : np.mean(ratios),
    }


# ─────────────────────────────────────────
# 2. VISUALIZE DATASET
# ─────────────────────────────────────────
def plot_dataset_overview(data: dict, stats: dict, save_path: str = "plots/01_dataset_overview.png"):
    """Vẽ tổng quan dataset: phân phối class, kích thước, mẫu ảnh"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0f0f1a')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"real_art": "#00d4aa", "ai_art": "#ff6b6b"}
    TEXT_COLOR = "#e0e0e0"

    # ── Tiêu đề ──────────────────────────────
    fig.text(0.5, 0.97, "🎨 Dataset Overview — Real Art vs AI Art",
             ha='center', va='top', fontsize=18, color='white',
             fontweight='bold', fontfamily='monospace')

    # ── 1. Biểu đồ cột số lượng ảnh ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    counts = {cls: len(paths) for cls, paths in data.items()}
    bars = ax1.bar(counts.keys(), counts.values(),
                   color=[COLORS[c] for c in counts.keys()],
                   edgecolor='white', linewidth=0.5, width=0.5)
    for bar, (cls, cnt) in zip(bars, counts.items()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{cnt:,}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax1.set_title('Số lượng ảnh theo class', color=TEXT_COLOR, fontsize=11)
    ax1.set_ylabel('Số ảnh', color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.spines[:].set_color('#333355')

    # ── 2. Pie chart tỉ lệ ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')
    wedges, texts, autotexts = ax2.pie(
        counts.values(), labels=counts.keys(),
        colors=[COLORS[c] for c in counts.keys()],
        autopct='%1.1f%%', startangle=90,
        textprops={'color': TEXT_COLOR, 'fontsize': 10},
        wedgeprops={'edgecolor': '#0f0f1a', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    ax2.set_title('Tỉ lệ class', color=TEXT_COLOR, fontsize=11)

    # ── 3. Phân phối chiều rộng ──────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1a1a2e')
    for cls, paths in data.items():
        sample = random.sample(paths, min(300, len(paths)))
        widths = []
        for p in sample:
            try:
                with Image.open(p) as img:
                    widths.append(img.size[0])
            except Exception:
                pass
        ax3.hist(widths, bins=30, alpha=0.7, label=cls, color=COLORS[cls], edgecolor='none')
    ax3.set_title('Phân phối chiều rộng (px)', color=TEXT_COLOR, fontsize=11)
    ax3.set_xlabel('Width', color=TEXT_COLOR)
    ax3.set_ylabel('Frequency', color=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.spines[:].set_color('#333355')
    ax3.legend(facecolor='#1a1a2e', labelcolor=TEXT_COLOR, fontsize=9)

    # ── 4. Phân phối chiều cao ───────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor('#1a1a2e')
    for cls, paths in data.items():
        sample = random.sample(paths, min(300, len(paths)))
        heights = []
        for p in sample:
            try:
                with Image.open(p) as img:
                    heights.append(img.size[1])
            except Exception:
                pass
        ax4.hist(heights, bins=30, alpha=0.7, label=cls, color=COLORS[cls], edgecolor='none')
    ax4.set_title('Phân phối chiều cao (px)', color=TEXT_COLOR, fontsize=11)
    ax4.set_xlabel('Height', color=TEXT_COLOR)
    ax4.tick_params(colors=TEXT_COLOR)
    ax4.spines[:].set_color('#333355')
    ax4.legend(facecolor='#1a1a2e', labelcolor=TEXT_COLOR, fontsize=9)

    # ── 5. Sample ảnh real_art ───────────────
    for i, cls in enumerate(CONFIG["class_names"]):
        paths = data.get(cls, [])
        samples = random.sample(paths, min(4, len(paths)))
        for j, p in enumerate(samples):
            ax = fig.add_subplot(gs[1 + i, j])
            ax.set_facecolor('#1a1a2e')
            try:
                img = Image.open(p).convert("RGB").resize((160, 160))
                ax.imshow(np.array(img))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(COLORS[cls])
                    spine.set_linewidth(2)
                if j == 0:
                    ax.set_ylabel(cls.replace('_', ' ').title(),
                                  color=COLORS[cls], fontsize=10, fontweight='bold')
                ax.set_title(f"#{j+1}", color=TEXT_COLOR, fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_pixel_distribution(data: dict, save_path: str = "plots/01_pixel_distribution.png"):
    """So sánh phân phối pixel RGB giữa real và AI art"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle('Phân phối pixel RGB: Real vs AI Art', color='white', fontsize=14, fontweight='bold')

    COLORS_CH = {'real_art': ['#ff4444', '#44ff44', '#4488ff'],
                 'ai_art':   ['#ff9944', '#99ff44', '#ff44ff']}
    CHANNELS = ['Red', 'Green', 'Blue']

    pixel_data = defaultdict(lambda: [[], [], []])

    for cls, paths in data.items():
        sample = random.sample(paths, min(100, len(paths)))
        for p in sample:
            try:
                arr = np.array(Image.open(p).convert("RGB").resize((64, 64))).reshape(-1, 3)
                for ch in range(3):
                    pixel_data[cls][ch].extend(arr[:, ch].tolist())
            except Exception:
                pass

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, CHANNELS)):
        ax.set_facecolor('#1a1a2e')
        for cls in CONFIG["class_names"]:
            vals = pixel_data[cls][ch_idx]
            ax.hist(vals, bins=50, alpha=0.65, density=True,
                    label=cls.replace('_', ' ').title(),
                    color=COLORS_CH[cls][ch_idx], edgecolor='none')
        ax.set_title(f'{ch_name} channel', color='white', fontsize=12)
        ax.set_xlabel('Pixel value (0-255)', color='#aaa')
        ax.set_ylabel('Density', color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.spines[:].set_color('#333355')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


# ─────────────────────────────────────────
# 3. TIỀN XỬ LÝ & RESIZE
# ─────────────────────────────────────────
def preprocess_image(path: str, size: tuple = CONFIG["image_size"]) -> np.ndarray | None:
    """
    Đọc ảnh, resize, chuẩn hóa.
    Trả về numpy array shape (H, W, 3), dtype uint8
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"[ERROR] Cannot load {path}: {e}")
        return None


def build_splits(data: dict) -> dict:
    """
    Tạo train/val/test split stratified theo class.
    Returns: {"train": [(path, label)], "val": ..., "test": ...}
    """
    all_paths, all_labels = [], []
    for cls, paths in data.items():
        label = CONFIG["label_map"][cls]
        all_paths.extend(paths)
        all_labels.extend([label] * len(paths))

    # Split: train 70 / val 15 / test 15
    X_tv, X_test, y_tv, y_test = train_test_split(
        all_paths, all_labels,
        test_size=CONFIG["test_size"],
        stratify=all_labels,
        random_state=CONFIG["random_seed"]
    )
    val_ratio = CONFIG["val_size"] / (1 - CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_ratio,
        stratify=y_tv,
        random_state=CONFIG["random_seed"]
    )

    return {
        "train": list(zip(X_train, y_train)),
        "val"  : list(zip(X_val,   y_val)),
        "test" : list(zip(X_test,  y_test)),
    }


def save_split_manifest(splits: dict, save_path: str = "processed/split_manifest.json"):
    """Lưu danh sách file theo split để dùng lại"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        split: [{"path": p, "label": l} for p, l in items]
        for split, items in splits.items()
    }
    with open(save_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVE] Manifest saved: {save_path}")


def plot_split_distribution(splits: dict, save_path: str = "plots/01_split_distribution.png"):
    """Visualize phân phối train/val/test"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    inv_label = {v: k for k, v in CONFIG["label_map"].items()}
    split_names = list(splits.keys())
    split_data = {sn: defaultdict(int) for sn in split_names}

    for sn, items in splits.items():
        for _, label in items:
            split_data[sn][inv_label[label]] += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle('Phân phối dữ liệu Train / Val / Test', color='white', fontsize=14, fontweight='bold')

    COLORS = {"real_art": "#00d4aa", "ai_art": "#ff6b6b"}
    SPLIT_COLORS = {"train": "#5555ff", "val": "#ffaa00", "test": "#ff5555"}

    for ax, sn in zip(axes, split_names):
        ax.set_facecolor('#1a1a2e')
        counts = split_data[sn]
        bars = ax.bar(counts.keys(), counts.values(),
                      color=[COLORS[c] for c in counts.keys()],
                      edgecolor='white', linewidth=0.5, width=0.4)
        for bar, (cls, cnt) in zip(bars, counts.items()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(cnt), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
        total = sum(counts.values())
        ax.set_title(f'{sn.upper()}  (n={total})', color=SPLIT_COLORS[sn], fontsize=12, fontweight='bold')
        ax.tick_params(colors='#aaa')
        ax.set_ylabel('Số ảnh', color='#aaa')
        ax.spines[:].set_color('#333355')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


# ─────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BƯỚC 1: XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # 1. Quét dataset
    print("\n[1/5] Quét dataset...")
    data = scan_dataset(CONFIG["dataset_root"])
    if not data:
        print("[ERROR] Không tìm thấy dữ liệu. Kiểm tra lại đường dẫn dataset_root.")
        sys.exit(1)

    for cls, paths in data.items():
        print(f"  {cls}: {len(paths)} files")

    # 2. Kiểm tra ảnh corrupt
    print("\n[2/5] Kiểm tra ảnh lỗi...")
    clean_data = {}
    for cls, paths in data.items():
        valid, corrupt = verify_images(paths)
        print(f"  {cls}: {len(valid)} valid | {len(corrupt)} corrupt")
        clean_data[cls] = valid

    # 3. Thống kê
    print("\n[3/5] Thống kê kích thước ảnh...")
    for cls, paths in clean_data.items():
        stats = get_image_stats(paths)
        print(f"  [{cls}]")
        print(f"    Count       : {stats['count']}")
        print(f"    Mean W×H    : {stats['mean_width']:.0f} × {stats['mean_height']:.0f} px")
        print(f"    Std W×H     : {stats['std_width']:.0f} × {stats['std_height']:.0f} px")
        print(f"    Max W×H     : {stats['max_wh']}")
        print(f"    Aspect ratio: {stats['mean_ratio']:.3f}")
        print(f"    Color modes : {stats['modes']}")

    # 4. Visualize
    print("\n[4/5] Tạo biểu đồ...")
    all_stats = {cls: get_image_stats(paths) for cls, paths in clean_data.items()}
    plot_dataset_overview(clean_data, all_stats)
    plot_pixel_distribution(clean_data)

    # 5. Tạo splits
    print("\n[5/5] Tạo train/val/test split...")
    splits = build_splits(clean_data)
    for sn, items in splits.items():
        labels = [l for _, l in items]
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  {sn:5s}: {len(items)} items | real={dist.get(0,0)} | ai={dist.get(1,0)}")

    save_split_manifest(splits)
    plot_split_distribution(splits)

    print("\n✅ Hoàn thành bước 1! Plots saved trong thư mục plots/")
    print("   → Chạy tiếp: python 02_feature_engineering.py")


if __name__ == "__main__":
    main()