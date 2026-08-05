# SteelDefectYOLO

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11%2Fv26-green)

Real-time steel surface defect detection using YOLO architectures, optimized for the NEU-DET dataset.

---

## Table of Contents

- [Features](#features)
- [Performance](#performance)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Configurations](#model-configurations)
- [ONNX Export](#onnx-export)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Features

- ✅ **Multi-model support**: YOLOv8, YOLOv11, YOLOv26 (n/s/m/l/x variants)
- ✅ **YOLO26 dual-head evaluation**: One-to-one (NMS-free) vs One-to-many (traditional)
- ✅ **Industrial metrics**: FNR, FPR, per-class performance analysis
- ✅ **ONNX export**: Production-ready deployment with FP16/INT8 quantization
- ✅ **Optimized for small datasets**: Specialized augmentation presets
- ✅ **Auto model download**: Models cached in `models/` directory

---

## Performance

### Best Models (NEU-DET validation set)

| Model | Detection Head | mAP@0.5 | Precision | Recall | Training Time | Inference (CPU) |
|-------|----------------|---------|-----------|--------|---------------|-----------------|
| **YOLO26n** | one-to-many | **84.33%** | 83.92% | 77.68% | 38 min | 64 FPS |
| YOLO26s | one-to-many | 83.94% | 83.10% | 77.96% | 68 min | ~50 FPS |
| YOLO11s | default | 83.40% | 78.60% | 78.60% | - | ~55 FPS |

**Key Findings:**
- YOLO26n achieves highest mAP@0.5 despite being the smallest model
- One-to-many detection head consistently outperforms one-to-one by ~1.5%
- Optimal for industrial deployment: high precision (83.9%) with acceptable recall (77.7%)

### Per-Class Performance (YOLO26n, one-to-many)

| Defect Type | mAP@0.5 | Precision | Recall | Notes |
|-------------|---------|-----------|--------|-------|
| scratches | 96.30% | 89.50% | 95.61% | Best detected |
| pitted_surface | 89.09% | 89.37% | 80.82% | Clear features |
| patches | 91.18% | 88.16% | 86.74% | Well-defined |
| inclusion | 84.68% | 76.53% | 82.29% | Moderate |
| crazing | 73.47% | 82.06% | 65.67% | Challenging (texture) |
| rolled-in_scale | 71.23% | 77.91% | 54.92% | Most difficult (elongated) |

---

## Dataset

### NEU-DET Overview

The NEU-DET dataset contains 1,800 grayscale images (200×200 px) of steel surface defects across 6 classes:

| Class ID | Class Name | Samples | Characteristics |
|----------|------------|---------|-----------------|
| 0 | crazing | 300 | Fine cracks, texture-based |
| 1 | inclusion | 300 | Foreign material, varied size |
| 2 | patches | 300 | Localized discoloration |
| 3 | pitted_surface | 300 | Small holes, corrosion |
| 4 | rolled-in_scale | 300 | Elongated defects, difficult |
| 5 | scratches | 300 | Linear marks, clear edges |

**Data Split**: 1,416 training / 354 validation (80/20 stratified)

**Download**: [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

---

## Installation

### Step 1: Create Project Directory

```bash
mkdir steel_defect_project
cd steel_defect_project
```

### Step 2: Set Up Python Environment

**Option A: Using venv (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Option B: Using conda**
```bash
conda create -n steel_defect python=3.10
conda activate steel_defect
```

### Step 3: Clone Repository

```bash
git clone https://github.com/MaxML154/SteelDefectYOLO.git
cd SteelDefectYOLO
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- `ultralytics==8.4.107` - YOLO framework (version pinned for reproducibility)
- `torch>=2.5.0` - PyTorch backend
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `pyyaml>=6.0` - Configuration files
- `onnxruntime` (optional) - ONNX inference

### Step 5: Prepare Dataset

1. Download NEU-DET dataset and extract to a directory (e.g., `../NEU-DET/`)

2. Update dataset path in `configs/neudet_cleaned.yaml`:
```yaml
path: /path/to/your/NEU-DET  # Update this line
train: labels_train.txt
val: labels_val.txt
```

3. The dataset should have this structure:
```
NEU-DET/
├── IMAGES/
│   ├── crazing_1.jpg
│   ├── crazing_2.jpg
│   └── ...
└── ANNOTATIONS/
    ├── crazing_1.xml
    ├── crazing_2.xml
    └── ...
```

### Step 6: Verify Installation

```bash
python src/train_yolo.py --help
```

If you see the help message, installation is successful!

---

## Quick Start

### 1. Train YOLO26n (Recommended)

```bash
python src/train_yolo.py \
  --config configs/yolo26n.yaml \
  --data configs/neudet_cleaned.yaml \
  --name yolo26n_experiment
```

**Model will be auto-downloaded** on first run and cached in `models/` directory.

### 2. Evaluate Model

#### Standard Evaluation
```bash
python src/evaluate_yolo.py \
  --model runs/detect/yolo26n_experiment/weights/best.pt \
  --data configs/neudet_cleaned.yaml \
  --imgsz 640
```

#### YOLO26 Dual-Head Evaluation (compare detection heads)
```bash
python src/evaluate_yolo.py \
  --model runs/detect/yolo26n_experiment/weights/best.pt \
  --data configs/neudet_cleaned.yaml \
  --imgsz 640 \
  --end2end both \
  --plots
```

### 3. Export to ONNX

```bash
python src/export_to_onnx.py \
  --model runs/detect/yolo26n_experiment/weights/best.pt \
  --end2end false \
  --imgsz 640 \
  --simplify
```

**Output:** `runs/detect/yolo26n_experiment/weights/best_one_to_many.onnx`

### 4. Test ONNX Model

```bash
python src/test_onnx_model.py \
  --model runs/detect/yolo26n_experiment/weights/best_one_to_many.onnx \
  --imgsz 640 \
  --benchmark 100
```

**Expected Performance (CPU):** ~64 FPS, ~16ms latency

---

## Model Configurations

Pre-configured YAML files in `configs/` directory:

### YOLO Model Configs

| Config | Model | Params | Best Use Case |
|--------|-------|--------|---------------|
| `yolo26n.yaml` | YOLOv26n | ~3M | **Production (best mAP)** |
| `yolo26s.yaml` | YOLOv26s | ~11M | High accuracy alternative |
| `yolo11s.yaml` | YOLOv11s | ~9M | Baseline comparison |
| `yolo8n.yaml` | YOLOv8n | ~3M | Legacy support |

### Dataset Configs

| Config | Description |
|--------|-------------|
| `neudet_cleaned.yaml` | **Recommended** - Cleaned labels, stratified split |
| `neudet.yaml` | Original dataset |
| `neudet_pseudo3.yaml` | Experimental (pseudo-labeling) |

### Training from Config

```bash
python src/train_yolo.py --config configs/yolo26n.yaml
```

Config file structure:
```yaml
model: yolo26n.pt
epochs: 150

train_params:
  imgsz: 640
  batch: 16
  optimizer: auto
  lr0: 0.001
  # ... additional params
```

---

## ONNX Export

### Basic Export (CPU/GPU deployment)

```bash
python src/export_to_onnx.py \
  --model runs/detect/<run>/weights/best.pt \
  --end2end false \
  --imgsz 640 \
  --simplify
```

### FP16 Export (GPU acceleration)

```bash
python src/export_to_onnx.py \
  --model runs/detect/<run>/weights/best.pt \
  --end2end false \
  --imgsz 640 \
  --half \
  --simplify
```

**Benefits:**
- Model size reduced by 50%
- ~2x faster inference on GPU
- Accuracy loss <0.5%

### Dynamic Batch Size

```bash
python src/export_to_onnx.py \
  --model runs/detect/<run>/weights/best.pt \
  --end2end false \
  --imgsz 640 \
  --dynamic \
  --simplify
```

### YOLO26 Detection Heads

YOLO26 models have dual detection heads:

| Head Mode | Flag | Description | When to Use |
|-----------|------|-------------|-------------|
| **one-to-many** | `--end2end false` | Traditional + NMS | **Production (best mAP)** |
| one-to-one | `--end2end true` | NMS-free, end-to-end | Real-time priority |

**Recommendation:** Use `--end2end false` for highest accuracy (84.33% mAP@0.5).

### Testing ONNX Model

```bash
# Basic test (random input)
python src/test_onnx_model.py \
  --model model.onnx \
  --benchmark 100

# Test with real image
python src/test_onnx_model.py \
  --model model.onnx \
  --image path/to/test.jpg \
  --benchmark 100

# GPU acceleration
python src/test_onnx_model.py \
  --model model.onnx \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --benchmark 1000
```

**Output includes:**
- Model metadata and shapes
- Inference time statistics (mean, P50, P95, P99)
- Throughput (FPS)
- Numerical validation (NaN/Inf checks)

---

## Project Structure

```
SteelDefectYOLO/
├── configs/               # Configuration files
│   ├── neudet_cleaned.yaml      # Dataset config (recommended)
│   ├── yolo26n.yaml             # YOLO26n model config
│   ├── yolo26s.yaml             # YOLO26s model config
│   ├── yolo11s.yaml             # YOLO11s model config
│   └── yolo8*.yaml              # YOLOv8 configs
├── src/                   # Source code
│   ├── data/
│   │   └── indus_argumentation.py  # Augmentation presets
│   ├── train_yolo.py             # Training script
│   ├── evaluate_yolo.py          # Evaluation with industrial metrics
│   ├── export_to_onnx.py         # ONNX export utility
│   └── test_onnx_model.py        # ONNX testing and benchmarking
├── tools/                 # Utilities
│   ├── converter_neudet.py       # XML to YOLO label converter
│   └── visualize.py              # Visualization tools
├── models/                # Downloaded model weights (auto-created)
├── runs/                  # Training outputs (auto-created)
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## Advanced Usage

### Augmentation Modes

Pass `--aug-mode` to customize data augmentation:

```bash
python src/train_yolo.py \
  --config configs/yolo26n.yaml \
  --aug-mode enhanced
```

| Mode | When to Use | Description |
|------|-------------|-------------|
| `default` | Recommended | Balanced augmentation for general training |
| `strong` | Overfitting | Aggressive augmentation for small datasets |
| `enhanced` | High accuracy | Advanced techniques for mAP > 0.80 |
| `edge` | Fine-tuning | Minimal augmentation for quantized models |

### Resume Training

```bash
python src/train_yolo.py \
  --model runs/detect/<run>/weights/last.pt \
  --resume
```

### Custom Hyperparameters

```bash
python src/train_yolo.py \
  --data configs/neudet_cleaned.yaml \
  --model yolo26n.pt \
  --epochs 200 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --patience 50
```

### Evaluation with Custom Thresholds

```bash
python src/evaluate_yolo.py \
  --model runs/detect/<run>/weights/best.pt \
  --data configs/neudet_cleaned.yaml \
  --conf 0.001 \
  --iou 0.6 \
  --save-txt \
  --save-json \
  --plots
```

---

## Industrial Metrics

The evaluation script computes specialized metrics for industrial defect detection:

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| **FNR** | 1 - Recall | <1% | False Negative Rate (missed defects) - **critical** |
| **FDR** | 1 - Precision | <5% | False Discovery Rate (false alarms) - acceptable |
| **mAP@0.5** | - | >80% | Mean Average Precision at IoU=0.5 |

**Priority:** Minimize FNR first (missing defects is costly), then optimize precision.

---

## Troubleshooting

### Common Issues

**1. Model download fails:**
```bash
# Manually download model from https://github.com/ultralytics/assets/releases
# Place in models/ directory: models/yolo26n.pt
```

**2. CUDA out of memory:**
```bash
# Reduce batch size
python src/train_yolo.py --config configs/yolo26n.yaml --batch 8
```

**3. Dataset path not found:**
```bash
# Update path in configs/neudet_cleaned.yaml to absolute path
path: /absolute/path/to/NEU-DET
```

**4. Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{steeldefectyolo2026,
  author = {MaxML154},
  title = {SteelDefectYOLO: Real-time Steel Surface Defect Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/MaxML154/SteelDefectYOLO}
}
```

### NEU-DET Dataset

```bibtex
@article{song2013noise,
  title={Noise-robust texture description using local contrast patterns via global measures},
  author={Song, Kechen and Yan, Yunhui},
  journal={IEEE Signal Processing Letters},
  volume={21},
  number={1},
  pages={93--97},
  year={2013},
  publisher={IEEE}
}
```

### YOLO Architectures

```bibtex
@software{Jocher_Ultralytics_YOLO_2023,
  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title = {Ultralytics YOLO},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  version = {8.0.0}
}
```

---

## References

This project builds upon and references the following works:

### Papers

1. **YOLOv8/v11/v26 Architecture**
   - Ultralytics YOLO Documentation: https://docs.ultralytics.com/

2. **NEU Surface Defect Database**
   - Song, K., & Yan, Y. (2013). "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects." *Applied Surface Science*, 285, 858-864.
   - Dataset: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/

3. **Industrial Defect Detection**
   - Božič, J., Tabernik, D., & Skočaj, D. (2021). "Mixed supervision for surface-defect detection: From weakly to fully supervised learning." *Computers in Industry*, 129, 103459.
   - Tabernik, D., Šela, S., Skvarč, J., & Skočaj, D. (2020). "Segmentation-based deep-learning approach for surface-defect detection." *Journal of Intelligent Manufacturing*, 31(3), 759-776.

4. **YOLO for Industrial Inspection**
   - Li, J., Su, Z., Geng, J., & Yin, Y. (2018). "Real-time detection of steel strip surface defects based on improved YOLO detection network." *IFAC-PapersOnLine*, 51(21), 76-81.

### Related Projects

1. **Ultralytics YOLO**
   - Repository: https://github.com/ultralytics/ultralytics
   - Official YOLO implementation with extensive features

2. **Surface Defect Detection Datasets**
   - NEU-CLS: https://github.com/abin24/Surface-Inspection-defect-detection-dataset
   - DAGM: https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html
   - KolektorSDD: https://www.vicos.si/resources/kolektorsdd/

3. **Industrial Anomaly Detection**
   - MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
   - AITEX: https://www.aitex.es/afid/

### Tools & Frameworks

- **PyTorch**: https://pytorch.org/
- **ONNX**: https://onnx.ai/
- **OpenCV**: https://opencv.org/

---

## License

This project is licensed under the MIT License - see below for details.

**Note:** The NEU-DET dataset and Ultralytics YOLO have their own licenses. Please ensure compliance with their terms.

---

## Acknowledgments

- **NEU-DET Dataset**: Northeastern University (China)
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **PyTorch**: https://pytorch.org/

---

## Contact

For questions or issues, please open an issue on GitHub:
- **Author**: MaxML154
- **Repository**: https://github.com/MaxML154/SteelDefectYOLO

---

**Last Updated**: 2026-08-05