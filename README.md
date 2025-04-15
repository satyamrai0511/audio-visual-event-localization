# 🎯 Audio-Visual Event Localization in Videos

A research-driven project to design an **autoregressive model** integrating both **audio** and **video** signals for **temporal event localization**. By aligning auditory and visual streams, the goal is to detect and classify meaningful events more precisely.

---

## 📌 Project Goals

- End-to-end **multimodal event localization** pipeline
- Autoregressive modeling of **audio + video** over time
- Fuse audio-visual features using a **bidirectional LSTM** (audio) and **CNN** (video), then a **GRU** decoder
- Achieve accurate event classification and localization in real or simulated data

---

## 📦 Project Structure

```text
audio-visual-event-localization/
├── data/                    # Sample videos, audios, and optional annotations
├── models/                  # Model definitions (fusion model, test scripts)
├── notebooks/               # EDA and experimentation notebooks
├── outputs/
│   ├── features/            # Saved audio features (.npy)
│   ├── frames/              # Extracted video frames
│   ├── labels/              # Simulated or real label files (.npy)
│   ├── inference/           # Inference outputs (predictions, logs, etc.)
│   └── results/             # Any evaluation or result logs
├── src/
│   ├── feature_extraction/  # audio_features.py, video_frames.py
│   ├── models/              # fusion_model.py, test_fusion_model.py
│   ├── preprocessing/       # sync.py, utils.py, generate_fake_labels.py
│   └── training/            # dataset.py, train.py, evaluate.py, inference.py
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # License file
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/satyamrai0511/audio-visual-event-localization.git
cd audio-visual-event-localization
```

### 2. Create & Activate Virtual Environment

#### Windows (Git Bash)
```bash
python -m venv venv
source venv/Scripts/activate
```
#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Export PYTHONPATH (if needed)

```bash
export PYTHONPATH=.
```

---

## 🚀 Pipeline Overview

### 1. **Preprocessing**
- **sync.py**: Checks and prints audio/video durations
- **utils.py**: Helper functions (time formatting, logging)
- **generate_fake_labels.py**: Create simulated event labels for demonstration

### 2. **Feature Extraction**
- **audio_features.py**: Extracts log-mel spectrograms from `.wav` files
- **video_frames.py**: Extract frames from `.mp4` videos at given FPS

### 3. **Dataset & Model**
- **dataset.py**: `AudioVisualDataset` merges audio features, frames, and optional labels
- **fusion_model.py**: Defines the autoregressive fusion model (audio encoder + video encoder + GRU decoder)

### 4. **Training & Evaluation**
- **train.py**: Trains the model with cross-entropy on simulated or real labels
- **evaluate.py**: Computes classification metrics (precision, recall, F1) and mAP

### 5. **Inference**
- **inference.py**: Runs a complete pipeline (extract features, load model, get predictions)

---

## 🏋️ Training

1. Place your `.wav` and `.mp4` files in `data/`  
2. Extract features:
   ```bash
   python src/feature_extraction/audio_features.py
   python src/feature_extraction/video_frames.py
   ```
3. (Optional) Generate fake labels:
   ```bash
   python src/preprocessing/generate_fake_labels.py
   ```
4. Run training:
   ```bash
   export PYTHONPATH=.
   python src/training/train.py
   ```
   This will produce `latest_model.pth`

---

## 📊 Evaluation

```bash
export PYTHONPATH=.
python src/training/evaluate.py
```

- Loads the model checkpoint
- Computes classification report (precision, recall, F1)
- Calculates per-class Average Precision (AP) → shown as mAP

---

## 🤖 Inference

```bash
export PYTHONPATH=.
python src/training/inference.py
```

- Takes `data/sample_video_1.mp4` + `data/sample_audio_1.wav`
- Extracts features
- Runs them through the model
- Prints a timeline of predicted classes

---

## 🧩 Example Usage

**Command:**
```bash
python src/training/inference.py \
    --audio_path data/sample_audio_1.wav \
    --video_path data/sample_video_1.mp4 \
    --model_path latest_model.pth
```
(Output in `outputs/inference`)

---

## 🧠 Future Work

- [ ] Integrate real event timestamps and classes
- [ ] Implement more complex fusion strategies (transformers, cross-attention)
- [ ] Add real-time streaming inference
- [ ] Deploy on Streamlit / FastAPI

---

## 🤝 Contributing

Contributions are welcome! Feel free to open a PR with suggestions, bug fixes, or new ideas.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

