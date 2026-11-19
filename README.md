# Liveness Detection — Passive Face Anti-Spoofing + Active Challenge–Response

A two-stage face **liveness detection** system that combines a **passive** deep-learning anti-spoofing model with an **active** challenge-response stage, delivered as a single real-time web application.

- **Passive stage** — a multi-stream model (*HybridPAD*) inspects a short recording for visual and motion cues of spoofing and returns a **LIVE / SPOOF** verdict.
- **Active stage** — the user is asked to perform three randomly chosen actions in real time; the system checks they are done correctly, in order, and by the **same person**.
- A final decision is issued only when **both** stages agree.

> Each stage covers the other's blind spot: the passive model catches presentation patterns, while the active challenge proves a live person is responding *now*.

---

## Repository structure

```
.
├── notebooks/                        # experiments & evaluation (see "Notebooks" below)
│   ├── OULU-NPU_Protocols.ipynb
│   ├── SiW-Mv2_Protocols.ipynb
│   ├── Cross_Dataset_Evaluation.ipynb
│   ├── Combined_Pool_Model_1.ipynb
│   ├── Combined_Pool_Model_2.ipynb
│   ├── Combined_Pool_Model_3.ipynb
│   └── FAS_Inference_load_best_model.ipynb
│
└── HybridPAD_app/                    # the final real-time web app
    ├── Functions/                    # all module files (config, pipeline, model, loss, ...)
    ├── main_fas.py                   # entry point — starts the application
    ├── mediapipe detectors           # face / hand landmark + gesture detectors
    └── output_combinedData/model_1   # contains the best model checkpoints (not included in the repo as it is a large file)
```

---

## Notebooks

| Notebook | What it does |
|----------|-------------|
| **OULU-NPU Protocols** | Trains and evaluates the passive model on the OULU-NPU protocols (Protocol I–IV, including the leave-one-camera-out and all-unseen folds). |
| **SiW-Mv2 protocols** | Trains and evaluates on the SiW-Mv2 protocols (known attacks, leave-one-attack-out, cross-domain). |
| **Cross-dataset evaluation** | Trains on one dataset and tests on the other (OULU-NPU ↔ SiW-Mv2) to measure cross-domain generalisation. |
| **Combined Pool — Model 1** | Baseline combined model trained on the pooled OULU-NPU + SiW-Mv2 data, with an unseen held-out test set. |
| **Combined Pool — Model 2** | Model 1 with a revised data-augmentation pipeline. |
| **Combined Pool — Model 3** | Model 2 extended with the supervised-contrastive loss weight raised to λ = 0.5. |
| **FAS Inference (load best model - Model 1)** | Loads the trained Model 1 checkpoint and runs the passive scoring. |

Folder **HTML_VERSIONS_OF_NOTEBOOKS/** contains all the above jupyter notebooks in HTML version. This is added as a reference and backup.

Each notebook is **self-documenting**: it contains the full, ordered steps — data cataloguing, preprocessing, dataset/DataLoader construction, training, threshold selection, and evaluation — so the pipeline can be reproduced end-to-end once data access is available.

---

## The passive model (HybridPAD)

A short clip is reduced to **16 frames**; each face is detected and aligned, and **three complementary streams** are derived — **RGB** appearance, an **FFT frequency map**, and **optical flow** (**RAFT**). Each stream has its own CNN backbone (ConvNeXt-Tiny / EfficientNet-B0 / ResNet-34), the three are combined by an attention **stream-gate**, modelled over time by a BiLSTM + Transformer, and classified as LIVE / SPOOF. The decision threshold is chosen on validation by minimising **ACER**.

---

## The active challenge–response stage

Three actions are drawn at random (without repetition, in order) from a fixed library of **20 actions** — 6 head movements, 6 facial expressions, 8 hand gestures — giving **6,840** possible ordered sequences. The response is recorded, each action is detected with **MediaPipe**, and the sequence is accepted only if every action is **sustained** and performed **in order**. Identity continuity between the two stages is verified with **ArcFace** face embeddings.

---

## Data availability and ethics

**The datasets are not included in this repository.** They are **biometric data** and are subject to their respective licence agreements and data-protection requirements. The video datasets (OULU-NPU and SiW-Mv2) are **access-controlled** and are stored **only on the University computing cluster (RACC)**, where all training and evaluation were carried out. No raw face data, cached frames, or derived biometric artefacts are committed here.

> The Jupyter notebooks contain the detailed, ordered steps — including the exact preprocessing, dataset construction, splitting, training and evaluation procedure — so the work is **fully reproducible** by an authorised user who has obtained the datasets under licence and placed them at the paths configured in the notebooks. Anyone wishing to reproduce the results must request access to the datasets from their original providers and comply with the applicable licence and ethics conditions.

---

## Running the integrated (passive + active) web application on local PC (webcam enabled)

### Requirements

- Python 3.10+
- PyTorch, TorchVision (for the model and RAFT optical flow)
- ONNX Runtime (face detection)
- MediaPipe (active-stage face/hand detectors)
- InsightFace (`buffalo_l`, ArcFace — same-person check)
- FastAPI + Uvicorn (web server), OpenCV, NumPy, pandas, scikit-learn, Decord

Install into a virtual environment, for example:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt       # install the libraries with the help of the requirements.txt file provided
```

> A GPU is recommended for training, while the web application runs on CPU

### Run the main file

```bash
cd HybridPAD_app
python main_fas.py
```

Download the complete `Interactive_Web_Application` folder and navigate to it.
On the command-prompt, run `python main_fas.py` to start the application. Follow the instructions on the screen. The app loads the passive model from its saved checkpoint, and starts a FastAPI/Uvicorn server to serve the interface.

---

## System Architecture and Data Flow of the Web Application

The system follows a single continuous flow, with each stage responsible for one clear task. A session begins with a short setup step, followed by passive recording and spoof scoring, then moves to the active challenge–response stage, where the user performs a set of random actions. The outputs from both stages are then combined into a result, which is displayed on screen and recorded in the system logs.

This separation is also reflected in the code structure, where individual files independently handle frame measurements, challenge validation, session control, browser communication, and logging. As a result, thresholds can be adjusted or new challenge actions added without changing unrelated parts of the system, making the overall design easier to understand, maintain, and extend.

| Module | Responsibility |
|--------|---------------|
| `main_fas.py` | Starts the application |
| `CHL_server.py` | Runs the live session and connects every stage together |
| `CHL_config.py` | Holds shared settings in one place |
| `CHL_position_checker.py` | Checks whether the face is positioned correctly |
| `CHL_stability.py` | Confirms the position is held long enough before recording |
| `CHL_recording_monitor.py` | Watches for problems during the passive recording |
| `CHL_capture.py` | Saves frames and builds the recorded video |
| `CHL_scoring.py` | Runs the trained model and returns a live or spoof verdict |
| `CHL_identity.py` | Confirms the same person stays in front of the camera |
| `CHL_challenge.py` | Picks the random actions for the active stage |
| `CHL_face_blendshapes.py`, `CHL_hand_landmarker.py`, `CHL_primitives.py` | Recognise specific head, face, and hand actions |
| `CHL_challenge_validator.py` | Combines the per-frame detections and checks the actions were performed correctly and in order |
| `CHL_session_id.py` | Generates a unique identifier for each session |
| `CHL_session_log.py` | Records the outcome of every session |

---

## Reproducing the Training-Experiments

1. Obtain the datasets under licence and place them at the paths set in each notebook's configuration cell (on the computing cluster).
2. Open the relevant notebook (e.g. *Combined Pool — Model 1*) and run the cells in order. `NOTE: The 'Functions' folder and the module files inside it should be downloaded as well to run the notebooks`
3. Checkpoints and metrics are written to per-experiment output folders (e.g. `output_combinedData/model_1/best_model_combined.pth`, plus metrics `.json`, cached predictions and figures). Only the **best** checkpoint (lowest validation ACER) is kept per run.

---

## Model versions

| Version | Change |
|---------|--------|
| **Model 1** | Baseline combined-pool pipeline. |
| **Model 2** | Model 1 + revised data augmentation. |
| **Model 3** | Model 2 + supervised-contrastive weight λ = 0.5. |

Model 1 is the version loaded by the web application.

---

## Notes

- Full evaluation metrics (APCER, BPCER, ACER, and ROC-AUC) for every protocol and setting are produced inside the corresponding notebooks.
- An earlier Gradio prototype of the interface was replaced by the current FastAPI + JavaScript application for finer real-time control.

---

## Acknowledgements

Developed as part of an MSc Research Project (2025/26).