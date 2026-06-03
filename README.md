# Cardgame_Bluffing_Detection

# ComfyUI Poker AI Nodes

A ComfyUI custom node package for real-time poker tell analysis using facial emotion recognition, skeleton-based action recognition, XGBoost bluff detection, and visual overlay rendering.

---

## ComfyUI-DDAMFN

Real-time facial emotion recognition using the **DDAM-FN** (Dual-Direction Attention Map with Feature Network) architecture.

### Node: `DDAMFN Emotion Recognizer`

**Inputs**
| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Video frames (batch) |
| `model_path` | LIST | `.pth` weight file from `pretrained/` |
| `frame_interval` | INT | Process every N-th frame (default: 4) |

**Outputs**
| Name | Type | Description |
|------|------|-------------|
| `dominant_emotion` | STRING | Per-frame dominant emotion log |
| `full_analysis` | STRING | Per-frame probability breakdown for all 7 emotions |

**Emotion classes (RAF-DB):** Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral

### Installation

```
custom_nodes/
└── ComfyUI-DDAMFN/
    ├── networks/
    │   ├── DDAM.py
    │   └── MixedFeatureNet.py
    ├── pretrained/       ← place your .pth weights here
    ├── ddamfn_node.py
    └── __init__.py
```

**Requirements:** `torch` `torchvision` `numpy` `Pillow`

---

## ComfyUI-PokerActionGCN

Skeleton-based action recognition using **CTR-GCN**, combined with XGBoost bluff detection, LLM prompt fusion, and visual overlay rendering.

### Node: `CTRGCN Action Recognizer`

Extracts body skeletons with YOLOv8-Pose and classifies actions using a spatio-temporal graph convolutional network.

**Inputs**
| Name | Type | Description |
|------|------|-------------|
| `image_frames` | IMAGE | Video frames (batch) |
| `model_path` | LIST | `.pth` weight file from `pretrained/` |
| `confidence_threshold` | FLOAT | YOLOv8 keypoint confidence threshold (default: 0.5) |
| `frame_interval` | INT | Process every N-th frame (default: 4) |

**Outputs**
| Name | Type | Description |
|------|------|-------------|
| `dominant_action` | STRING | Per-frame dominant action log |
| `action_full_analysis` | STRING | Per-frame probability breakdown for all 5 actions |

**Action classes:** Stable/Calm, Hand Fidgeting, Touching Face/Neck, Posture Shifting, Freezing/Rigid

---

### Node: `Poker ML Classifier`

Fuses facial emotion data, body action data, and game log into an XGBoost bluff probability prediction with a SHAP-style explanation report.

**Inputs:** `game_log`, `facial_tell`, `action_tell` (STRING)

**Outputs:** `bluff_decision`, `ml_shap_report` (STRING)

---

### Node: `Poker Prompt Fusion`

Formats all analysis data into a structured LLM prompt (Llama-3 / ChatML format) for a poker AI assistant to give strategic advice.

**Inputs:** `game_log`, `facial_tell`, `action_tell` (STRING)

**Output:** `master_prompt` (STRING)

---

### Node: `Poker Visual Overlay`

Renders real-time emotion and action labels on each video frame, with a full result report on the final frame.

**Inputs:** `image_frames`, `emotion_text`, `action_text`, `bluff_result`, `ml_shap_report`

**Output:** `IMAGE`

### Installation

```
custom_nodes/
└── ComfyUI-PokerActionGCN/
    ├── networks/
    │   └── ctrgcn.py
    ├── pretrained/       ← place your .pth weights here
    ├── ctrgcn_node.py
    ├── overlay_nodes.py
    └── __init__.py
```

**Requirements:** `torch` `numpy` `Pillow` `ultralytics` `pandas` `xgboost`
