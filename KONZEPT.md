# ASL Sign Recognition — Technisches Konzept

## 1. Architektur-Entscheidung: MediaPipe Landmarks vs. CNN

### Vergleich

| Kriterium | MediaPipe + MLP | Klassisches CNN |
|---|---|---|
| **Eingabedaten** | 63 Landmarks (x/y/z × 21) | 64×64 RGB Bild |
| **Lichtempfindlichkeit** | ✅ Invariant | ⚠️ Stark abhängig |
| **Hintergrundempfindlichkeit** | ✅ Vollständig invariant | ⚠️ Hintergrund stört |
| **Hautfarben-Robustheit** | ✅ Keine Textur nötig | ⚠️ Oft überangepasst |
| **Modellgröße** | ~200 KB | ~5–50 MB |
| **Trainingszeit** | Minuten | Stunden |
| **Echtzeit-FPS** | 45–60+ FPS | 20–40 FPS |
| **Buchstaben J/Z** | ❌ Motion nötig | ❌ Motion nötig |
| **Genauigkeit (gut trainiert)** | 95–99% | 92–97% |

### Empfehlung: **MediaPipe + MLP**

Für den Einsatz an der Webcam ist der Landmark-Ansatz klar überlegen:
- Die 21 Hand-Keypoints sind strukturierte, semantisch sinnvolle Features
- Training konvergiert zuverlässig mit kleinen Datensätzen
- Kein Overfitting auf Lichtverhältnisse oder Hintergründe
- Deutlich geringere Latenz → flüssige Echtzeit-Erkennung

Das CNN bleibt als Fallback implementiert und eignet sich, wenn Textur-Information
entscheidend ist (z. B. feine Handstellungen die Landmarks nicht unterscheiden).

---

## 2. Datensatz-Layout (erwartet)

```
data/asl_dataset/
    A/   P1_A_1.jpg  P1_A_2.jpg ...
    B/   ...
    1/   P1_1_1.jpg  P1_1_2.jpg ...
    2/   ...
    ...
```

Dateiname-Format spielt keine Rolle — alle `.jpg`/`.png`/`.jpeg` im Ordner werden geladen.

**Hinweis:** Die Buchstaben J und Z werden standardmäßig ausgelassen, da sie
Bewegungssequenzen erfordern (kein statisches Einzelbild).

---

## 3. Workflow

```
1. Preprocessing
   python src/preprocess.py --mode landmarks
   -> data/dataset_landmarks.npz

2. Training
   python src/train.py --mode landmarks
   -> models/asl_landmarks_best.keras
   -> models/label_map.npy

3. Evaluation
   python src/evaluate.py --mode landmarks
   -> figures/confusion_matrix_landmarks.png

4. Echtzeit-Inferenz
   python src/inference_realtime.py
   -> Webcam-Fenster mit Live-Overlay
```

---

## 4. Overfitting-Prävention & Robustheit

### Daten-Augmentation (CNN-Mode)
Eingebaut in `model_cnn.py` via Keras Preprocessing-Layers:
- `RandomFlip`, `RandomRotation(±8°)`, `RandomZoom(±10%)`
- `RandomBrightness(±15%)`, `RandomContrast(±15%)`

### Regularisierung (beide Modelle)
- **L2-Regularisierung** auf alle Dense/Conv-Schichten (λ = 1e-4)
- **Dropout** nach jedem Block (Rate 0.3–0.4)
- **Batch Normalisation** nach jeder Aktivierung
- **Early Stopping** (Patience=12 Epochen, überwacht val_accuracy)
- **ReduceLROnPlateau** (Faktor 0.5 nach 5 Epochen ohne Verbesserung)

### Robustheit bei unterschiedlichen Lichtverhältnissen
1. **Landmark-Ansatz wählen** — eliminiert das Problem strukturell
2. **Histogramm-Gleichverteilung** im CNN-Preprocessing für schwierige Bilder:
   ```python
   img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
   ```
3. **Training-Daten diversifizieren**: Bilder bei verschiedenen Lichtverhältnissen aufnehmen
4. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) für Webcam-Frames:
   ```python
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   enhanced = clahe.apply(gray_frame)
   ```
5. **Mediapipe-Confidence-Schwellwert** erhöhen (`min_detection_confidence=0.7`)
   bei guten Lichtverhältnissen, senken bei schlechten

### Verbesserung der Erkennungsrate
- **Temporales Smoothing**: Letzten N Frames mitteln (implementiert via `SmoothPredictor`)
- **Confidence-Schwellwert**: Nur Vorhersagen > 0.70 anzeigen (verhindert Fehlalarme)
- **Transferlernen** (CNN): EfficientNetB0 oder MobileNetV3 als Backbone (wenn
  der eigene Datensatz < 5.000 Bilder je Klasse hat)
- **Konfusionsklassen gezielt augmentieren**: Nach Auswertung der Confusion Matrix
  ähnliche Zeichen (z. B. M/N/S) mit mehr Variations-Daten nachtrainieren

---

## 5. Dateistruktur

```
ASL_SignRecognition/
├── data/
│   ├── asl_raw_images/          <- Deinen Datensatz hier ablegen
│   ├── dataset_landmarks.npz    <- generiert von preprocess.py
│   └── dataset_cnn.npz          <- generiert von preprocess.py
├── models/
│   ├── asl_landmarks_best.keras
│   ├── asl_cnn_best.keras
│   └── label_map.npy
├── figures/
│   └── confusion_matrix_landmarks.png
├── src/
│   ├── preprocess.py            <- Datenaufbereitung
│   ├── model_landmarks.py       <- MLP-Architektur
│   ├── model_cnn.py             <- CNN-Architektur
│   ├── train.py                 <- Training-Loop
│   ├── evaluate.py              <- Confusion Matrix + Report
│   └── inference_realtime.py    <- Webcam-Echtzeit-Erkennung
├── requirements.txt
└── KONZEPT.md                   <- dieses Dokument
```

---

## 6. Installation

```bash
# Virtuelle Umgebung empfohlen
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

Für GPU-Training (optional):
```bash
pip install tensorflow[and-cuda]   # TF 2.13+
```
