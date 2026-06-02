# ASL Sign Recognition

Echtzeiterkennung amerikanischer Gebärdensprache (ASL) per Webcam.  
Erkennt 34 statische Zeichen: Buchstaben A–Z (ohne J, Z) und Ziffern 0–9.

## Technologien

- Python 3.x
- TensorFlow / Keras
- MediaPipe (Hand-Landmark-Extraktion)
- OpenCV

## Funktionsweise

```
Webcam  ->  MediaPipe (21 Hand-Landmarks)  ->  MLP-Modell  ->  Zeichen + Konfidenz
```

Das Modell klassifiziert keine Rohbilder, sondern normalisierte Koordinaten der 21 Handpunkte (63 Features). Dadurch ist die Erkennung unabhängig von Beleuchtung, Hintergrund und Hautfarbe.

**Test-Accuracy auf unbekannter Person: 92,6 %**

## Setup

```bash
pip install -r requirements.txt
```

## Verwendung

```bash
# 1. Datensatz vorverarbeiten
python src/preprocess.py --mode landmarks

# 2. Modell trainieren
python src/train.py --mode landmarks

# 3. Echtzeit-Erkennung starten
python src/inference_realtime.py
```

## Datensatz

Der Datensatz *ASL Raw Images* wird nicht mitgeliefert.  
Erwartet wird folgende Struktur:

```
data/asl_dataset/
    A/   P1_A_1.jpg  ...  P10_A_1000.jpg
    B/   ...
    0/   P1_0_1.jpg  ...
```

10 Personen, 100 Bilder pro Zeichen und Person, Aufteilung: P1–P8 Training, P9 Validierung, P10 Test.

## Dokumentation

Siehe [`docs/gebaerdenerkennung_v1.md`](docs/gebaerdenerkennung_v1.md)
