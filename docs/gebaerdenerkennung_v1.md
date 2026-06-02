# Gebärdenerkennung V1 – Technische Dokumentation

**Projektbezeichnung:** ASL Sign Recognition  
**Version:** 1.0  
**Stand:** Juni 2026  
**Technologiestack:** Python 3.x, TensorFlow 2.20, MediaPipe 0.10, OpenCV

---

## 1. Zielsetzung

Ziel der Version 1 ist die Echtzeiterkennung statischer Handzeichen der amerikanischen Gebärdensprache (American Sign Language, ASL) mithilfe einer Webcam. Das System erkennt die Buchstaben A bis Z sowie die Ziffern 0 bis 9 und gibt das erkannte Zeichen gemeinsam mit einem Konfidenzwert visuell im Kamerabild aus. Die Verarbeitung erfolgt lokal auf der Hardware des Benutzers ohne Internetverbindung.

---

## 2. Systemarchitektur

Die Verarbeitungspipeline gliedert sich in drei aufeinanderfolgende Stufen:

```
Webcam-Frame  ->  Hand-Landmark-Extraktion  ->  Klassifikationsmodell  ->  Ausgabe
```

### 2.1 Hand-Landmark-Extraktion

Zur Erkennung der Hand im Kamerabild wird die Bibliothek MediaPipe (Google) in der Tasks-API-Variante (ab Version 0.10.13) eingesetzt. MediaPipe lokalisiert 21 anatomische Punkte der Hand (Handgelenk, Fingerknöchel, Fingerkuppen) und liefert deren normierte Koordinaten im Format (x, y, z) relativ zur Bildgrösse.

Diese 21 Punkte werden anschliessend in der Funktion `landmarks_to_features` (Datei `src/mediapipe_utils.py`) normalisiert:

1. Das Handgelenk (Landmark 0) wird als Ursprung gesetzt, alle übrigen Koordinaten werden relativ dazu berechnet.
2. Die gesamte Koordinatenmatrix wird durch ihren maximalen Absolutwert dividiert, sodass alle Werte im Intervall [-1, 1] liegen.
3. Das Ergebnis ist ein eindimensionaler Vektor der Länge 63 (21 Punkte * 3 Koordinaten).

Diese Normalisierung stellt sicher, dass Unterschiede in Handgrösse, Position im Bild und Abstand zur Kamera keinen Einfluss auf die Klassifikation haben.

### 2.2 Klassifikationsmodell

Als Klassifikationsmodell wird ein mehrschichtiges neuronales Netz (Multi-Layer Perceptron, MLP) verwendet. Die Entscheidung gegen ein Convolutional Neural Network (CNN) auf dem Rohbild wurde bewusst getroffen, da der Landmark-Ansatz gegenüber Beleuchtung, Hintergrund und Hautfarbe invariant ist und gleichzeitig eine deutlich geringere Modellkomplexität aufweist.

#### Modellarchitektur

| Schicht | Typ | Ausgabedimension | Parameter |
|---|---|---|---|
| Eingabe | Input | 63 | 0 |
| 1 | GaussianNoise (sigma=0.02) | 63 | 0 |
| 2 | BatchNormalization | 63 | 252 |
| 3 | Dense | 512 | 32.256 |
| 4 | BatchNormalization + ReLU + Dropout (0.5) | 512 | 2.048 |
| 5 | Dense | 256 | 131.072 |
| 6 | BatchNormalization + ReLU + Dropout (0.5) | 256 | 1.024 |
| 7 | Dense | 128 | 32.768 |
| 8 | BatchNormalization + ReLU + Dropout (0.5) | 128 | 512 |
| 9 | Dense + Softmax | 34 | 4.386 |

**Gesamt: 204.318 Parameter (ca. 798 KB)**

#### Regularisierungsmassnamen

- **GaussianNoise (sigma=0.02):** Addiert während des Trainings ein kleines normalverteiltes Rauschen auf die Eingabe-Landmarks. Dies simuliert leichte Handstellungsunterschiede zwischen verschiedenen Personen und verbessert die Generalisierung.
- **Dropout (Rate 0.5):** Deaktiviert in jedem Trainingsschritt zufällig 50 % der Neuronen. Das Netz kann sich dadurch nicht auf einzelne Neuronen verlassen und lernt robustere Repräsentationen.
- **BatchNormalization:** Normalisiert die Aktivierungen jeder Schicht. Dies stabilisiert den Trainingsprozess und erlaubt höhere Lernraten.
- **L2-Regularisierung (lambda=1e-4):** Bestraft grosse Gewichte und verhindert Overfitting.
- **Label Smoothing (0.1):** Verhindert, dass das Modell übermässig hohe Konfidenzwerte lernt, was die Kalibrierung und Generalisierung verbessert.

### 2.3 Temporale Glättung

Um Flackern in der Echtzeiterkennung zu reduzieren, wird über die letzten 7 Frames gemittelt (`SmoothPredictor` in `src/inference_realtime.py`). Ein Zeichen wird nur angezeigt, wenn der gemittelte Konfidenzwert den Schwellenwert von 0.70 überschreitet.

---

## 3. Datensatz

### 3.1 Beschreibung

Verwendet wird der Datensatz *ASL Raw Images*. Er enthält Aufnahmen von 10 Personen (P1–P10), je 100 Bilder pro Zeichen und Person.

**Dateikonvention:** `P{Person}_{Zeichen}_{Index}.jpg`  
Beispiel: `P3_A_247.jpg` – Person 3, Zeichen A, Bild 247

**Verzeichnisstruktur:**
```
data/asl_dataset/
    A/    P1_A_1.jpg  ...  P10_A_1000.jpg
    B/    ...
    0/    P1_0_1.jpg  ...  P10_0_1000.jpg
    ...
```

### 3.2 Erkannte Klassen

Version 1 unterstützt 34 statische Klassen:

- **Buchstaben (24):** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- **Ziffern (10):** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Die Buchstaben J und Z sind in Version 1 nicht enthalten, da deren Ausführung eine Bewegungssequenz erfordert und mit einem einzelnen statischen Frame nicht zuverlässig klassifiziert werden kann.

### 3.3 Datenaufteilung

Die Aufteilung erfolgt personenbasiert, nicht zufällig. Dieser Ansatz verhindert Data Leakage, da andernfalls Bilder derselben Person in Trainings- und Testmenge auftreten würden, was die gemessene Testgenauigkeit künstlich erhöhen würde.

| Menge | Personen | Bilder pro Klasse | Gesamt |
|---|---|---|---|
| Training | P1 – P8 | 800 | 27.171 |
| Validierung | P9 | 100 | 3.400 |
| Test | P10 | 100 | 3.400 |

Von den 34.000 Bildern konnten 29 (0,085 %) nicht verarbeitet werden, da MediaPipe in diesen Aufnahmen keine Hand erkannt hat. Diese wurden übersprungen.

---

## 4. Training

### 4.1 Hyperparameter

| Parameter | Wert |
|---|---|
| Optimizer | Adam |
| Initiale Lernrate | 0.001 |
| Batch-Grösse | 64 |
| Maximale Epochen | 80 |
| Early Stopping | val_loss, Patience 12 |
| Lernraten-Reduktion | ReduceLROnPlateau, Faktor 0.5, Patience 5 |
| Verlustfunktion | Categorical Crossentropy mit Label Smoothing 0.1 |

### 4.2 Trainingsablauf

Das Training wurde auf einer CPU ohne GPU-Beschleunigung durchgeführt. Early Stopping beendete das Training nach Epoche 48, da sich der Validierungsverlust nicht weiter verbesserte. Das beste Modell (niedrigster val_loss) wurde automatisch gespeichert.

---

## 5. Ergebnisse

| Metrik | Wert |
|---|---|
| Trainings-Accuracy | ca. 99 % |
| Validierungs-Accuracy (P9) | ca. 97 % |
| Test-Accuracy (P10, unbekannte Person) | **92,6 %** |

Die Test-Accuracy von 92,6 % wurde auf einer Person gemessen, die dem Modell während des Trainings und der Validierung vollständig unbekannt war. Dieser Wert gibt somit eine realistische Einschätzung der zu erwartenden Erkennungsrate bei neuen Benutzern.

---

## 6. Projektstruktur

```
ASL_SignRecognition/
|
+-- data/
|   +-- asl_dataset/              Rohdaten (A-Z, 0-9)
|   +-- dataset_landmarks.npz     Vorverarbeitete Features (generiert)
|
+-- models/
|   +-- asl_landmarks_best.keras  Trainiertes Modell
|   +-- label_map.npy             Zuordnung Index zu Klasse
|   +-- hand_landmarker.task      MediaPipe-Modell (automatisch geladen)
|
+-- src/
|   +-- preprocess.py             Datenvorverarbeitung und Feature-Extraktion
|   +-- mediapipe_utils.py        MediaPipe Tasks API, Landmark-Normalisierung
|   +-- model_landmarks.py        MLP-Architektur und Kompilierung
|   +-- train.py                  Trainingsloop mit Callbacks
|   +-- evaluate.py               Konfusionsmatrix und Klassifikationsbericht
|   +-- inference_realtime.py     Echtzeit-Inferenz via Webcam
|
+-- docs/
|   +-- gebaerdenerkennung_v1.md  Diese Dokumentation
|
+-- requirements.txt              Python-Abhängigkeiten
```

---

## 7. Verwendung

### 7.1 Installation

```bash
pip install -r requirements.txt
```

### 7.2 Datenvorverarbeitung

```bash
python src/preprocess.py --mode landmarks
```

Beim ersten Aufruf wird das MediaPipe-Handmodell automatisch heruntergeladen (ca. 25 MB). Die verarbeiteten Features werden unter `data/dataset_landmarks.npz` gespeichert.

### 7.3 Training

```bash
python src/train.py --mode landmarks
```

### 7.4 Echtzeit-Inferenz

```bash
python src/inference_realtime.py
```

Das Webcam-Fenster wird geöffnet. Das erkannte Zeichen und der Konfidenzwert werden live eingeblendet. Die Anwendung wird durch Drücken der Taste Q beendet.

Optionale Parameter:
```bash
python src/inference_realtime.py --cam 1   # zweite Webcam verwenden
```

---

## 8. Bekannte Einschränkungen

| Einschränkung | Beschreibung |
|---|---|
| Fehlende Zeichen | J und Z sind nicht enthalten (Bewegungssequenz erforderlich) |
| Einzelne Hand | Simultane Erkennung zweier Hände wird nicht unterstützt |
| Keine Wortebene | Es werden einzelne Zeichen erkannt, keine Wörter oder Sätze |
| Lichtverhältnisse | Bei sehr schlechter Beleuchtung kann MediaPipe die Hand nicht lokalisieren |
| Statische Zeichen | Das Modell klassifiziert Einzelframes, keine zeitlichen Sequenzen |

---

## 9. Ausblick (V2)

- Erkennung von J und Z über Bewegungssequenzen (z.B. LSTM)
- Zusammensetzen erkannter Buchstaben zu Wörtern mit Pause-Erkennung
- Text-to-Speech-Ausgabe des erkannten Textes
- Erweiterung auf beide Hände
