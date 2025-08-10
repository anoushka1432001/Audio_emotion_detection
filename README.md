# Audio-based Emotion Prediction

## 📌 Project Description
This project aims to **predict emotions from audio data** using deep learning techniques. We utilize two datasets — **CREMA-D** and **TESS** — to train and evaluate a **Convolutional Neural Network (CNN)** for classifying emotions based on speech.

---

## ⚙️ Installation & Requirements

### 1. Clone the repository
```bash
git clone https://github.com/your-username/audio-emotion-prediction.git
cd audio-emotion-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> For detailed package information, refer to `requirements.txt`.

---

## 📂 Dataset Details

### **CREMA-D**
- **Source**: [CREMA-D GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
- **Description**: Audio-visual dataset of facial and vocal emotional expressions.
- **Format**: WAV
- **Emotions**: Anger, Disgust, Fear, Happy, Neutral, Sad
- **Sample Count**: 7,442 clips from 91 actors.

**Data Columns**:
- `Emotions`: Categorical (angry, disgust, fear, happy, sad, neutral)
- `Path`: File path to the audio file.

---

### **TESS**
- **Source**: [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)
- **Description**: 200 target words spoken in the phrase “Say the word ____” by 2 actresses (aged 26 and 64 years).
- **Format**: WAV
- **Emotions**: Anger, Disgust, Fear, Happy, Pleasant Surprise, Sad, Neutral
- **Sample Count**: 2,800 files (200 words × 7 emotions × 2 actresses).

**Data Columns**:
- `Emotions`: Categorical (angry, disgust, fear, happy, neutral, ps, sad)
- `Path`: File path to the audio file.

---

## 🔍 Key Findings (EDA Summary)
- Balanced distribution of emotions in both datasets.
- **MFCC features** show distinct patterns for different emotions.
- Spectrograms and waveforms provide **visual insights** into emotional characteristics of speech.

---

## 🚀 How to Use the Notebook

1. Open **`Audio-based-Emotion-Prediction.ipynb`** in Jupyter Notebook or Google Colab.
2. Ensure all required libraries are installed (see `requirements.txt`).
3. Run the notebook sequentially to:
   - Load and preprocess the data
   - Perform EDA
   - Train the CNN model
   - Evaluate model performance

---

## 📊 Results & Visualizations

- CNN model achieves **high accuracy** in emotion classification.
- Visualizations include:
  - Waveforms and spectrograms of sample audio files.
  - Training and validation accuracy/loss curves.
  - Confusion matrix of model predictions.

> For detailed results, refer to `reports/README.md`.

---

## 🔮 Future Enhancements
- Experiment with other architectures (LSTM, Transformer).
- Apply **transfer learning** from pre-trained audio models.
- Develop a **real-time emotion prediction** system.
- Explore **multi-modal emotion recognition** (audio + text + video).

---

## 🤝 Contributing
Contributions are welcome!  
Fork the repo, make changes, and submit a PR with your improvements.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 📦 Usage Example

```python
import pandas as pd
import librosa

# Load dataset
crema_df = pd.read_csv('path_to_crema_csv')
tess_df = pd.read_csv('path_to_tess_csv')

# Access audio
file_path = crema_df['Path'][0]
audio, sr = librosa.load(file_path)
```

**Data Preprocessing**:
- Extract features: MFCC, chroma, mel, contrast, tonnetz.
- Normalize features with `StandardScaler`.
- Encode labels with `LabelEncoder` or `OneHotEncoder`.

> **Note**: Ensure you have the necessary permissions and follow dataset usage guidelines.
