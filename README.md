# Audio_emotion_detection
Audio emotion detection Project Description
This project aims to predict emotions from audio data using deep learning techniques. We utilize two datasets, CREMA-D and TESS, to train and evaluate a Convolutional Neural Network (CNN) for classifying emotions based on speech.

Installation & Requirements
To set up the project environment:

Clone the repository:

git clone https://github.com/your-username/audio-emotion-prediction.git
cd audio-emotion-prediction
Install the required packages:

pip install -r requirements.txt
For detailed package information, refer to requirements.txt.

Dataset Details
CREMA-D: 7,442 audio samples of emotional speech from 91 actors
TESS: 2,800 audio samples of emotional speech from 2 actresses
Both datasets contain recordings of various emotions including anger, disgust, fear, happiness, sadness, and neutral states.

For more information, see data/README.md.

Key Findings (EDA Summary)
Balanced distribution of emotions in both datasets
MFCC features show distinct patterns for different emotions
Spectrograms and waveforms provide visual insights into emotional characteristics of speech
How to Use the Notebook
Open Audio-based-Emotion-Prediction.ipynb in Jupyter Notebook or Google Colab
Ensure all required libraries are installed (see requirements.txt)
Run cells sequentially to:
Load and preprocess the data
Perform exploratory data analysis
Train the CNN model
Evaluate model performance
Results & Visualizations
The CNN model achieves high accuracy in emotion classification
Visualizations include:
Waveforms and spectrograms of sample audio files
Training and validation accuracy/loss curves
Confusion matrix of model predictions
For detailed results, refer to reports/README.md.

Future Enhancements
Experiment with other deep learning architectures (LSTM, Transformer)
Incorporate transfer learning from pre-trained audio models
Develop a real-time emotion prediction system
Explore multi-modal emotion recognition (combining audio with text or video)
🤝 Contributing
Feel free to fork this repository and submit pull requests if you have improvements or new insights to add.
Audio-based Emotion Prediction Dataset
Dataset Overview
This project uses two datasets for audio-based emotion prediction:

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
TESS (Toronto Emotional Speech Set)
CREMA-D Dataset
Source: https://github.com/CheyneyComputerScience/CREMA-D
Description: Audio-visual dataset of facial and vocal emotional expressions
File Format: WAV
Emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
Sample Count: 7,442 clips from 91 actors
Data Columns
Emotions: Categorical (angry, disgust, fear, happy, sad, neutral)
Path: String (file path to the audio file)
TESS Dataset
Source: https://tspace.library.utoronto.ca/handle/1807/24487
Description: Set of 200 target words spoken in the carrier phrase "Say the word _____" by two actresses (aged 26 and 64 years)
File Format: WAV
Emotions: Anger, Disgust, Fear, Happy, Pleasant Surprise, Sad, Neutral
Sample Count: 2,800 files (200 target words × 7 emotions × 2 actresses)
Data Columns
Emotions: Categorical (angry, disgust, fear, happy, neutral, ps, sad)
Path: String (file path to the audio file)
Usage
Load the datasets using pandas:

import pandas as pd

crema_df = pd.read_csv('path_to_crema_csv')
tess_df = pd.read_csv('path_to_tess_csv')
Access audio files using the 'Path' column in each dataframe.

Use librosa to load and process audio files:

import librosa

audio, sr = librosa.load(file_path)
Data Preprocessing
Extract features like MFCC, chroma, mel, contrast, and tonnetz from audio files.
Normalize features using StandardScaler.
Encode emotion labels using LabelEncoder or OneHotEncoder.
Note
Ensure you have the necessary permissions and follow the dataset's usage guidelines when using these datasets for your project.


License
This project is open-source and available under the MIT License.
