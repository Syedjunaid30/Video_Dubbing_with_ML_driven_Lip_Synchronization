# 🎬 AI Video Dubbing with ML-Driven Lip Synchronization

An intelligent video dubbing system powered by **deep learning** that enables users to dub a video in multiple languages with realistic **lip-sync accuracy**, powered by the [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) model.



---

## 📦 Download Project (Pre-packaged ZIP)

You can download the complete working project including models and scripts as a ZIP from Dropbox:

🔗 [Download ZIP from Dropbox](https://www.dropbox.com/scl/fi/aw2dtj3jk61b8d7x83i79/Video_Dubbing_with_ML_driven_Lip_Synchronization.zip?rlkey=npsmhiplekxapf687s5jzpb1a&st=n30s2y6s&dl=0)

---

## 📂 Folder Structure

```
ai-video-dubbing/
├── app.py # Flask backend for video upload and dubbing
├── templates/
│ └── index.html # Frontend interface
├── static/
│ ├── uploads/ # Uploaded videos
│ ├── outputs/ # Dubbed output videos
│ └── styles.css # Custom styling
├── wav2lip/
│ └── inference.py # Wav2Lip execution script
├── checkpoints/
│ └── wav2lip.pth # Pretrained Wav2Lip model (download manually)
├── requirements.txt # All dependencies
└── README.md # Project documentation
```

---

## 🚀 Features

- Translate spoken audio in a video into another language.
- Generate voice-over audio using realistic Text-to-Speech.
- Sync the new voice to the speaker’s lips using deep learning.
- Simple Flask-based web interface to upload and dub videos.

---

## 🌐 Supported Languages

- Hindi 🇮🇳  
- French 🇫🇷  
- German 🇩🇪  
- Spanish 🇪🇸  

---

## 🔧 Technologies Used

| Feature               | Tech/Tool Used                        |
|-----------------------|----------------------------------------|
| Backend               | Python Flask                          |
| Lip-Sync              | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| Audio Translation     | Google Translate API                  |
| Text-to-Speech        | gTTS or pyttsx3                       |
| Video Processing      | FFmpeg, OpenCV                        |
| Frontend              | HTML, CSS                             |

---

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Syedjunaid30/Video_Dubbing_with_ML_driven_Lip_Synchronization.git
cd Video_Dubbing_with_ML_driven_Lip_Synchronization
```
### 2. Install Required Packages

```bash
pip install -r requirements.txt
```
✅ Make sure you're using Python 3.7 or later

### 3. Download the Wav2Lip Pretrained Model
Visit the official Wav2Lip repo:
https://github.com/Rudrabha/Wav2Lip

Download the wav2lip.pth model and place it in the checkpoints/ directory.

---
## 💻 Run the Web App
```bash
python app.py
```
Open in browser at:
```bash
http://127.0.0.1:5000
```
## 🙏 Acknowledgments
🔗 Wav2Lip
Wav2Lip: Accurately Lip-syncing Videos In The Wild

Authors: Rudrabha Mukhopadhyay et al.

GitHub: https://github.com/Rudrabha/Wav2Lip

License: MIT

---
## 📸 Demo Screenrecoding 
 - its in the folder
---
## ✍️ Author
Syed Junaid

Sheikh Ameen

Saad Syed Kaleemulla

Zubair Abdul Aziz

---
## 📜 License

This project is released under the MIT License.
This license also respects and follows the license terms of the original Wav2Lip repository.
