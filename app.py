import os
import time
import asyncio
import nest_asyncio
import pymysql
import librosa
import numpy as np
import torch

pymysql.install_as_MySQLdb()

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from transformers import MarianMTModel, MarianTokenizer
import whisper
import subprocess
import edge_tts
from gtts import gTTS

# Fix asyncio issues on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nest_asyncio.apply()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/video_dubbing'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)

UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'
TEMP_FOLDER = 'temp/'
for folder in [UPLOAD_FOLDER, STATIC_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

whisper_model = whisper.load_model("large-v3", device=device)

translation_model_name = "Helsinki-NLP/opus-mt-en-hi"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

def translate_text(text):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = translation_model.generate(**inputs)
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

def detect_gender_from_face(image_path):
    try:
        from deepface import DeepFace
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='retinaface'
        )
        gender = analysis[0]['dominant_gender'].lower()
        print(f"[Gender Detection] Dominant gender: {gender}")
        return 'male' if 'male' in gender else 'female'
    except Exception as e:
        print("[ERROR] Gender detection failed:", e)
        return "female"


async def generate_tts(text, output_audio, retries=3, delay=5, gender="female"):
    voice = "hi-IN-MadhurNeural" if gender == "male" else "hi-IN-SwaraNeural"
    print(f"[TTS] Selected voice: {voice} for gender: {gender}")  # Debug line

    for attempt in range(retries):
        try:
            tts = edge_tts.Communicate(text, voice)
            await asyncio.wait_for(tts.save(output_audio), timeout=30)
            return
        except Exception as e:
            print(f"TTS Attempt {attempt+1} failed: {e}")
            time.sleep(delay)

    print("Falling back to gTTS...")
    tts = gTTS(text=text, lang="hi")
    tts.save(output_audio)


def lip_sync(video_path, audio_path, output_path):
    checkpoint = "wav2lip/checkpoints/wav2lip.pth"
    script = "wav2lip/inference.py"
    if not os.path.exists(checkpoint) or not os.path.exists(script):
        raise FileNotFoundError("Wav2Lip model or script missing.")
    
    cmd = (
        f"python {script} "
        f"--checkpoint_path \"{checkpoint}\" "
        f"--face \"{video_path}\" "
        f"--audio \"{audio_path}\" "
        f"--outfile \"{output_path}\" "
        f"--nosmooth "
        f"--pads 0 20 0 0 "
        f"--resize_factor 1"
    )
    subprocess.run(cmd, shell=True, check=True)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", languages={"hi": "Hindi"})

@app.route('/process', methods=['GET', 'POST'])
def process():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'GET':
        return redirect(url_for('dashboard'))
    video_file = request.files.get('video')
    if not video_file or video_file.filename == '':
        return "No video uploaded."
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)
    result = whisper_model.transcribe(video_path)
    original_text = result["text"]
    translated_text = translate_text(original_text)
    frame_temp = os.path.join(TEMP_FOLDER, "temp_frame_000.jpg")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vf", "select=eq(n\\,0)", "-vsync", "vfr", frame_temp])
    speaker_gender = detect_gender_from_face(frame_temp)
    print(f"Detected speaker gender: {speaker_gender}")
    audio_path = os.path.join(STATIC_FOLDER, "dubbed_audio.mp3")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_tts(translated_text, audio_path, gender=speaker_gender))
    output_path = os.path.join(STATIC_FOLDER, "dubbed_video.mp4")
    lip_sync(video_path, audio_path, output_path)
    return render_template("result.html",
                           original_text=original_text,
                           translated_text=translated_text,
                           video_file=url_for('static', filename='dubbed_video.mp4'))

@app.route('/download')
def download():
    path = os.path.join(STATIC_FOLDER, "dubbed_video.mp4")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="dubbed_video.mp4")
    return "File not found."

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists.')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pword = request.form['password']
        user = User.query.filter_by(username=uname).first()
        if user and bcrypt.check_password_hash(user.password, pword):
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
