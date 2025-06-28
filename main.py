import os
import time
import asyncio
import nest_asyncio
import pymysql
import librosa
import numpy as np
pymysql.install_as_MySQLdb()

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from transformers import MarianMTModel, MarianTokenizer
import whisper
import subprocess
import edge_tts
import aiohttp

# Fix for asyncio on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Async fix
nest_asyncio.apply()

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# SQLAlchemy config
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/video_debugging'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Bcrypt
bcrypt = Bcrypt(app)

# Directories
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'
TEMP_FOLDER = 'temp/'

for folder in [UPLOAD_FOLDER, STATIC_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# ML Models
whisper_model = whisper.load_model("large-v3")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

# Translation
def translate_text(text, target_lang="hi"):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = translation_model.generate(**inputs)
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

def detect_gender(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 75 < pitch < 300:  # Only valid human pitch
                pitch_values.append(pitch)

        if not pitch_values:
            print("[WARN] No valid pitch values found.")
            return "female"  # fallback

        avg_pitch = np.mean(pitch_values)
        print(f"[DEBUG] Filtered Avg Pitch: {avg_pitch:.2f} Hz")

        if avg_pitch < 165:
            return "male"
        else:
            return "female"

    except Exception as e:
        print("[ERROR] Gender detection failed:", e)
        return "female"


# TTS
async def generate_tts(text, output_audio, retries=3, delay=5, voice_lang="hi", gender="female"):
    voice_map = {
        "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
        "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
        "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
        "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"}
    }
    voice = voice_map.get(voice_lang, {}).get(gender, "hi-IN-SwaraNeural")

    for attempt in range(retries):
        try:
            tts = edge_tts.Communicate(text, voice)
            await tts.save(output_audio)
            return
        except Exception as e:
            print(f"TTS Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

# Lip sync
def lip_sync(video_path, audio_path, output_path):
    checkpoint = "wav2lip/checkpoints/wav2lip.pth"
    script = "wav2lip/inference.py"
    if not os.path.exists(checkpoint) or not os.path.exists(script):
        raise FileNotFoundError("Wav2Lip model or script missing.")
    cmd = f"python {script} --checkpoint_path {checkpoint} --face \"{video_path}\" --audio \"{audio_path}\" --outfile \"{output_path}\""
    subprocess.run(cmd, shell=True, check=True)

# Home
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    languages = {
        "hi": "Hindi",
        "fr": "French",
        "es": "Spanish",
        "de": "German"
    }

    return render_template("index.html", languages=languages)

# Process
@app.route('/process', methods=['GET', 'POST'])
def process():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return redirect(url_for('index'))

    video_file = request.files.get('video')
    if not video_file or video_file.filename == '':
        return "No video uploaded."

    target_lang = request.form.get('language', 'hi')

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # STT
    result = whisper_model.transcribe(video_path)
    original_text = result["text"]

    # Translate
    translated_text = translate_text(original_text, target_lang=target_lang)

    # Extract audio for gender detection
    audio_temp = os.path.join(TEMP_FOLDER, "temp_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_temp
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Detect gender based on audio
    speaker_gender = detect_gender(audio_temp)

    # Debugging: Print the detected gender
    print(f"Detected speaker gender: {speaker_gender}")

    # TTS
    audio_path = os.path.join(STATIC_FOLDER, "dubbed_audio.mp3")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_tts(translated_text, audio_path, voice_lang=target_lang, gender=speaker_gender))

    # Lip sync
    output_path = os.path.join(STATIC_FOLDER, "dubbed_video.mp4")
    lip_sync(video_path, audio_path, output_path)

    return render_template("result.html",
                           original_text=original_text,
                           translated_text=translated_text,
                           video_file=url_for('static', filename='dubbed_video.mp4'))

# Download
@app.route('/download')
def download():
    path = os.path.join(STATIC_FOLDER, "dubbed_video.mp4")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="dubbed_video.mp4")
    else:
        return "File not found."

# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pword = request.form['password']
        if User.query.filter_by(username=uname).first():
            flash("Username already taken.")
            return redirect(url_for('register'))

        hashed_pw = bcrypt.generate_password_hash(pword).decode('utf-8')
        user = User(username=uname, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful.")
        return redirect(url_for('login'))

    return render_template("register.html")

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pword = request.form['password']
        user = User.query.filter_by(username=uname).first()
        if user and bcrypt.check_password_hash(user.password, pword):
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))

    return render_template("login.html")

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out.")
    return redirect(url_for('login'))

# Run app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
