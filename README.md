# 🎙️ ResoNate AI - Speech Emotion Recognition

ResoNate AI is a full-stack Django application that uses a custom-built PyTorch Convolutional Neural Network (CNN) to recognize human emotions from speech. By analyzing acoustic features extracted from audio recordings, the system predicts one of four emotional states: **Happy, Sad, Angry,** or **Neutral**.

The application combines secure user authentication, real-time machine learning inference, and an interactive analytics dashboard to provide an end-to-end AI-powered experience.

# ✨ Features

### 🎤 Real-Time Emotion Detection
- Upload `.wav` audio files or record your voice directly from the browser.
- Receive instant emotion predictions powered by a trained CNN model.

### 🧠 Deep Learning Model
- Custom PyTorch CNN trained on the **RAVDESS** speech emotion dataset.
- Uses **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction.
- Predicts four emotion classes:
  - Happy
  - Sad
  - Angry
  - Neutral

### 📊 Analytics Dashboard
- Interactive dashboard built using **Chart.js**.
- Visualizes:
  - Emotion distribution over time
  - Prediction history
  - Average confidence scores

### 📈 Prediction Confidence
- Displays Softmax probability scores for every emotion class.
- Helps users understand the model's confidence instead of only showing the final prediction.

### 🔐 Secure Authentication
- Django authentication system
- OTP-based password reset via email
- Automatic inactive session timeout
- Secure authentication workflow

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Django 6.x |
| **Machine Learning** | PyTorch, Scikit-learn, python_speech_features |
| **Frontend** | HTML5, CSS3 (Glassmorphism UI), JavaScript |
| **Charts** | Chart.js |
| **Database** | SQLite (Development), PostgreSQL (Production Ready) |
| **Environment** | python-dotenv |

---

# 🚀 Local Installation

## 1. Clone the Repository

```bash
git clone <repository_url>
cd EmotionSenseAI
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
DEBUG=True

SECRET_KEY=your_secure_django_secret_key

EMAIL_HOST_USER=your_email@gmail.com

EMAIL_HOST_PASSWORD=your_16_digit_app_password

GMAIL_WEBHOOK_URL=your_google_apps_script_webhook_url
```

> **Production Note**
>
> Some free hosting platforms (such as Render) restrict SMTP access. To enable email functionality in production, configure a Google Apps Script webhook and provide its URL using `GMAIL_WEBHOOK_URL`.

---

## 5. Apply Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

---

## 6. Run the Development Server

```bash
python manage.py runserver
```

Visit:

```
http://127.0.0.1:8000/
```

---

# 🤖 Machine Learning Model

The repository includes the trained model file:

```
best_model.pth
```

The model is intentionally tracked in Git (approximately **500 KB**) so the project remains fully self-contained and can be deployed directly on platforms such as **Render**, **Railway**, or **Heroku** without requiring external model storage or download pipelines.

---

# 📌 Future Improvements

- Support additional emotion classes
- Real-time microphone streaming
- Transformer-based speech emotion models
- Audio waveform visualization
- User profile and report export
- REST API for external integrations

---

# 📄 License

© 2026 **ResoNate AI**. All Rights Reserved.