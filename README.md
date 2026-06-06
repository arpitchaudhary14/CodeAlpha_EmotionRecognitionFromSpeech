# ResoNate AI - Speech Emotion Recognition
ResoNate AI is a production-ready Django application that utilizes a custom PyTorch Convolutional Neural Network (CNN) to detect emotional states (Happy, Sad, Angry, Neutral) directly from human speech audio patterns. 
The platform is designed to be fully self-contained, handling secure user authentication, real-time ML inference, and deep data analytics via an interactive Dashboard.
## Features
- **Real-Time Speech Analyzer**: Upload `.wav` files or record directly from the browser to receive instant AI predictions.
- **Deep Neural Network**: Powered by a PyTorch CNN trained on the RAVDESS dataset, leveraging Mel-Frequency Cepstral Coefficients (MFCC) for advanced acoustic feature extraction.
- **AI Insights Dashboard**: A dedicated analytics hub powered by Chart.js, visualizing your historical emotion distribution footprints and confidence averages.
- **Advanced Probability Breakdown**: Exposes the CNN's raw `Softmax` output layer, showing the exact percentage probabilities across all emotion classes instead of just the top prediction.
- **Enterprise Security**: 
  - Dynamic OTP email verification for password resets.
  - Rate-limited and secure `django.contrib.auth` integration.
  - Automated AFK (Away From Keyboard) session timeouts.
## Tech Stack
- **Backend Framework**: Django 6.x
- **Machine Learning**: PyTorch, python_speech_features, Scikit-learn
- **Frontend UI**: HTML5, Vanilla CSS (Glassmorphism UI), JavaScript, Chart.js
- **Database**: SQLite (Development) / PostgreSQL (Ready)
- **Environment Management**: python-dotenv
## Local Setup
### 1. Clone the Repository
```bash
git clone <repository_url>
cd EmotionSenseAI
```
### 2. Set up the Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables
Create a `.env` file in the root directory (using `.env.example` as a template):
```env
DEBUG=True
SECRET_KEY=your_secure_django_secret_key
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_16_digit_app_password
```
### 5. Run Migrations & Start Server
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
## ML Model Note
The repository contains `best_model.pth` (~500KB) tracked directly in Git to ensure the repository remains fully self-contained for deployment on platforms like Railway or Heroku without requiring a secondary S3 model downloading pipeline.
## License
&copy; 2026 ResoNate AI. All rights reserved.
