# AI-Based Deepfake Detection System

A web-based application for detecting manipulated (deepfake) videos using deep learning and computer vision techniques. The system allows users to upload a video, analyzes its authenticity, and provides a detailed report indicating whether the content is genuine or manipulated.

This project is being developed as a major project for the Bachelor of Engineering in Software Engineering under Pokhara University.

---

## Project Objective

The main objective of this project is to develop an accessible platform that can assist users in identifying deepfake videos by combining visual and audio analysis techniques. The system is designed to provide meaningful results in a simple and user-friendly interface.

---

## Features

- User registration and login
- Secure video upload
- Video frame extraction
- Face detection and preprocessing
- AI-based deepfake prediction
- Audio analysis
- Confidence score generation
- Downloadable analysis report
- User dashboard with analysis history

---

## Technology Stack

### Frontend

- React
- Vite
- Tailwind CSS
- Axios

### Backend

- Django
- Django REST Framework

### Artificial Intelligence

- PyTorch
- OpenCV
- Librosa
- NumPy
- Pandas

### Database

- PostgreSQL

### Development Tools

- Git
- GitHub
- Visual Studio Code

---

## Project Structure

```
AI-Deepfake-Detection/
│
├── frontend/
├── backend/
├── ai-engine/
├── docs/
├── reports/
├── datasets/
├── README.md
└── .gitignore
```

---

## Workflow

1. User uploads a video.
2. The backend extracts video frames and audio.
3. Faces are detected from extracted frames.
4. The trained deep learning model analyzes the frames.
5. Audio features are analyzed separately.
6. The results are combined to generate a final prediction.
7. A report containing the confidence score and analysis is displayed to the user.

---

## Current Progress

- [x] Project initialization
- [x] Git repository setup
- [x] Backend initialization
- [x] Frontend initialization
- [ ] Authentication module
- [ ] Video upload module
- [ ] AI model integration
- [ ] Audio analysis
- [ ] Report generation
- [ ] Deployment

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Byteforgeee/AI-Deepfake-Detection.git
```

### Backend

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

python manage.py runserver
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## Future Improvements

- Live webcam detection
- Browser extension support
- Mobile application
- Real-time streaming analysis
- Improved model accuracy using larger datasets
- Explainable AI visualization

---

## Team Members

- Sunil Giri
- Ankit Katwal
- Bhupesh Bhatt
- Lalit Nath


---

## Supervisor

Mr. Amit Srivastava

Department of Software Engineering

---

## License

This project is developed for academic and research purposes only.

## Model Training
See `model-training/README.md` for the deepfake detection model training pipeline (EfficientNet + LSTM, trained on Celeb-DF-v2, 96.94% test accuracy).

## Model Training
See `model-training/README.md` for the deepfake detection model training pipeline (EfficientNet + LSTM, trained on Celeb-DF-v2, 96.94% test accuracy).
