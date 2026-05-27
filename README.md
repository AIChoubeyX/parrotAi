# 🤖 Video Intelligence Agent

An AI-powered application that transcribes video content and builds a Retrieval-Augmented Generation (RAG) system to provide intelligent, context-aware responses about the video material.

## 📋 Features

- **Video Input Support**: Accept video URLs or live video streams
- **Transcription**: Automatically transcribe audio from videos using advanced speech-to-text
- **RAG System**: Build a knowledge base from transcribed content
- **Intelligent Agent**: Query the video content with natural language and receive contextual answers
- **Context-Aware Responses**: Leverages RAG to provide accurate information based on the video content

## 🎯 How It Works

1. **Input**: User provides a video URL or live stream link
2. **Processing**: Application downloads and processes the video
3. **Transcription**: Audio is transcribed to text using speech recognition
4. **Knowledge Base**: Transcribed content is indexed and stored for retrieval
5. **Agent Interface**: Users can ask questions about the video content
6. **Intelligent Responses**: The RAG agent retrieves relevant content and generates informed answers

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd parrotAi
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Project Structure

```
parrotAi/
├── utils/
│   └── audio_processor.py    # Audio processing utilities
├── downloads/                # Temporary storage for video files
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## 🛠️ Usage

(To be updated as the project develops)

## 📝 License

(Add your license here)

## 👤 Author

(Your name/organization)

---

**Note**: This is an early-stage project. Features and documentation will be updated as development progresses.
