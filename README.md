# Automated Captioning with Whisper

This project provides an automated captioning system using OpenAI's Whisper for speech-to-text and script matching functionality.

## **Features**
- Real-time speech recognition using Whisper.
- Matching recognized text with a pre-defined script.
- Supports audio input via microphone.

---

## **Requirements**

- Python 3.8 or higher
- pip (Python package installer)
- ffmpeg (installed locally)

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/ohjinyxung/Automated-Captioning.git
   cd Automated-Captioning

2. Create a virtual environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install required dependencies:
    ```bash
    pip install -r requirements.txt

4. Install ffmpeg:
On macOS:
    ```bash
    brew install ffmpeg
On Ubuntu:
    ```
     sudo apt update
     sudo apt install ffmpeg
On Windows:
     ```
     * Download and install ffmpeg from https://ffmpeg.org/download.html.
     * Add ffmpeg to the system PATH.

Usage
Ensure your microphone is connected and working.

Run the script to start real-time captioning:
    ```bash
    python transcribe_match.py
Speak into the microphone. The recognized text will be matched with the predefined script (script.json) and displayed.

Project Files
* transcribe_match.py: Main script for speech-to-text and script matching.
* requirements.txt: List of dependencies.
* script.json: Example script data for matching.

Known Issues
PyAudio installation on macOS:
    If you encounter issues installing PyAudio, ensure portaudio is installed:
        ```bash
        brew install portaudio
        pip install pyaudio
Missing ffmpeg:
    Ensure ffmpeg is installed and accessible in your system PATH.
