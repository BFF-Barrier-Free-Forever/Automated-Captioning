import whisper
import pyaudio
import numpy as np
import json
import os
from tempfile import NamedTemporaryFile
import difflib
import wave

# 대본 매칭 함수
def match_script(transcribed_text, script, match_length=3):
    """
    Whisper 전사 결과를 대본 데이터와 매칭합니다.
    Parameters:
    - transcribed_text (str): Whisper로부터 얻은 텍스트
    - script (list): 대본 데이터 (JSON 형태로 로드됨)
    - match_length (int): 매칭할 음절 길이
    Returns:
    - dict: 매칭된 대본 줄 또는 매칭 실패 메시지
    """
    transcribed_partial = transcribed_text[:match_length]  # 앞 음절만 추출
    print(f"Matching with partial text: {transcribed_partial}")
    best_match = None
    highest_similarity = 0

    for line in script:
        line_partial = line["text"][:match_length]
        similarity = difflib.SequenceMatcher(None, transcribed_partial, line_partial).ratio()
        print(f"Comparing '{transcribed_partial}' with '{line_partial}' -> Similarity: {similarity}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = line

    return best_match if highest_similarity > 0.5 else {"text": "No match found"}

# 디버그용 오디오 데이터를 저장하는 함수
def save_audio_debug(audio_data, filename="debug_audio.wav"):
    """
    디버깅용으로 오디오 데이터를 WAV 파일로 저장합니다.
    Parameters:
    - audio_data (bytes): 오디오 데이터
    - filename (str): 저장할 파일 이름
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # 모노
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(16000)  # 샘플링 레이트
        wf.writeframes(audio_data)

# Whisper로 오디오 전사
def transcribe_audio_chunk(audio_data, model):
    """
    Whisper를 사용하여 오디오 청크를 텍스트로 변환합니다.
    Parameters:
    - audio_data (bytes): 오디오 데이터
    - model: Whisper 모델 객체
    Returns:
    - str: 전사된 텍스트
    """
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        # WAV 파일로 저장
        with wave.open(temp_audio, "wb") as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)  # 샘플링 레이트
            wf.writeframes(audio_data)

        temp_audio_path = temp_audio.name

    try:
        # Whisper로 전사
        result = model.transcribe(temp_audio_path, language="ko")
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""
    finally:
        os.remove(temp_audio_path)

# 실시간 마이크 입력 처리 (오버랩 처리 추가)
def start_recognition(chunk_duration=2, overlap_duration=1, total_duration=10):
    """
    마이크에서 입력받아 Whisper로 청크 단위로 처리하고 대본 매칭 결과를 출력합니다.
    Parameters:
    - chunk_duration (int): 각 청크의 길이(초)
    - overlap_duration (int): 청크 간 겹치는 길이(초)
    - total_duration (int): 총 실행 시간(초)
    """
    # Whisper 모델 로드
    model = whisper.load_model("base")

    # 대본 데이터 로드
    with open("script.json", "r", encoding="utf-8") as f:
        script = json.load(f)

    # PyAudio 설정
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"Listening for {total_duration} seconds in {chunk_duration}-second chunks with {overlap_duration}-second overlap...")

    frames = []
    chunk_frames = int(RATE / CHUNK * chunk_duration)
    overlap_frames = int(RATE / CHUNK * overlap_duration)
    total_chunks = int((total_duration - chunk_duration) / (chunk_duration - overlap_duration)) + 1

    previous_frames = []  # 이전 청크 데이터 저장

    try:
        for _ in range(total_chunks):
            print("Recording chunk...")
            frames = []

            for _ in range(chunk_frames):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            # 오버랩 처리: 이전 청크 데이터 추가
            combined_frames = previous_frames[-overlap_frames:] + frames
            audio_data = b"".join(combined_frames)

            # 디버그용 파일 저장
            save_audio_debug(audio_data, "debug_audio.wav")
            print("Debug audio saved as debug_audio.wav.")

            transcription = transcribe_audio_chunk(audio_data, model)
            print(f"Transcribed Text: {transcription}")

            # 대본 매칭
            matched_line = match_script(transcription, script)
            print(f"Matched Line: {matched_line}")

            # 현재 청크를 이전 청크 데이터로 저장
            previous_frames = frames
    except KeyboardInterrupt:
        print("Stopped by user.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    start_recognition(chunk_duration=2, overlap_duration=1, total_duration=10)
