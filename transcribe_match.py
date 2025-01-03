import whisper
import pyaudio
import difflib
import wave
import os
import json
from tempfile import NamedTemporaryFile

# 대본 데이터를 JSON 파일에서 로드
def load_script(file_path):
    """
    JSON 파일에서 대본 데이터를 로드합니다.
    Parameters:
    - file_path (str): JSON 파일 경로
    Returns:
    - list: 대본 데이터 리스트
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 대본 매칭 함수
def match_script(transcribed_text, script, previous_match=None, next_line_to_check=None, first_check=False):
    """
    Whisper 전사 결과를 대본 데이터와 매칭합니다.
    Parameters:
    - transcribed_text (str): Whisper로부터 얻은 텍스트
    - script (list): 대본 데이터 리스트
    - previous_match (dict): 이전 매칭된 대본 줄
    - next_line_to_check (dict): 이전에 매칭된 문장의 다음 줄
    - first_check (bool): 첫 번째 매칭 여부 플래그
    Returns:
    - dict: 매칭된 대본 줄과 매칭 방식
    """
    if not transcribed_text.strip():  # 전사된 텍스트가 비어 있는 경우 처리
        print("Transcribed text is empty. Skipping matching.")
        return {"text": "No transcription", "method": "none", "next_line_to_check": next_line_to_check}

    best_match = None
    highest_similarity = 0

    # 첫 번째 매칭: 첫 대본과 유사도 확인
    if first_check:
        first_script_line = script[0]["text"]
        similarity = difflib.SequenceMatcher(None, transcribed_text, first_script_line).ratio()
        print(f"Similarity with first line: {similarity}\n")
        if similarity >= 0.3:
            print(f"Matched with first line: {first_script_line} -> Similarity: {similarity}\n")
            next_line_to_check = script[1] if len(script) > 1 else None
            return {"text": script[0]["text"], "method": "first_line", "next_line_to_check": next_line_to_check}

    # 이전 매칭된 줄의 다음 줄과 비교
    if next_line_to_check:
        similarity = difflib.SequenceMatcher(None, transcribed_text, next_line_to_check["text"]).ratio()
        print(f"Similarity with next line to check: {similarity}\n")
        if similarity >= 0.3:
            print(f"Matched with next line to check: {next_line_to_check['text']} -> Similarity: {similarity}\n")
            next_line_index = script.index(next_line_to_check) + 1
            next_line_to_check = script[next_line_index] if next_line_index < len(script) else None
            return {"text": next_line_to_check["text"], "method": "next_line_to_check", "next_line_to_check": next_line_to_check}

    # 전체 스크립트에서 가장 유사한 줄 찾기
    for line in script:
        line_text = line["text"]
        similarity = difflib.SequenceMatcher(None, transcribed_text, line_text).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = line

    if best_match:
        print(f"Best Match: {best_match['text']} -> Similarity: {highest_similarity}\n")
        next_line_index = script.index(best_match) + 1
        next_line_to_check = script[next_line_index] if next_line_index < len(script) else None
        return {"text": best_match["text"], "method": "overall", "next_line_to_check": next_line_to_check}
    else:
        print("No suitable match found.")
        return {"text": "No match found", "method": "none", "next_line_to_check": next_line_to_check}

# 디버그용 오디오 데이터를 저장
def save_audio_debug(audio_data, filename="debug_audio.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)

# Whisper로 오디오 전사
def transcribe_audio_chunk(audio_data, model):
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        with wave.open(temp_audio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        temp_audio_path = temp_audio.name

    try:
        result = model.transcribe(temp_audio_path, language="ko")
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""
    finally:
        os.remove(temp_audio_path)

# 실시간 마이크 입력 처리
def start_recognition(script_path, chunk_duration=2, overlap_duration=1, total_duration=10):
    model = whisper.load_model("base")

    # 대본 데이터 로드
    script = load_script(script_path)

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

    previous_match = None  # 이전 매칭된 대본
    has_started = False  # 첫 시작 여부 플래그

    try:
        for _ in range(total_chunks):
            print("Recording chunk...")
            frames = []

            for _ in range(chunk_frames):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            combined_frames = frames if not has_started else previous_frames[-overlap_frames:] + frames
            audio_data = b"".join(combined_frames)

            save_audio_debug(audio_data, "debug_audio.wav")
            print("Debug audio saved as debug_audio.wav.")

            transcription = transcribe_audio_chunk(audio_data, model)
            print(f"Transcribed Text: {transcription}")

            if not has_started and transcription.strip():  # 첫 전사된 텍스트가 있는 경우 시작
                has_started = True
                print("First non-empty transcription detected, starting script matching...")

            if has_started:
                previous_match = match_script(transcription, script, previous_match)
                print(f"Matched Line: {previous_match}")

            previous_frames = frames
    except KeyboardInterrupt:
        print("Stopped by user.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    # JSON 파일 경로 지정
    start_recognition(script_path="script.json", chunk_duration=2, overlap_duration=1, total_duration=10)
