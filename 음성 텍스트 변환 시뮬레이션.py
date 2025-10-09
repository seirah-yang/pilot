import speech_recognition as sr

r = sr.Recognizer()
with sr.AudioFile("nurse_note.wav") as source:
    audio = r.record(source)

try:
    text = r.recognize_google(audio, language="ko-KR")
    print("음성 인식 결과:", text)
except Exception as e:
    print("인식 실패:", e)
