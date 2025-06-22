import cv2
import pytesseract
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from googletrans import Translator
import os

# Укажи путь к tesseract (если Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Переводчик
translator = Translator()

# Путь к видео
video_path = "news_sample.mp4"  # замените на своё видео
output_path = "annotated_output.mp4"

# Функция обработки каждого кадра
def annotate_frame(get_frame, t):
    frame = get_frame(t)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # OCR
    text = pytesseract.image_to_string(img, lang='eng+kaz+rus')
    text = text.strip()
    if text:
        print(f"Найден текст: {text}")
        # Перевод
        translated = translator.translate(text, dest='en').text

        # Аннотация (вывод текста)
        annotation = f"Translated: {translated}"
        txt_clip = TextClip(annotation, fontsize=24, color='white', bg_color='black')
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(1)

        # Совмещение
        result = CompositeVideoClip([ImageClip(frame).set_duration(1), txt_clip])
        return result.get_frame(0)
    return frame

# Чтение и обработка видео
video = VideoFileClip(video_path)
processed = video.fl(annotate_frame)
processed.write_videofile(output_path, codec='libx264')

