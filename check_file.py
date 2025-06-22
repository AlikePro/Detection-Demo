import os

file_path = r"C:\Users\kadyr\OneDrive\Рабочий стол\Alimzhan\RealTimeObjectDetection\venv\Lib\site-packages\mediapipe\modules\hand_landmark\hand_landmark_tracking_cpu.binarypb"

if os.path.exists(file_path):
    print("Файл найден!")
else:
    print("Файл отсутствует!")