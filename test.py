import cv2
import numpy as np
import mediapipe as mp
import os

# import re

def func(value1, value2):
    return int(value1 * value2)

def LFROI_extraction(image):
    global str_message1
    global str_message2

    out_image = image.copy()

    # black_image = np.zeros((size_LFROI, size_LFROI, 3), np.uint8)
    # white_image = black_image + 200
    is_detected_face = False

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image)
        (image_height, image_width) = image.shape[:2]

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                points = []
                for index, landmark in enumerate(face.landmark):
                    x = func(landmark.x, image_width)
                    y = func(landmark.y, image_height)
                    if index in [33, 263, 2, 61, 291]:
                        cv2.circle(out_image, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    else:
                        cv2.circle(out_image, center=(x, y), radius=1, color=(0, 0, 255), thickness=-1)
                        #cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    points.append((x, y))


                # rect_LFROI, normalized_image, new_points_LFROI = LFROI_extraction_sub(image, points)
                # LFROI = normalized_image[rect_LFROI[1]:rect_LFROI[3], rect_LFROI[0]:rect_LFROI[2]]

            is_detected_face = True
            #st.image(normalized_image)

            return out_image, is_detected_face
        else:
            print("not face")

    str_message1 = "no face detected"
    str_message2 = ""
    
    return out_image, is_detected_face


def process_images_in_folder(input_folder_path):
    total_images = 0
    correct_predictions = 0


    # 出力フォルダが存在しない場合は作成
    # os.makedirs(output_folder_path, exist_ok=True)

    # フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder_path):
        # 画像ファイルのみを処理
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像の拡張子を指定
            # 画像のフルパスを作成
            image_path = os.path.join(input_folder_path, filename)
            
            # try:
                # 画像を読み込む
            image_cv = cv2.imread(image_path)
          #       image_height, image_width, channels = image_cv.shape[:3]

                #print(image_cv)
            if image_cv is None:
                print(f"Error: Unable to read image {filename}. Skipping...")
                continue
            
            # 画像処理関数を呼び出す
            processed_image, predict_result = LFROI_extraction(image_cv)



# 使用例
input_folder_path = 'frame'  # 処理する画像フォルダのパス
# output_folder_path = 'processed_images_1'  # 出力フォルダのパス

# text_file_path = 'label.txt'  # テキストファイルのパス
process_images_in_folder(input_folder_path)