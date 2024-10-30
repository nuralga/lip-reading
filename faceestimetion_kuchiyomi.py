# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import glob
# import os

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# mp_drawing = mp.solutions.drawing_utils

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# folder_path = 'frame'  # ここを適切なフォルダのパスに変更
# image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

# print(image_paths)


# for image_path in image_paths:
#     # 各画像を読み込み
#     image = cv2.imread(image_path)
#     print(image_path)
#     image = cv2.flip(image, 1)

#     start = time.time()

#     # convert the color space from BGR to RGB
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)     # e.g. selfie-view display

#     # To improve performance
#     image.flags.writeable = False
    
#     # Get the result
#     results = face_mesh.process(image)
    
#     # To improve performance
#     image.flags.writeable = True
    
#     # Convert the color space from RGB to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     img_h, img_w, img_c = image.shape
#     face_3d = []
#     face_2d = []

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
#                     if idx == 1:
#                         nose_2d = (lm.x * img_w, lm.y * img_h)
#                         nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

#                     x, y = int(lm.x * img_w), int(lm.y * img_h)

#                     # Get the 2D Coordinates
#                     face_2d.append([x, y])

#                     # Get the 3D Coordinates
#                     face_3d.append([x, y, lm.z])       
            
#             # Convert it to the NumPy array
#             face_2d = np.array(face_2d, dtype=np.float64)

#             # Convert it to the NumPy array
#             face_3d = np.array(face_3d, dtype=np.float64)

#             # The camera matrix
#             focal_length = 1 * img_w

#             cam_matrix = np.array([ [focal_length, 0, img_h / 2],
#                                     [0, focal_length, img_w / 2],
#                                     [0, 0, 1]])

#             # The distortion parameters
#             dist_matrix = np.zeros((4, 1), dtype=np.float64)

#             # Solve PnP
#             success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

#             # Get rotational matrix
#             rmat, jac = cv2.Rodrigues(rot_vec)

#             # Get angles
#             angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

#             # Get the y rotation degree
#             x = angles[0] * 360
#             y = angles[1] * 360
#             z = angles[2] * 360
        

#             # See where the user's head tilting
#             if y < -10:
#                 text = "Looking Left"
#             elif y > 10:
#                 text = "Looking Right"
#             elif x < -10:
#                 text = "Looking Down"
#             elif x > 10:
#                 text = "Looking Up"
#             else:
#                 text = "Forward"

#             # Display the nose direction
#             nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

#             p1 = (int(nose_2d[0]), int(nose_2d[1]))
#             p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
#             cv2.line(image, p1, p2, (255, 0, 0), 3)

#             # Add the text on the image
#             cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#             cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#         end = time.time()
#         totalTime = end - start

#         #fps = 1 / totalTime
#         #print("FPS: ", fps)

#         #cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

#         mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     #connections=mp_face_mesh.FACE_CONNECTIONS,
#                     connections=mp_face_mesh.FACEMESH_CONTOURS,                     
#                     landmark_drawing_spec=drawing_spec,
#                     connection_drawing_spec=drawing_spec)

#     print(image_path)
#     cv2.imshow('Head Pose Estimation', image)

#     if cv2.waitKey(5) & 0xFF == 27:
#         break
        
#         # cv2.waitKey(0)


#cap.release()


import numpy as np
import math
import cv2
from PIL import Image
import mediapipe as mp
import torch
from torchvision import transforms
import os

# left eye contour
landmark_left_eye_points = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]
# right eye contour
landmark_right_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

size_LFROI = 160 # [pixel]
size_graph_width = 200 # [pixel]
size_graph_height = 140 # [pixel]

# data transform
transform = transforms.Compose([
    #transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Initialize Mediapipe FaceMesh globally
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# training model path
model_path = r"/Users/nurzh/AGH/lip-reading/new_kuchiyomi/Lab_03_student.7z/streamlit_KuchiYomi_mouth_shape_recognition-main/model/model_P00.pth"
# load model
model = torch.load(model_path)

def set_model(target_person_id):
    model_file_name = "model/model_" + target_person_id + ".pth"
    model = torch.load(model_file_name)

# load device : cpu
device = torch.device("cpu")
model.to(device)

# モデルを評価モードにする
model.eval()

magrin = 5

str_message1 = ""
str_message2 = ""



def pil2cv(image):
    ''' PIL型 --> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        # モノクロ
        pass
    elif new_image.shape[2] == 3:
        # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
        
    return new_image


def cv2pil(image):
    ''' OpenCV型 --> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:
        # モノクロ
        pass
    elif new_image.shape[2] == 3:
        # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]

    new_image = Image.fromarray(new_image)

    return new_image


def func(value1, value2):
    return int(value1 * value2)


def LFROI_extraction_sub(image, face_points0):
    global str_message1
    global str_message2

    image_height, image_width, channels = image.shape[:3]

    image_cx = image_width / 2
    image_cy = image_height / 2

    left_eye_x = face_points0[33][0]
    left_eye_y = face_points0[33][1]

    right_eye_x = face_points0[263][0]
    right_eye_y = face_points0[263][1]

    nose_x = face_points0[2][0]
    nose_y = face_points0[2][1]

    eye_distance2 = (left_eye_x - right_eye_x) * (left_eye_x - right_eye_x) + (left_eye_y - right_eye_y) * (left_eye_y - right_eye_y)
    eye_distance = math.sqrt(eye_distance2)

    value = float(left_eye_y - right_eye_y) / float(left_eye_x - right_eye_x)
    if left_eye_x != right_eye_x:
        eye_angle = math.atan(float(left_eye_y - right_eye_y) / float(left_eye_x - right_eye_x))
    else:
        eye_angle = 0

    eye_angle = math.degrees(eye_angle)

    target_eye_distance = 160
    scale = target_eye_distance / eye_distance
    cx = nose_x
    cy = nose_y

    mat_rot = cv2.getRotationMatrix2D((int(cx), int(cy)), eye_angle, scale)
    tx = image_cx - cx
    ty = image_cy - cy
    mat_tra = np.float32([[1, 0, tx], [0, 1, ty]])

    normalized_image1 = cv2.warpAffine(image, mat_rot, (int(image_width), int(image_height)))
    normalized_image2 = cv2.warpAffine(normalized_image1, mat_tra, (int(image_width), int(image_height)))

    face_points1 = np.array([face_points0])
    face_points2 = cv2.transform(face_points1, mat_rot)
    face_points3 = cv2.transform(face_points2, mat_tra)
    face_points3 = np.squeeze(face_points3)
    
    #for p in face_points3:
    #    x = p[0]
    #    y = p[1]
    #    cv2.circle(normalized_image2, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)

    lip_x = (face_points3[61][0] + face_points3[291][0]) / 2
    lip_y = (face_points3[61][1] + face_points3[291][1]) / 2
    left = int(lip_x - target_eye_distance / 2)
    top = int(lip_y - target_eye_distance / 3)
    right = left + target_eye_distance
    bottom = top + target_eye_distance

    str_message1 = "eye distance = %.0f pixel" % eye_distance
    str_message2 = "angle = %.2f deg" % eye_angle
    
    return (left, top, right, bottom), normalized_image2, face_points3


def LFROI_extraction(image):
    global str_message1
    global str_message2

    out_image = image.copy()

    black_image = np.zeros((size_LFROI, size_LFROI, 3), np.uint8)
    white_image = black_image + 200
    is_detected_face = False

    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    image = cv2.flip(image, 1)

#     start = time.time()

    # convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)     # e.g. selfie-view display

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
                    points = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                              if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                        if idx == 1:
                                                  nose_2d = (lm.x * img_w, lm.y * img_h)
                                                  nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                                        
                              x, y = int(lm.x * img_w), int(lm.y * img_h)

                              # Get the 2D Coordinates
                              face_2d.append([x, y])

                              # Get the 3D Coordinates
                              face_3d.append([x, y, lm.z])


                              px = func(lm.x, img_w)
                              py = func(lm.y, img_h)
                              if idx in [33, 263, 2, 61, 291]:
                                        cv2.circle(out_image, center=(px,py), radius=3, color=(0, 255, 0), thickness=-1)
                                        cv2.circle(out_image, center=(px, py), radius=1, color=(255, 255, 255), thickness=-1)
                              else:
                                        cv2.circle(out_image, center=(px, py), radius=1, color=(0, 0, 255), thickness=-1)
                                        #cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                              points.append((px, py))
                    
          # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

            # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360
        

                    # See where the user's head tilting
                    if y < -10:
                              text = "Looking Left"
                    elif y > 10:
                              text = "Looking Right"
                    elif x < -10:
                              text = "Looking Down"
                    elif x > 10:
                              text = "Looking Up"
                    else:
                              text = "Forward"

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

          #   Add the text on the image
                    # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          


                    # print(points)
          rect_LFROI, normalized_image, new_points_LFROI = LFROI_extraction_sub(image, points)

          LFROI = normalized_image[rect_LFROI[1]:rect_LFROI[3], rect_LFROI[0]:rect_LFROI[2]]

          print("drawing")
          mp_drawing.draw_landmarks(
                              image=image,
                              landmark_list=face_landmarks,
                              #connections=mp_face_mesh.FACE_CONNECTIONS,
                              connections=mp_face_mesh.FACEMESH_CONTOURS,                     
                              landmark_drawing_spec=drawing_spec,
                              connection_drawing_spec=drawing_spec)

          is_detected_face = True
            #st.image(normalized_image)

          return image, LFROI, is_detected_face


        #end = time.time()
        #totalTime = end - start

        #fps = 1 / totalTime
        #print("FPS: ", fps)

        #cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            

        #print(image)
        #results = face_mesh.process(image)
           # (image_height, image_width) = image.shape[:2]

#          if results.multi_face_landmarks:
#             print(f"Detected landmarks: {results.multi_face_landmarks}")
#             for face in results.multi_face_landmarks:

                              # points = []
                              # x = func(landmark.x, img_w)
                              # y = func(landmark.y, img_h)
                              # if index in [33, 263, 2, 61, 291]:
                              #           cv2.circle(out_image, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                              #           cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                              # else:
                              #           cv2.circle(out_image, center=(x, y), radius=1, color=(0, 0, 255), thickness=-1)
                              #           #cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                              # points.append((x, y))
                              #print(x,y)

          # print("ok")
          # rect_LFROI, normalized_image, new_points_LFROI = LFROI_extraction_sub(image, points)

          # LFROI = normalized_image[rect_LFROI[1]:rect_LFROI[3], rect_LFROI[0]:rect_LFROI[2]]

          # is_detected_face = True
          #   #st.image(normalized_image)

          # return out_image, LFROI, is_detected_face
    else:
          print("not face")

    str_message1 = "no face detected"
    str_message2 = ""
    
    return out_image, white_image, is_detected_face

# def LFROI_extraction(image):
#     global str_message1
#     global str_message2

#     out_image = image.copy()
#     black_image = np.zeros((size_LFROI, size_LFROI, 3), np.uint8)
#     white_image = black_image + 200
#     is_detected_face = False

#     mp_drawing = mp.solutions.drawing_utils
#     drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#     image = cv2.flip(image, 1)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False

#     results = face_mesh.process(image)

#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     img_h, img_w, img_c = image.shape
#     face_3d = []
#     face_2d = []

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             points = []
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 x, y = int(lm.x * img_w), int(lm.y * img_h)
#                 face_2d.append([x, y])
#                 face_3d.append([x, y, lm.z])
#                 px = func(lm.x, img_w)
#                 py = func(lm.y, img_h)
#                 color = (0, 255, 0) if idx in [33, 263, 2, 61, 291] else (0, 0, 255)
#                 cv2.circle(out_image, center=(px, py), radius=3 if idx in [33, 263, 2, 61, 291] else 1, color=color, thickness=-1)
#                 points.append((px, py))

#         rect_LFROI, normalized_image, new_points_LFROI = LFROI_extraction_sub(image, points)
#         LFROI = normalized_image[rect_LFROI[1]:rect_LFROI[3], rect_LFROI[0]:rect_LFROI[2]]
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_CONTOURS,                     
#             landmark_drawing_spec=drawing_spec,
#             connection_drawing_spec=drawing_spec
#         )
#         is_detected_face = True
#         return image, LFROI, is_detected_face

#     str_message1 = "no face detected"
#     str_message2 = ""
#     return out_image, white_image, is_detected_face


def preprocess(image, transform):   
    image = transform(image)  # PIL
    C, H, W = image.shape
    image = image.reshape(1, C, H, W)
    
    return image


def make_graph_image(values):
    graph_image = np.zeros((size_graph_height, size_graph_width, 3), np.uint8)

    fontface = cv2.FONT_HERSHEY_PLAIN
    label = ["a", "i", "u", "e", "o", "N"]
    fontscale = 1.0
    thickness = 1

    max_idx = np.argmax(values)

    x0 = 90
    for idx, v in enumerate(values):
        x1 = x0 + int(v * 100)
        y0 = 10 + idx * 20
        y1 = 10 + (idx + 1) * 20
        if idx == max_idx:
            cv2.rectangle(graph_image, (x0, y0), (x1, y1), (0, 0, 255), -1)
            #cv2.rectangle(graph_image, (x0+1, y0+1), (x1-1, y1-1), (200, 200, 255), -1)
        else:
            cv2.rectangle(graph_image, (x0, y0), (x1, y1), (0, 255, 0), -1)
            cv2.rectangle(graph_image, (x0+1, y0+1), (x1-1, y1-1), (200, 255, 200), -1)

        (w, h), baseline = cv2.getTextSize(label[idx], fontface, fontscale, thickness)
        x = int((20 - w) / 2)
        cv2.putText(graph_image, label[idx], (x, y1-3), fontface, fontscale, (255, 255, 255), thickness)

        str_value = "(%0.3f)" % v
        cv2.putText(graph_image, str_value, (25, y1-3), fontface, fontscale, (255, 255, 255), thickness)
        
    return graph_image


def prediction(model, crop_image):
    with torch.no_grad():
        # 予測
        outputs = model(crop_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        graph_image = make_graph_image(probabilities)
        
        # 予測結果をクラス番号に変換
        _, predicted = torch.max(outputs, 1)

    return predicted, graph_image


def lip_reading(image_cv):
    image_height, image_width, channels = image_cv.shape[:3]

    # LFROI extraction
    # 五つのmissing ScriptRunContext
    image_cv, LFROI_cv, is_detected_face = LFROI_extraction(image_cv)
    
    out_image_cv = image_cv.copy()
    #out_image_cv = image_cv.copy()

    if is_detected_face == True:
        out_image_cv[magrin:size_LFROI+magrin, magrin:size_LFROI+magrin] = LFROI_cv

        LFROI_array = cv2pil(LFROI_cv)
        #crop_image_pil = preprocess(LFROI_array, transform)
        crop_image_pil = preprocess(LFROI_cv, transform)

        # predict
        predict, graph_image_cv = prediction(model, crop_image_pil)
        out_image_cv[magrin:magrin+size_graph_height, image_width-1-magrin-size_graph_width:image_width-1-magrin] = graph_image_cv
    
    cv2.putText(out_image_cv, str_message1, (20, image_height-60), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
    cv2.putText(out_image_cv, str_message2, (20, image_height-40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)

    return out_image_cv


def process_images_in_folder(input_folder_path, output_folder_path):
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder_path, exist_ok=True)

    # フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder_path):
        # 画像ファイルのみを処理
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像の拡張子を指定
            # 画像のフルパスを作成
            image_path = os.path.join(input_folder_path, filename)
            
            try:
                # 画像を読み込む
                image_cv = cv2.imread(image_path)
          #       image_height, image_width, channels = image_cv.shape[:3]

                #print(image_cv)
                if image_cv is None:
                    print(f"Error: Unable to read image {filename}. Skipping...")
                    continue
                
                # 画像処理関数を呼び出す
                processed_image = lip_reading(image_cv)

                # 結果を保存する
                output_path = os.path.join(output_folder_path, 'processed_' + filename)
                cv2.imwrite(output_path, processed_image)

                print(f"Processed {filename} and saved to {output_path}.")
                
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

# 使用例
input_folder_path = 'word_frame'  # 処理する画像フォルダのパス
output_folder_path = 'processed_images_word_1'  # 出力フォルダのパス
process_images_in_folder(input_folder_path, output_folder_path)
