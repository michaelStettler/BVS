import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import tqdm

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from utils.load_config import load_config
from utils.load_data import load_data


"""
run: python -m projects.memory_efficiency.05_get_MediaPipe_LMK_on_FERG
"""
#%%
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
print("-- Config loaded --")
print()

#%%
# Load data
train_data = load_data(config, get_raw=True)
train_label = train_data[1]
test_data = load_data(config, train=False, get_raw=True)
test_label = test_data[1]
print("shape train_data[0]", np.shape(train_data[0]))
print("shape test_data[0]", np.shape(test_data[0]))

#%%
# predict LMK
def predict_lmk(data):
    annotated_images = []
    landmarks = []
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, img in enumerate(data):
            # convert numpy to fit pipeline as uint8
            image = np.array(img).astype(np.uint8)

            # Convert the BGR image to RGB before processing.
            # results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = face_mesh.process(image)

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
            annotated_images.append(annotated_image)
            landmarks.append(face_landmarks)

    return np.array(landmarks, dtype=np.float16)


# predict lmk
train_lmk = predict_lmk(train_data[0])
test_lmk = predict_lmk(test_data[0])

#%%
# save lmk
np.save(os.path.join(config['directory'], "MediaPipe_train_LMK.npy"), train_lmk)
np.save(os.path.join(config['directory'], "MediaPipe_test_LMK.npy"), test_lmk)
