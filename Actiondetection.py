
#   1.Import and install Dependencies =========================================================================================


#  step 1.2 import

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # 




#   2.Keypoints using MP Holistic ============================================================================================



# step 2.2 設定或引入媒體管道 # Initialize Mediapipe Holistic

mp_holistic=mp.solutions.holistic # Holistic model
mp_drawing=mp.solutions.drawing_utils # Drawing utilities



# step 2.3 媒體管道偵測功能

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results





# step 2.4 draw landmark setting

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
        # 上面改成註解因為右側執行訊息:AttributeError: module 'mediapipe.python.solutions.holistic' has no attribute 'FACE_CONNECTIONS'. Did you mean: 'HAND_CONNECTIONS'?
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections



# step 2.5 draw style landmark setting   
    # ** step 2.5 cann't work **slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.

def draw_styled_landmarks(image, results):

    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 

    # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    #                          ) 
    # # Draw left hand connections
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    #                          ) 
    # # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# # step 2.1 設定攝影機

# cap=cv2.VideoCapture(0) #Set mediapipe model 


# # aft step 2.3 媒體管道偵測功能 add below # Make detections # 聲明能夠存取我們的整體模型_Access mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():  
        
#         #Read feed
#         ret,frame=cap.read()


# # after step 2.3 媒體管道偵測功能 add below # Make detections
#         # image,results=mediapipe_detection(frame,model)
#         image,results=mediapipe_detection(frame,holistic) 
    


# # aft step 2.4 draw landmark setting # Draw landmarks
#         draw_landmarks(image, results)


#         #Show to screen
#         # cv2.imshow('OpenCV Feed',frame) 
#         cv2.imshow('OpenCV Feed',image)  # Draw landmarks then modify to image

#         #Break gracefully
#         if cv2.waitKey(500)&0xFF==ord('q'): 
#             break
# cap.release()
# cv2.destroyAllWindows()




# #   3. Extract Keypoint Values============================================================================================


# # step 3.1 處理和提取不同的關鍵點

# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)


# # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
# # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
# # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
# rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


# -----------------------------------------------------------------------------------------------------------------------------

def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([rh]) # [pose, lh, rh] >> remove face



#   4. Setup Folders for Collection============================================================================================

# step 4.1 

# Path for exported data, numpy arrays

base_path='D:/Actiondetection' # 定義基本路徑
# DATA_PATH=os.path.join('MP_Data')   # 改成window路徑
DATA_PATH=os.path.join(base_path,'MP_Data')
print(DATA_PATH)  # 輸出: D:/Actiondetection/MP_Data

# Actions that we try to detect
# actions = np.array(['hello', 'thanks', 'iloveyou'])
actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

# Thirty videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 10

# Folder start
start_folder = 1


# step 4.2 創建一堆不同的資料夾 _ hello/thanks/iloveyou

# for action in actions: 
#     dirmax=np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
#     for sequence in range(1,no_sequences+1):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action,str(dirmax+sequence)))
#         except:
#             pass

# ChatGpt modified as below
# Create directories for each action and sequence
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    dirmax = max([int(d) for d in os.listdir(action_path) if d.isdigit()], default=0)
    for sequence in range(1, no_sequences + 1):
        sequence_path = os.path.join(action_path, str(dirmax + sequence))
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)



#   5. Collect Keypoint Values for Training and Testing================================================================================

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')
                np.save(npy_path, keypoints)

                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

            # Update the message after finishing collecting frames for a sequence
            print(f'Collecting frames for {action} Video Number {sequence} complete.')

    cap.release()
    cv2.destroyAllWindows()



#   6. Preprocess Data and Create Labels and Features================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []  # 引入數據,序列和標籤 
for action in actions:

    # for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):

    for sequence in range(no_sequences):
        window = []    # 該特定序列獲得的所有不同帪
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

    x=np.arry(sequences)
    y=to_categorical(labels).astype(int)
    y
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05)



#   7. Build and Train LSTM Neural Nerwork================================================================================

from tensorflow.keras.models import Sequential # 神經序列
from tensorflow.keras.layers import LSTM, Dense # 提供了建構神經網路的時間組件
from tensorflow.keras.callbacks import TensorBoard # 允許在張量版內部執行一些日記記錄

# 張量版回調
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# 神經網路
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res=[.7,0.2,0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) #編譯的模型 

model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])




#   8. Make Predictions=============================================================================================================

res = model.predict(x_test)



#   9. Save Weights=================================================================================================================

model.save('action.h5')
del model
model.load_weights('action.h5')



#   10. Evaluation using Confusion Matrix and Accuracy================================================================================

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()



#   11. Test in Real Time============================================================================================================

