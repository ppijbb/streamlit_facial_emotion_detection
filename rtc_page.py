import string

import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import random
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as _IMG


# load model
model = model_from_json(open("caer_face.json", "r").read())

# load weights
model.load_weights('caer_face.h5')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# face detection
face_haar_cascade = cv2.CascadeClassifier("C:\\Users\\Lenovo\\.conda\\envs\\python_3_9_env\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

def process_face(image):
    try:
        image.flags.writeable = False
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            # print('WORKING')
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y - 10:y + w + 10,
                       x - 10:x + h + 10]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = _IMG.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels *= 0.8

            # print(img_pixels.shape)

            predictions = model.predict(img_pixels)

            # print({"행복": predictions[0][0],
            #        "슬픔": predictions[0][1],
            #        "중립": predictions[0][2]})
            # find max indexed array

            max_index = np.argmax(predictions[0])

            # emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotions = ['happy', 'sad', 'neutral']
            predicted_emotion = emotions[max_index]

            cv2.putText(image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return cv2.flip(image, 1)

    except Exception as e:
        print("Exception", e)
        return cv2.flip(image, 1)


def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process_face(img)
        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def show():
    webrtc_ctx = webrtc_streamer(
        key=string.punctuation,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
        desired_playing_state=True,
        video_html_attrs={
            # "style": {"width": "50%", "margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

if __name__ == "__main__":
    show()