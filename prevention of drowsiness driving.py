import dlib
import cv2
import numpy as np
import time
import serial

#아두이노 통신
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
#라즈베리파이에서 명령어 sudo chmod 777 /dev/ttyACM0 로 권한 주기.
#아두이노 통신 함수, 정수값을 보냄.
def send_integer(value):
    ser.write(str(value).encode()) # 정수 값을 문자열로 변환
    ser.write(b'\n')

# 각 부위별 랜드마크, 찍는 점의 갯수
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Variables for tracking eye closure
closed_threshold = 3  # 눈 감고 있는 시간 임계값 (초)
face_not_detected_threshold = 2  # 얼굴 감지되지 않는 시간 임계값 (초)
vid_in = cv2.VideoCapture(0)  # 외부 웹캠 사용
threshold = 0.3  # 임계값 조정- 값이 커질수록 더 잘 감지됨.

# Variables for tracking closed time
closed_start_time = None
eyes_closed_flag = False
face_not_detected_flag = True
face_not_detected_start_time = time.time()  # 얼굴 감지가 시작된 시간 초기화


def eye_aspect_ratio(eye):
    # 눈의 랜드마크를 넘파이 배열로 변환
    eye = np.array(eye, dtype=np.float32)
    
    A = np.linalg.norm(eye[1] - eye[5])# 두 눈의 수직길이
    B = np.linalg.norm(eye[2] - eye[4])
 
    #눈의 가로 길이
    C = np.linalg.norm(eye[0] - eye[3])
 
    #눈이 감겼는지 계산한 값
    ear = (A + B) / (2.0 * C)
    return ear

#눈 감겼는지 판단
def eyes_closed(landmarks):
    left_eye_pts = [landmarks[i] for i in LEFT_EYE]#왼쪽 눈 랜드마크 추출
    right_eye_pts = [landmarks[i] for i in RIGHT_EYE] #오른쪽 눈 랜드마크 추출

    #눈의 길이 계산
    left_eye_ear = eye_aspect_ratio(left_eye_pts)
    right_eye_ear = eye_aspect_ratio(right_eye_pts)

    #계산된 값
    avg_ear = (left_eye_ear + right_eye_ear) / 2.0

    if avg_ear < threshold: #설정한 임계값과 비교
        return True
    else:
        return False

while True:
    # Get frame from video
    ret, image_o = vid_in.read()

    # 영상 사이즈 조정
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img_gray, 1)

    #얼굴 인식중
    if len(face_detector) > 0:
        face_not_detected_flag = False
    #얼굴 인식 안됨.
    else:
        if not face_not_detected_flag:
            face_not_detected_flag = True
            face_not_detected_start_time = time.time()

    #특정 시간동안 얼굴을 인식 못하면 값을 보냄.
    if face_not_detected_flag and time.time() - face_not_detected_start_time >= face_not_detected_threshold:
        send_integer(2)#아두이노로 2의 값을 보냄.
        print("Face not detected for more than {} seconds".format(face_not_detected_threshold))
        face_not_detected_flag = False

        # 얼굴 감지될 때까지 기다림
        while True:
            ret, image_o = vid_in.read()
            img_gray = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)
            face_detector = detector(img_gray, 1)
            if len(face_detector) > 0:
                face_not_detected_flag = False
                break
            
    for face in face_detector:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)

        landmarks = predictor(image, face)  

        landmark_list = []

        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

        if eyes_closed(landmark_list):
            #눈을 감았을 때
            if not eyes_closed_flag:
                closed_start_time = time.time()
                
                eyes_closed_flag = True
            elif closed_start_time is not None and time.time() - closed_start_time >= closed_threshold:
                print("Eyes have been closed for more than {} seconds".format(closed_threshold))
                send_integer(1)#아두이노로 1의 값을 보냄.
                closed_start_time = None
        else:
            eyes_closed_flag = False
            
        
        if eyes_closed_flag:
            cv2.putText(image, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            send_integer(3)#아두이노로 3의 값을 보냄.
            

    cv2.imshow('result', image)

    # wait for keyboard input
    key = cv2.waitKey(1)

    #esc 누르면 종료
    if key == 27:
        send_integer(3)#아두이노로 3의 값을 보냄.
        ser.close()
        break

vid_in.release()
cv2.destroyAllWindows()
