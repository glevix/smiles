##################################### IMPORTS ########################################
import cv2
import time
from lenet import LeNet
#################################### CONSTANTS #######################################
cascade_path = 'haarcascade_frontalface_default.xml'
MAX_SNAPSHOTS = 3
DELAY_SECONDS = 1
BUILT_IN_CAMERA = 0
SAVE_NAME = 'snapshot'
MIN_CONFIDENCE = 0.6
NETWORK_INPUT_SHAPE = (28, 28)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
MIN_FACE_SIZE = (45, 45)

##################################### GLOBALS ########################################
cascade = cv2.CascadeClassifier(cascade_path)
model = LeNet.load_bagged(5)

##################################### METHODS ########################################


def process(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    num_faces = len(faces)
    if num_faces == 0:
        return frame, False
    num_smiles = 0
    for (x, y, w, h) in faces:
        face = cv2.resize(image[y:y+h, x:x+w], NETWORK_INPUT_SHAPE)
        is_smile = model.predict_single(face, MIN_CONFIDENCE)
        if is_smile:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            num_smiles = num_smiles + 1
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
    return frame, num_smiles == num_faces


def run():
    count = 0
    start = 0
    cap = cv2.VideoCapture(BUILT_IN_CAMERA)  # 0 for camera, filename for video
    while True:
        ret, im = cap.read()  # get current frame
        if not ret:
            break
        if start > 0:
            if time.time() - start < DELAY_SECONDS:
                cv2.imshow('frame', im)
                continue
            else:
                start = 0
        processed, smiles = process(im.copy())
        if smiles:
            cv2.imwrite(SAVE_NAME + '_' + str(count) + '.jpg', im)
            count = count + 1
            start = time.time()
        cv2.imshow('frame', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= MAX_SNAPSHOTS:
            break
    cap.release()
    cv2.destroyAllWindows()


run()
