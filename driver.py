import cv2
from lenet import LeNet

cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

model = LeNet.load_bagged(5)


def process(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    toSave = image.copy()
    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    count = len(faces)
    if not count:
        return frame
    smiles = 0
    for (x, y, w, h) in faces:
        face = cv2.resize(image[y:y+h, x:x+w], (28, 28))
        is_smile = model.predict_single(face, 0.6)
        if is_smile:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            smiles = smiles + 1
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if smiles == count:
        cv2.imwrite('snap.jpg', toSave)
    return frame


cap = cv2.VideoCapture(0)  # 0 for camera, filename for video

while True:
    ret, im = cap.read()  # get current frame
    if not ret:
        break
    processed = process(im)
    cv2.imshow('frame', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
