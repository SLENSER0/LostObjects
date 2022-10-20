import torch
import cv2
import time
from skimage.metrics import structural_similarity
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = 0  # распознавать только людей

# модель для распозрнания объектов в рамках
model1 = torch.hub.load('ultralytics/yolov5', 'yolov5m')

alert = False

video = ''
img = '' # кадр, в котором нет потерянных предметов и людей
cap = cv2.VideoCapture(video)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter('video.avi', fourcc, 10, (1280, 960))

first_frame = cv2.imread(img) if len(img) != 0 else cap.read()[1]

start = 10**25

while True:
    ret, frame = cap.read()
    r = model(frame)
    person = False
    try:
        person = r.pandas().xyxy[0].value_counts('name').iloc[0]  # количество человек в кадре
        if person:
            start = time.time()
    except:
        pass

    if time.time() - start > 6:  # (time.time() - start) - время, которое прошло с последнего появления человека в кадре

        first_frame_gray = cv2.cvtColor(first_frame.copy(), cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Различие в кадрах
        (score, diff) = structural_similarity(first_frame_gray, frame_gray, full=True)
        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(first_frame.copy().shape, dtype='uint8')
        filled_after = frame.copy()

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Найти все отличия
            # if cv2.contourArea(c) > 300 and not (370 < x < 700 and 30 < y < 100):
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if cv2.contourArea(c) > 350:
                cropped = frame[y-10:y+h+10,x-10:x+w+10]
                r2 = model1(cropped)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                try:
                    k = r2.pandas().xyxy[0].value_counts('name').iloc[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    if not alert:
                        print('Кто-то оставил предмет')
                        alert = True
                except:
                    pass
    else:
        alert = False

        # if contours != 0:
        #     print('Кто-то оставил предмет')

    cv2.imshow('frame2', frame)
    # video.write(frame.copy())
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()


# video.release()

