import os.path
import cv2
from HandTrackingModule import HandDetector
import numpy as np
import math
import time
from glob import glob
from predict import PredictHandSign

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
# Set the width of the webcam feed
cap.set(3, 1280)  # 1280
# Set the height of the webcam feed
cap.set(4, 720)  # 720

# Initialize hand detector with specific confidence and one hand at a time
detector = HandDetector(detectionCon=0.8, maxHands=1)
# Initialize the hand sign prediction model
model = PredictHandSign()

# Function to create an opaque rectangle on the image
def opacityRec(img, bbox, color=(255, 0, 255), alpha=0.5):
    x1, y1, x2, y2 = bbox
    shapes = np.zeros_like(img, np.uint8)
    cv2.rectangle(shapes, (x1, y1), (x2, y2), color, cv2.FILLED)
    mask = shapes.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
    return img

# Function to draw buttons on the image
def drawButtons(img, buttonList):
    for button in buttonList:
        x1, y1, x2, y2 = button.bbox
        img = opacityRec(img, button.bbox)
        cv2.putText(img, button.text, (x1 + 20, y2 - 20),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    # Draw result box
    cv2.rectangle(img, (1020, 20), (1200, 150), (150, 50, 200), cv2.FILLED)
    cv2.rectangle(img, (1020, 150), (1200, 190), (101, 28, 138), cv2.FILLED)
    cv2.rectangle(img, (1020, 190), (1200, 260), (101, 28, 138), cv2.FILLED)
    opacityRec(img, (300, 590, 990, 690), color=(101, 28, 138))
    cv2.putText(img, "<Press S to cap>", (1020, 175),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(img, "<Press W to write || C to clear>", (500, 685),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    return img

# Function to create a subwindow for hand detection
def subWindow(frame, hands, imgSize, offset):
    # Create a white frame for the subwindow
    frameWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        y = max(20, y)
        x = max(20, x)
        frameCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            frameResize = cv2.resize(frameCrop, (wCal, imgSize))
            frameResizeShape = frameResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            frameWhite[:, wGap:wCal + wGap] = frameResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            frameResize = cv2.resize(frameCrop, (imgSize, hCal))
            frameResizeShape = frameResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            frameWhite[hGap:hCal + hGap, :] = frameResize

    return frameWhite

# Class to represent a button
class Button():
    def __init__(self, bbox, text):
        self.bbox = bbox
        self.text = text

# List to hold all buttons
buttonList = []
keys = ["A", "B", "C", "I", "<3", "U"]
# Create and position buttons based on the keys layout
for x, key in enumerate(keys):
    x1, y1 = x * 150 + 90, 20
    x2, y2 = x1 + 120, y1 + 100
    buttonList.append(Button([x1, y1, x2, y2], key))

# Initialize variables
keyWord = ""
offset = 20
imgSize = 300
folder = "data/train"
counter = 0
writer = ""
pred = ""

# Main loop to process the video feed
while True:
    flag, frame = cap.read()
    if not flag:
        break
    # Detect hand
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, draw=False)
    lmList, bboxInfo = detector.findPosition(frame)
    # Draw all buttons on the image
    frame = drawButtons(img=frame, buttonList=buttonList)
    frameWhite = subWindow(frame, hands, imgSize, offset)
    # Predict hand with model ResNet34
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        pred = model.predict(frameWhite)  # Predict hand sign
        cv2.putText(frame, pred, (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)

    # If landmarks are detected
    if lmList:
        for button in buttonList:
            x1, y1, x2, y2 = button.bbox
            text = button.text
            fingerTip_x = lmList[8][0]
            fingerTip_y = lmList[8][1]
            if x1 < fingerTip_x < x2 and y1 < fingerTip_y < y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 200), cv2.FILLED)
                cv2.putText(frame, text, (x1 + 20, y2 - 20),
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
                keyWord = button.text
                counter = 0
    # Display keyword
    cv2.putText(frame, keyWord, (1050, 120),
                cv2.FONT_HERSHEY_PLAIN, 7, (255, 255, 255), 4)
    # Display counter
    cv2.putText(frame, str(counter), (1077, 245),
                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
    # Display writer text
    cv2.putText(frame, writer, (330, 660),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Show frames
    cv2.imshow("Frame", frame)
    cv2.imshow("frameWhite", frameWhite)
    # Setup hot keys
    key = cv2.waitKey(10)
    if key == ord("s"):
        if len(keyWord) != 0:
            counter = len(glob("{}/{}/*.jpg".format(folder, keyWord)))
            imgPath = os.path.join(folder, keyWord)
            cv2.imwrite(f"{imgPath}/Img_{time.time()}.jpg", frameWhite)
            counter += 1
    if key == ord("c"):
        writer = ""
    if key == ord("w"):
        if pred == "<3":
            writer += "{} ".format("Love")
        else:
            writer += "{} ".format(pred)
    if key == ord("q"):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()