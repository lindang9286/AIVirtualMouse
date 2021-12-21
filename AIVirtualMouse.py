import cv2
import numpy as np
import HandTrackingModule as htm
import mediapipe as mp
import time
import autopy

##########################
wCam, hCam = 640,480
frameR = 100 # Frame Reduction
smoothening = 7
#########################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cap.set(3,wCam)
cap.set(4,hCam)
#detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
tipIds = [4, 8, 12, 16, 20]
while True:
    #1. Find hand Landmarks
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    lmList = []
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append( [id, cx, cy])
                cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
            lmlist=[]

    if len(lmList) != 0:
        fingers = []
        x1, y1 = lmList [8][1:]
        x2, y2 = lmList [12][1:]
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers ==[1,1,0,0,0]:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX,clocY
        if fingers == [0, 1, 0, 0, 0]:
            autopy.mouse.click()

    #11. frame rate
    cTime = time.time()
    fps =1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    #12. display
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break