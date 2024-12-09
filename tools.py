import cv2
import numpy as np

def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    # print(add)
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,10)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

def showAnswers(img, myIndex, grading, ans, questions=10, choices=5):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    
    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)
            cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255)
            cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2), 20, myColor, cv2.FILLED)

def showAnswersR(img, myIndex, grading, ans, questions=10, choices=5):
    secW = int((img.shape[1]) / choices)
    secH = int(img.shape[0] / questions)
    
    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)
            cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255)
            cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2), 20, myColor, cv2.FILLED)

# def drawGrid(img,questions=10,choices=5):
#     secW = int(img.shape[1]/questions)
#     secH = int(img.shape[0]/choices)
#     for i in range (0,9):
#         pt1 = (0,secH*i)
#         pt2 = (img.shape[1],secH*i)
#         pt3 = (secW * i, 0)
#         pt4 = (secW*i,img.shape[0])
#         cv2.line(img, pt1, pt2, (255, 255, 0),2)
#         cv2.line(img, pt3, pt4, (255, 255, 0),2)

#     return img

def drawGrid(img, questions=10, choices=5):
    secW = int(img.shape[1] / choices)  # Divide image width by number of choices
    secH = int(img.shape[0] / questions)  # Divide image height by number of questions

    for i in range(0, questions + 1):  # Draw horizontal lines
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    for j in range(0, choices + 1):  # Draw vertical lines
        pt3 = (secW * j, 0)
        pt4 = (secW * j, img.shape[0])
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)


    return img


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver