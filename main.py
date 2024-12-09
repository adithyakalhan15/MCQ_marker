import cv2
import numpy as np
import tools

image_path = '33.png' #image to tested
questions=10 #No of questions
choices=5 #No of choises
heightImg = 700
widthImg  = 700
ans1 = [1, 2, 1, 2, 0, 1, 2, 4, 2, 4] 
ans2 = [1, 2, 2, 3, 4, 1, 2, 3, 0, 3] # Answers

count=0

image = cv2.imread(image_path)  # Read image
Limage = cv2.imread(image_path)
image = cv2.resize(image, (widthImg, heightImg))
FinalImage = image.copy()
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 70) #apply canny method

try:
    imgContour = image.copy()
    imgLContour = image.copy()
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #find all contours
    cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 5)  # Draw contours on picture

    rectCon = tools.rectContour(contours)
    biggestRectangle = tools.getCornerPoints(rectCon[1])  # Get biggest contour
    LeftRectangle = tools.getCornerPoints(rectCon[0]) #get left contour
    RightRecatngle = tools.getCornerPoints(rectCon[1]) #get right contour
    gradeRectangle = tools.getCornerPoints(rectCon[2])  # Get final marks contour

    if gradeRectangle.size != 0 and LeftRectangle.size != 0 and RightRecatngle.size != 0:
        
        LeftRectangle = tools.reorder(LeftRectangle)
        cv2.drawContours(imgLContour, LeftRectangle, -1, (0, 255, 0), 20)
        Lpts1 = np.float32(LeftRectangle)
        Lpts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # take points to warp
        Lmatrix = cv2.getPerspectiveTransform(Lpts1, Lpts2) # get transformation matrix
        LimgWarpColored = cv2.warpPerspective(image, Lmatrix, (widthImg, heightImg)) # apply warp perspective
        
        RightRecatngle = tools.reorder(RightRecatngle)
        cv2.drawContours(imgLContour, RightRecatngle, -1, (0, 255, 0), 20)
        Rpts1 = np.float32(RightRecatngle)
        Rpts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        Rmatrix = cv2.getPerspectiveTransform(Rpts1, Rpts2)
        RimgWarpColored = cv2.warpPerspective(image, Rmatrix, (widthImg, heightImg))
               
        cv2.drawContours(imgLContour, gradeRectangle, -1, (255, 0, 0), 20)
        gradePoints = tools.reorder(gradeRectangle)
        ptsG1 = np.float32(gradePoints)
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
        imgGradeDisplay = cv2.warpPerspective(image, matrixG, (325, 150))
        
        LimgWarpColored = cv2.resize(LimgWarpColored, (widthImg, heightImg))
        RimgWarpColored = cv2.resize(RimgWarpColored, (widthImg, heightImg))
        
        LimgWarpGray = cv2.cvtColor(LimgWarpColored, cv2.COLOR_BGR2GRAY)
        LimgThresh = cv2.threshold(LimgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
        
        RimgWarpGray = cv2.cvtColor(RimgWarpColored, cv2.COLOR_BGR2GRAY)  # apply grayscale
        RimgThresh = cv2.threshold(RimgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1] # apply threshold
        
        boxesL = tools.splitBoxes(LimgThresh) #split into boxes
        LcountR = 0
        LcountC = 0
        LmyPixelVal = np.zeros((questions, choices)) #create array to store data
        
        boxesR = tools.splitBoxes(RimgThresh)
        RcountR = 0
        RcountC = 0
        RmyPixelVal = np.zeros((questions, choices))
        
        for Limage in boxesL: #store non zero values - values of the sizes in pixels
            totalPixelsL = cv2.countNonZero(Limage)
            LmyPixelVal[LcountR][LcountC] = totalPixelsL
            LcountC += 1
            if LcountC == choices:
                LcountC = 0
                LcountR += 1
        
       
        for Rimage in boxesR:
            totalPixelsR = cv2.countNonZero(Rimage)
            RmyPixelVal[RcountR][RcountC] = totalPixelsR
            RcountC += 1
            if RcountC == choices:
                RcountC = 0
                RcountR += 1
        
        # filter and get the user answers                
        myIndexL = []
        for xL in range(questions):
            arrL = LmyPixelVal[xL]
            LmyIndexVal = np.where(arrL == np.amax(arrL))
            myIndexL.append(LmyIndexVal[0][0])
         
        myIndexR = []   
        for xR in range(questions):
            arrR = RmyPixelVal[xR]
            RmyIndexVal = np.where(arrR == np.amax(arrR))
            myIndexR.append(RmyIndexVal[0][0])
            
        Newgrading = []
        for x in range(questions):
            if ans1[x] == myIndexL[x]:
                Newgrading.append(1)
            else:
                Newgrading.append(0)
                
        for x in range(questions):    
            if ans2[x] == myIndexR[x]:
                Newgrading.append(1)
            else:
                Newgrading.append(0)
        
        Newscore = (sum(Newgrading) / (2*questions)) * 100
        print(myIndexR)
        
        tools.showAnswers(LimgWarpColored, myIndexL, Newgrading[:questions], ans1)
        tools.drawGrid(LimgWarpColored)
        LimgRawDrawings = np.zeros_like(LimgWarpColored)
        tools.showAnswers(LimgRawDrawings, myIndexL, Newgrading[:questions], ans1)
        LinvMatrix = cv2.getPerspectiveTransform(Lpts2, Lpts1)
        LimgInvWarp = cv2.warpPerspective(LimgRawDrawings, LinvMatrix, (widthImg, heightImg))
        
        tools.showAnswersR(RimgWarpColored, myIndexR, Newgrading[questions:], ans2)
        tools.drawGrid(RimgWarpColored)
        RimgRawDrawings = np.zeros_like(RimgWarpColored)
        tools.showAnswersR(RimgRawDrawings, myIndexR, Newgrading[questions:], ans2)
        RinvMatrix = cv2.getPerspectiveTransform(Rpts2, Rpts1)
        RimgInvWarp = cv2.warpPerspective(RimgRawDrawings, RinvMatrix, (widthImg, heightImg))
        
        imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # NEW BLANK IMAGE WITH GRADE AREA SIZE
        cv2.putText(imgRawGrade,str(int(Newscore))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(50,50,255),2) # ADD THE GRADE TO NEW IMAGE
        invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # INVERSE TRANSFORMATION MATRIX
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP
        
        # print(questions)
        
        imgFinal = cv2.addWeighted(FinalImage, 1, LimgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, RimgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,0.7,0)

        imageArray = ([image, imgGray, imgCanny, imgContour],
                      [imgLContour, RimgThresh, RimgWarpColored, imgFinal])
        
        
except Exception as e:
    print(f"Error: {e}")
    imageArray = ([image, imgGray, imgCanny, imgContour],
                [imgBlank, imgBlank, imgBlank, imgBlank])


labels = [["Original", "Gray", "Edges", "Contours"],
          ["Biggest Contour", "Threshold", "Warped", "Final"]]

stackedImage = tools.stackImages(imageArray, 0.5, labels)

cv2.imshow("OCR checker",stackedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
