import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

#some basic functions for ease
def load_img(name: str,mode: int) -> np.ndarray:
    return cv2.imread(name, mode)

def show_img(title: str,img: np.ndarray,wait: int) -> int:
    cv2.imshow(title, img)
    k = cv2.waitKey(wait)
    cv2.destroyWindow(title)
    return k

def save_img(name,img):
    if type(name) == str:
        cv2.imwrite(name, img)
    else:
        [cv2.imwrite("out/"+n+".jpg",i) for n,i in zip(name,img)]


#finding edge positions only
def getEdgePositions(img):
    positions = []
    m, n = img.shape
    for a in range(m):
        for b in range(n):
            if (img[a, b]!=0):
                positions.append((a, b))
    return positions

#reference point is taken as avg of all edge positions pixels
def getReferencePoint(edgePositions):
    a = 0
    b = 0
    for i in range(len(edgePositions)):
        a = a + edgePositions[i][0]
        b = b + edgePositions[i][1]
    a = a/len(edgePositions)
    b = b/len(edgePositions)
    return (int(a), int(b))

#constructing the R-table to get structure of template
def rTable(img, refPoint, edgePositions):
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)   #possibly tweaked
    abs_sobel64f = np.absolute(sobelx64f)
    rTable = {}
    for i, point in enumerate(edgePositions):
        rx = refPoint[0] - point[0]
        ry = refPoint[1] - point[1]
        r = (rx, ry)
        phi = abs_sobel64f[point[0], point[1]]
        if(phi not in list(rTable.keys())):
            rTable[phi] = [r]
        else:
            rTable[phi].append(r)
    rTable['refPoint'] = refPoint
    return rTable

#finding rTable for all theta with a jump of angle degrees. The tables are all stored in a list of length 360/angle.
def rTableWithRotation(templateCanny,angle=2):
    rTableWithRotation = []
    rows, columns = templateCanny.shape
    for i in range(int(360/angle)):
        theta = angle*i
        M = cv2.getRotationMatrix2D((columns/angle,rows/angle),theta,1)
        templateCannyRotated = cv2.warpAffine(templateCanny,M,(columns,rows))
        templateEdgePositions = getEdgePositions(templateCannyRotated)
        templateRefPoint = getReferencePoint(templateEdgePositions)
        rTableTheta = rTable(templateCannyRotated, templateRefPoint, templateEdgePositions)
        rTableWithRotation.append(rTableTheta)
    return rTableWithRotation

#finding the accumulatorTable for the main picture
def accumulatorTable(pictureCanny, template_rTable, angle=2):
    rows = pictureCanny.shape[0]
    columns = pictureCanny.shape[1]
    pictureEdgePositions = getEdgePositions(pictureCanny)
    accumulatorTable = np.ndarray((int(360/angle), rows, columns))
    sobelx64f = cv2.Sobel(pictureCanny,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)

    for theta, rTableTheta in enumerate(template_rTable):
        for i, edgePoint in enumerate(pictureEdgePositions):
            phi = abs_sobel64f[edgePoint[0], edgePoint[1]]
            if (phi in list(rTableTheta.keys())):
                temp = rTableTheta[phi]
                for r, vector in enumerate(temp):
                    x = edgePoint[0] + vector[0]
                    y = edgePoint[1] + vector[1]
                    if (x>=0 and x<rows) and (y>=0 and y<columns):
                        accumulatorTable[theta, x, y]+=1
            else:
                continue

    return accumulatorTable

#getting back the original image
def reconstruction(rTable, theta, a, b, pictureCanny, angle=2):
    rTable = rTable[int(theta)]
    pictureEdgePositions = getEdgePositions(pictureCanny)
    draw = np.ones_like(pictureCanny)*255
    mask = np.zeros_like(pictureCanny)*255
    maskingPoints = []
    rows = pictureCanny.shape[0]
    columns = pictureCanny.shape[1]
    sobelx64f = cv2.Sobel(pictureCanny,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)

    for i, edgePoint in enumerate(pictureEdgePositions):
        phi = abs_sobel64f[edgePoint[0], edgePoint[1]]
        if (phi in list(rTable.keys())):
            temp = rTable[phi]
            for r, vector in enumerate(temp):
                x = a - vector[0]
                y = b - vector[1]
                if (x>=0 and x<rows) and (y>=0 and y<columns):
                    cv2.circle(draw,(y, x), 1, (0,0,255), -1)
                    maskingPoints.append((y, x))
        else:
            continue
    cv2.fillConvexPoly(mask, np.int32(maskingPoints), (1.0, 1.0, 1.0), 16, 0)

    # M = cv2.getRotationMatrix2D((columns/angle,rows/angle),theta*angle,1)
    # print(theta*angle)
    # mask = cv2.warpAffine(mask,M,(columns,rows))
    return draw, mask

def hough(picture,template,angle=2):
    pictureCanny = cv2.Canny(picture, 100, 200)
    templateCanny = cv2.Canny(template, 100, 200)
    template_rTable = rTableWithRotation(templateCanny,angle)
    picture_accumulatorTable = accumulatorTable(pictureCanny, template_rTable,angle)
    theta, a, b = np.unravel_index(picture_accumulatorTable.argmax(),picture_accumulatorTable.shape)
    draw, mask = reconstruction(template_rTable, theta, a, b, pictureCanny,angle)
    return mask, draw

if __name__ ==  "__main__":
    
    img_path = sys.argv[1]
    temp_path = sys.argv[2]
    img = load_img(img_path,1)
    template = load_img(temp_path,1)
    angle = 2
    mask, draw = hough(img,template, angle)
