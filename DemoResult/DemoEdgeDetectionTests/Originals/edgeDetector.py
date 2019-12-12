import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pdb

def isMoreWhite(edges, row, col, colMax):
    for i in range (col, colMax):
        if edges[row][i] == 255:
            return True
    return False

# Funciton for finding last index of white
def findLastIndexOfWhite(edges, row, col, colMax):
    lastIndex = 0
    for i in range (col, colMax):
        if edges[row][i] == 255:
            lastIndex = i
    return (lastIndex)

def findLastIndexOfUsefulBlack(edges, row, col, colMax, contourthickness):
    lastIndex = 0
    for i in range (col, colMax):
        if edges[row][i] == 255:
            lastIndex = i
            if not isMoreWhite(edges, row, i, colMax):
                break
    return lastIndex-contourthickness

def ifLineHasWhite(edges, row, col = 0):
    colMax = edges.shape[1]
    whereFound = []
    for i in range (col, colMax):
        if edges[row][i] == 255:
            whereFound.append(i)
    return whereFound

# Resize while maintaining the ratio
def resizeImage(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original Image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def ifLineHas (edges, row, col=0, returnSoon = False, condition=230):
    for i in range(0, edges.shape[1]):
        if edges[row][i][0] == condition or edges[row][i][2] == condition or edges[row][i][2] == condition:
            print (edges[row][i])
        #print (edges[row][i])

tempfiles = os.listdir("Originals/")
files = []
for eachfile in tempfiles:
    if eachfile.endswith(".jpg") or eachfile.endswith(".jpeg") or eachfile.endswith(".png"):
        files.append(eachfile)
print(files)


for eachfile in files:
    image = cv2.imread("Originals/" + eachfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resizeImage(image, height=500)

    blur = cv2.bilateralFilter(image,27,25,25)
    edges = cv2.Canny(blur, 27, 27)

    """
    # Normalize white within the boundary
    for i in range (0, edges.shape[0]):
        firstwhitefound = False
        lastIndex = 0
        for j in range (0, edges.shape[1]):
            # If first white pixel is found, find out the last index inthe row where white is present
            if edges[i][j] == 255 and not firstwhitefound:
                firstwhitefound = True
                lastIndex = findLastIndexOfWhite(edges, i, j, edges.shape[1])
                        
            # Convert to white as long as the line is within range
            if not edges[i][j] == 255 and firstwhitefound and j <=lastIndex:
                edges[i][j] = 255
    """
    # transpose and do the same
    edges = edges.transpose()
    for i in range (0, edges.shape[0]):
        firstwhitefound = False
        lastIndex = 0
        for j in range (0, edges.shape[1]):
            # If first white pixel is found, find out the last index inthe row where white is present
            if edges[i][j] == 255 and not firstwhitefound:
                firstwhitefound = True
                lastIndex = findLastIndexOfWhite(edges, i, j, edges.shape[1])
                        
            # Convert to white as long as the line is within range
            if not edges[i][j] == 255 and firstwhitefound and j <=lastIndex:
                edges[i][j] = 255

    edges = edges.transpose()
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Extracting max area
    areas = []
    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    # Setting up final contour
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    finalContour = contours[max_area_index]
    edges = np.zeros((edges.shape), np.uint8)
    contourthickness = 6
    

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGRA)
    cv2.drawContours(edges, [finalContour], 0, (0, 0, 230), contourthickness, maxLevel = 0)

    for i in range (0, edges.shape[0]):
       for j in range (0, edges.shape[1]):
            if edges[i][j][0] == 0 and edges[i][j][1] == 0  and edges[i][j][2] == 0:
               edges[i][j][3] = 0
            else:
                edges[i][j][3] = 255

    numpy_horizontal_concat = np.concatenate((blur, edges), axis=1)
    newImageName = eachfile[:eachfile.index(".")] + "stepped.png"
    cv2.imwrite(newImageName, numpy_horizontal_concat)
    print ("Saved file: " + newImageName)
    

# Older code. Might be useful later
"""
    # get contours
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Test", closing)
    cv2.waitKey(0)
    # Extracting max area
    areas = []
    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    # Setting up final contour
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    finalContour = contours[max_area_index]
    finalImage = np.empty((imageGrey.shape))
    cv2.drawContours(finalImage, [finalContour], 0, (255, 255, 255), 10, maxLevel = 0)
    finalImage = (255-finalImage)


    # Just checking
    numpy_horizontal_concat = np.concatenate((imageGrey, closing), axis=1)
    numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, finalImage), axis=1)
    newImageName = eachfile[:eachfile.index(".")] + "stepped.png"    
    cv2.imwrite(newImageName, numpy_horizontal_concat)

"""
"""
    closing1 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image
    
    print (type(closing))    
    # Work with pixels
    for j in range (5, closing.shape[0]-5):
        blackfound = False
        whitefound = False
        blackOutlineDone = False
        for k in range (5, closing.shape[1]-5):
            # If found black
            if closing[j][k] == 0:
                blackfound = True
                whitefound = False
            else:
                whitefound = True
                blackfound = False
            
            # If black is found
            if blackfound:
                if closing[j][k-5] == 0:
                    blackOutlineDone = True
                elif closing[j][k+5] == 255:
                    blackOutlineDone = False
                if blackOutlineDone:
                    closing[j][k] = 255
        for k in range (5, closing.shape[1]-5):
            continue

    
    # To show next to each other
    #print (image.shape)
    #print (grey_3_channel.shape)
    grey_3_channel1 = cv2.cvtColor(closing1, cv2.COLOR_GRAY2BGR)
    grey_3_channel2 = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    numpy_horizontal_concat = np.concatenate((image, grey_3_channel1), axis=1)
    numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, grey_3_channel2), axis=1)
    
    # Writing image
    newImageName = eachfile[:eachfile.index(".")] + "stepped.png"
    cv2.imwrite(newImageName, numpy_horizontal_concat)
"""
