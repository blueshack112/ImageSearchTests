# Imports
import cv2
import json
import os
#For time evaluation
import time
current_milli_time = lambda: int(round(time.time() * 1000))

""" All paths """
# Json file
jsonpath = os.getcwd()
jsonpath = os.path.join(jsonpath, 'Datasets')
jsonpath = os.path.join(jsonpath, 'via-2.0.8')
jsonpath = os.path.join(jsonpath, 'data')
jsonpath = os.path.join(jsonpath, 'test.json')


# Image directory
imagedir = os.getcwd()
imagedir = os.path.join(imagedir, 'Datasets')
imagedir = os.path.join(imagedir, 'via-2.0.8')
imagedir = os.path.join(imagedir, 'data')
imagedir = os.path.join(imagedir, 'test')

_via_img_metadata = '_via_img_metadata'
regions = 'regions'
shape_attributes = 'shape_attributes'
filename = 'filename'
all_points_x = 'all_points_x'
all_points_y = 'all_points_y'

""" Read the json file """
# Check if path is present
if not os.path.exists(jsonpath):
    print("Path not found!")
    exit()
# Load json file
jsonfile = json.load(open(jsonpath))

# Load image data from the json file
imagesData = jsonfile[_via_img_metadata]


""" Set the first regions to all regions """
""" This code is no longer needed
firstTime = True
for index in imagesData:
    # Get image data from index
    imagedata = imagesData[index]
    
    # Get the region of first image data (As it is properly structured)
    if firstTime:
        firstregion = imagedata['regions']
        firstTime = False
    
    # set the current image data's region to the first region
    imagedata['regions'] = firstregion
    imagesData[index] = imagedata

jsonfile['_via_img_metadata'] = imagesData

with open(jsonpath, 'w') as outfile:
    json.dump(jsonfile, outfile)
exit()
"""

""" Start reading and processing image """
# Take one instance of the array you now have received
index = '0.jpg523408'
counter = 0
for index in imagesData:
    imageData = imagesData[index]
    # Find image path and load image
    imagepath = os.path.join(imagedir, imageData[filename])

    # Load image as array
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # New x and y points
    horxes1 = []
    horyes1 = []
    horxes2 = []
    horyes2 = []
    verxes1 = []
    veryes1 = []
    verxes2 = []
    veryes2 = []

    # clear the old list and create template
    imageData[regions].clear()
    newRegion = []

    time_to_run = current_milli_time()

    """ Getting horizontal points (left to right) """
    firstHorizontalLefttoRightColumn = None
    for row in range (0, image.shape[0], 10):
        # For every row in the array
        for column in range(0, image.shape[1]):
            # If a pixel is found that is not white (255), record that x and y value
            # And break the column loop
            if not image[row][column] == 255:
                horxes1.append(column)
                horyes1.append(row)
                if not firstHorizontalLefttoRightColumn:
                    firstHorizontalLefttoRightColumn = column
                else:
                    lastHorizontalLefttoRightColumn = column
                break

    """ Getting horizontal points (left to right) """
    firstHorizontalRighttoLeftColumn = None
    for row in range (image.shape[0]-1 , 0 , -10):
        for column in range (image.shape[1]-1, 0, -1):
            if not image[row][column] == 255:
                horxes2.append(column)
                horyes2.append(row)
                if not firstHorizontalRighttoLeftColumn:
                    firstHorizontalRighttoLeftColumn = column
                else:
                    lastHorizontalRighttoLeftColumn = column
                break
    
    """ Getting vertical points (down to up) """
    for column in range (lastHorizontalLefttoRightColumn, firstHorizontalRighttoLeftColumn, 10):
        # For every row in the array
        for row in range(image.shape[0]-1, 0, -1):
            # If a pixel is found that is not white (255), record that x and y value
            # And break the column loop
            if not image[row][column] == 255:
                verxes1.append(column)
                veryes1.append(row)
                break

    """ Getting vertical points (up to down) """
    for column in range (lastHorizontalRighttoLeftColumn, firstHorizontalLefttoRightColumn, -10):
        # For every row in the array
        for row in range(0, image.shape[0]):
            # If a pixel is found that is not white (255), record that x and y value
            # And break the column loop
            if not image[row][column] == 255:
                verxes2.append(column)
                veryes2.append(row)
                break

    # Finalizing all xes and yes
    finalX = horxes1 + verxes1 + horxes2 + verxes2
    finalY = horyes1 + veryes1 + horyes2 + veryes2
    
    # Creating the region and saving it in the json element
    fiveregion = {}
    fiveregion['region_attributes'] = {"teeshirt": "1"}
    fiveshapeattribute = {}
    fiveshapeattribute["name"] = "polyline"
    fiveshapeattribute["all_points_x"] = finalX
    fiveshapeattribute["all_points_y"] = finalY
    fiveregion[shape_attributes] = fiveshapeattribute
    newRegion.append(fiveregion)

    imageData[regions] = newRegion

    time_to_run = current_milli_time() - time_to_run

    # Assign the data to the json arrays and save the parent json
    #TODO: put x and y here
    imagesData[index] = imageData

    counter = counter + 1
    print ("Files processed: " + str(counter), end='\r')
    
# Set the new data to the json file and write it out
jsonfile[_via_img_metadata] = imagesData
with open(jsonpath, 'w') as outfile:
    json.dump(jsonfile, outfile)


# Debug stuff
print ("======================================================")
print ("Done")
print ("======================================================")
exit()