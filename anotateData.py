import cv2
import os

datasetPath = "Datasets/ShirtData/Data/Tshirts/Batch2/B2nopeople"
os.chdir(datasetPath)

# String suffixes for xml file
fileStart = "<annotation>" + "\n"
folderTag = "\t<folder>B2nopeople</folder>" + "\n"
filenameTagFormat = "\t<filename>{}</filename>" + "\n"
pathTagFormat = "\t<path>E:\\Hassan Work\\CNN Computer Vision\\ImageSearchTests\\Datasets\\ShirtData\\Data\\Tshirts\\Batch2\\B2nopeople\\{}</path>" + "\n"
sourceTag = "\t<source>\n\t\t<database>Unknown</database>\n\t</source>" + "\n"
sizeTagFormat = "\t<size>\n\t\t<width>{}</width>\n\t\t<height>{}</height>\n\t\t<depth>3</depth>\n\t</size>" # Width is shape[1] and height is shape[0] and depth is 3 + "\n"
segmentedTag = "\t<segmented>0</segmented>" + "\n"
objectStart = "\t<object>" + "\n"
namePoseTruncatedDifficultTag = "\t\t<name>teeshirt</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>1</truncated>\n\t\t<difficult>0</difficult>" + "\n"
bndboxStart = "\t\t<bndbox>" + "\n"
boxCoordinatesTagFormat = "\t\t\t<xmin>8</xmin>\n\t\t\t<ymin>8</ymin>\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>\n\t\t\t" + "\n"
bndboxEnd = "\t\t</bndbox>" + "\n"
objectEnd = "\t</object>" + "\n"
fileEnd = "</annotation>"


# Get all images
temp = os.listdir()
images = []
for i in temp:
    if i.endswith(".jpg") or i.endswith(".jpeg"):
        images.append(i)
temp = None

#Opening an image and reading its starting and ending values
for i in images:
    # Loading image
    tempImage = cv2.imread(i)

    # Calculating width and height
    width = tempImage.shape[1]
    height = tempImage.shape[0]
    ymax = tempImage.shape[0] - 8
    xmax = tempImage.shape[1] - 8

    # Determining name of the image and xml
    imageName = i
    xmlName = imageName[:imageName.index(".")] + ".xml"

    # Opening xml file
    xml = open (xmlName, 'w')
    
    # Writing xml contents
    xml.write(fileStart)
    xml.write(folderTag)
    xml.write(filenameTagFormat.format(i))
    xml.write(pathTagFormat.format(i))
    xml.write(sourceTag)
    xml.write(sizeTagFormat.format(str(width), str(height)))
    xml.write(segmentedTag)
    xml.write(objectStart)
    xml.write(namePoseTruncatedDifficultTag)
    xml.write(bndboxStart)
    xml.write(boxCoordinatesTagFormat.format(str(xmax), str(ymax)))
    xml.write(bndboxEnd)
    xml.write(objectEnd)
    xml.write(fileEnd)

    # Safely close the file
    xml.close()

    print("Done: " + i, end='\r')
    
    
    
