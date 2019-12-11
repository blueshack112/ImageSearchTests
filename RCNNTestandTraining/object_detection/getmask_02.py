import cv2
import pandas as pd
import numpy as np
import pdb
# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.


finalmask = cv2.imread("temp/pil.png")
finalmask = cv2.cvtColor(finalmask,cv2.COLOR_RGB2GRAY)
# Actual detection.
print ("Total: " + str(finalmask.shape[1]))
h, w = finalmask.shape[0], finalmask.shape[1]
line = [[0 for x in range(w)] for y in range(h)]

for i in range(0, finalmask.shape[0]):
    for j in range(0, finalmask.shape[1]):
        line[i][j] = str(int(finalmask[i,j]))

#del line[0]
#print (line.shape[0])
#print (line.shape[1])
line = pd.DataFrame(line)
#line.to_csv("temp/line.csv")
line = line.transpose()
finalLine = line

for i in range(0, line.shape[0]-1):
    j = line.iloc[i].isin(["102"]).any()   
    if j == False:
        finalLine = finalLine.drop(i)
        continue

line = finalLine.transpose()
finalLine = line
for i in range(0, line.shape[0]-1):
    j = line.iloc[i].isin(["102"]).any()   
    if j == False:
        finalLine = finalLine.drop(i)
        continue
print("Writing files")

finalLine.to_csv("temp/line.csv", header=False)
image = finalLine.to_numpy(dtype=np.uint8)
cv2.imwrite("temp/123.png",image)
print("One inference done, file written")
