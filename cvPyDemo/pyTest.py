import cv2
from CvPyMat import CvPyMat

test = CvPyMat();

imgPath = "./buri.jpg";

# Testing the conversion of Mat object to Python
# img2 = test.loadImageInCpp_Demo(imgPath);
# cv2.imshow("pYimg", img2);
# cv2.waitKey(0);

# Testing multiorientation person detection written in C++
img = cv2.imread(imgPath);

rotationDegreeStep 	= 5;
visualizeResults 	= True;
confidenceThr		= 0.5;

pathToDetector = "./inriaperson.xml";
pathToOutput   = "./detectorResults.csv";

test.multiRotPersDet(img, rotationDegreeStep, pathToDetector, confidenceThr, pathToOutput, visualizeResults);