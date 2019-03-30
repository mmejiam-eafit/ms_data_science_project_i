import cv2 # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np

DATASET_ROOT = "../datasets"
DATASET_FACES94 = DATASET_ROOT + "/faces94"
DATASET_FACES94_MALE = DATASET_FACES94 + "/male"
DATASET_FACES94_FEMALE = DATASET_FACES94 + "/female"
DATASET_FACES94_MALESTAFF = DATASET_FACES94 + "/malestaff"
DATASET_FACES95 = DATASET_ROOT + "/faces95"
DATASET_FACES96 = DATASET_ROOT + "/faces96"
DATASET_GRIMACE = DATASET_ROOT + "/grimace"

def readFaces94MaleFaces():
    return readImagesFromDataset(DATASET_FACES94_MALE)

def readFaces94FemaleFaces():
    return readImagesFromDataset(DATASET_FACES94_FEMALE)

def readFaces94MaleStaffFaces():
    return readImagesFromDataset(DATASET_FACES94_MALESTAFF)

def readFaces94AllFaces():
    npMaleFaces = readFaces94MaleFaces()
    npFemaleFaces = readFaces94FemaleFaces()
    npMaleStaffFaces = readFaces94MaleStaffFaces()
    
    return np.concatenate((npMaleFaces, npMaleStaffFaces, npFemaleFaces))

def readImagesFromDataset(datasetDir):
    images = []
    directories = glob.glob(datasetDir + "/*")
    for directory in directories:
        images += readImagesFromDirectory(directory)
    
    return np.array(images, dtype="float32")

def readImagesFromDirectory(directory):
    images = []
    imageNames = glob.glob(directory + "/*.jpg")
    for imageName in imageNames:
        image = cv2.imread(imageName)
        # Convert to gray in order to reduce the dimensionality of the  
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(imageGray)
        
    return images