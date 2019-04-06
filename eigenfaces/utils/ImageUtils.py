import cv2 # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

DATASET_ROOT = "../datasets"
DATASET_FACES94 = DATASET_ROOT + "/faces94"
DATASET_FACES94_MALE = DATASET_FACES94 + "/male"
DATASET_FACES94_FEMALE = DATASET_FACES94 + "/female"
DATASET_FACES94_MALESTAFF = DATASET_FACES94 + "/malestaff"
DATASET_FACES95 = DATASET_ROOT + "/faces95"
DATASET_FACES96 = DATASET_ROOT + "/faces96"
DATASET_GRIMACE = DATASET_ROOT + "/grimace"
DATASET_LANDSCAPE = DATASET_ROOT + "/naturalLandscapes"

DISTANCES = np.append(np.arange(1, 4), np.array([np.inf, 2.5, np.sqrt(2)/2]))

def readFaces94MaleFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_MALE, gray)

def readFaces94FemaleFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_FEMALE, gray)

def readFaces94MaleStaffFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_MALESTAFF, gray)

def readFaces94AllFaces(gray=False):
    npMaleFaces = readFaces94MaleFaces(gray)
    npFemaleFaces = readFaces94FemaleFaces(gray)
    npMaleStaffFaces = readFaces94MaleStaffFaces(gray)
    
    return np.concatenate((npMaleFaces, npMaleStaffFaces, npFemaleFaces))

def readImagesFromDataset(datasetDir, gray=False):
    images = []
    directories = glob.glob(datasetDir + "/*")
    for directory in directories:
        images += readImagesFromDirectory(directory, gray)
    
    return np.array(images, dtype="float32")

def readImagesFromDirectory(directory, gray=False, size=(180, 200)):
    images = []
    imageNames = glob.glob(directory + "/*.jpg")
    for imageName in imageNames:
        image = cv2.resize(cv2.imread(imageName), size)
        # Convert to gray in order to reduce the dimensionality of the data set
        # only if stated by the parameter for gray
        images.append(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if gray else image
        )
        
    return images

## ======= Read images of natural landscape

def readLandsCapeImage(gray=False):
    return readImagesFromDirectory(DATASET_LANDSCAPE, gray)


## ======= Calculate distance from image
def getNormsAndDistanceInfoFromBaseImage(
        base_image, 
        array_images,
        labels=np.array([]),
        outlier_percentage=0.05
    ):
    return_dict = {}
    outliers_dict = {}
    distances_norms = []
    Np, height, width = array_images.shape
    for i in DISTANCES:
        distance = np.linalg.norm(np.subtract(base_image, array_images).reshape(Np, height*width), ord=i, axis=1)
        distances_norms.append(distance)
    return_dict["norms"] = pd.DataFrame(np.array(distances_norms).T, columns=["Norm" + str(np.around(i, decimals=2)) for i in DISTANCES])

    for column in return_dict["norms"].columns:
        outliers = np.argwhere(
                return_dict["norms"][column] >= return_dict["norms"][column].quantile(1 - outlier_percentage)
        )
        outliers_dict[column] = {'indices': np.squeeze(outliers)}
        
    if labels.size != 0:
        false_observations_dict = {}
        for column in return_dict["norms"].columns:
            counter_false = Counter(labels[outliers_dict[column]["indices"]])
            false_observations_dict[column] = {"true_negatives": counter_false[0.0], "false_negatives": counter_false[1.0]}
        
        return_dict["falsitude_metrics"] = false_observations_dict
            
    return_dict["outliers"] = outliers_dict
    return return_dict

def visualizeOutlierInfo(distance_dict):
    for column in distance_dict['norms'].columns:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.scatter(
            np.arange(distance_dict['norms'][column].shape[0]),
            distance_dict['norms'][column],
            s=10, c='b', marker="s", label=column + ' Distances'
        )
        
        ax1.scatter(
            distance_dict["outliers"][column]["indices"],
            distance_dict["norms"][column][
                    distance_dict["outliers"][column]["indices"]
                    ],
            s=10, c='r', marker="o", label=column + ' Outliers'
        )
        
        plt.legend(loc='upper left');
        plt.show()
        
 # =======
