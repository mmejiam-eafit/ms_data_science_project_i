import cv2 # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_ROOT = "../datasets"
DATASET_FACES94 = DATASET_ROOT + "/faces94"
DATASET_FACES94_MALE = DATASET_FACES94 + "/male"
DATASET_FACES94_FEMALE = DATASET_FACES94 + "/female"
DATASET_FACES94_MALESTAFF = DATASET_FACES94 + "/malestaff"
DATASET_FACES95 = DATASET_ROOT + "/faces95"
DATASET_FACES96 = DATASET_ROOT + "/faces96"
DATASET_GRIMACE = DATASET_ROOT + "/grimace"
DATASET_LANDSCAPE = DATASET_ROOT + "/naturalLandscapes"

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
        distances=np.append(np.arange(1, 4), np.inf), 
        outlier_percentage=0.05
    ):
    return_dict = {}
    
    distances_norms = []
    Np, height, width = array_images.shape
    for i in distances:
        distance = np.linalg.norm(np.subtract(base_image, array_images).reshape(Np, height*width), ord=i, axis=1)
        distances_norms.append(distance)
    return_dict["norms"] = pd.DataFrame(np.array(distances_norms).T, columns=["Norm" + str(i) for i in distances])
    outliers_indices = []

    for column in return_dict["norms"].columns:
        outliers = np.argwhere(
                return_dict["norms"][column] >= return_dict["norms"][column].quantile(1 - outlier_percentage)
        )
        outliers_indices.append(outliers)
    outliers_arr = np.array(outliers_indices)
    print(outliers_arr.shape)

    return_dict["outliers"] = pd.DataFrame(
            outliers_arr.reshape(outliers_arr.shape[-3], outliers_arr.shape[-1]*outliers_arr.shape[-2]), 
            columns=["outliers" + i for i in return_dict["norms"].columns]
    )
    
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
            distance_dict["outliers_df"]["outliers" + column],
            distance_dict["norms"][column][distance_dict["outliers_df"]["outliers" + column]],
            s=10, c='r', marker="o", label=column + ' Outliers'
        )
        
        plt.legend(loc='upper left');
        plt.show()
        
 # =======


def getNormsAndDistanceInfoFromBaseImage_1(
        base_image, 
        array_images, 
        distances=np.append(np.arange(1, 4), np.inf), 
        outlier_percentage=0.05
    ):
    return_dict = {}
    
    distances_norms = []
    Np, height, width = array_images.shape
    for i in distances:
        distance = np.linalg.norm(np.subtract(base_image, array_images).reshape(Np, height*width), ord=i, axis=1)
        distances_norms.append(distance)
    return_dict["norms"] = pd.DataFrame(np.array(distances_norms).T, columns=["Norm" + str(i) for i in distances])
    outliers_indices = []
    outliers_col = []

    for column in return_dict["norms"].columns:
        outliers_col = np.where(
            return_dict["norms"][column] >= return_dict["norms"][column].quantile(1 - outlier_percentage),'r','k')
        return_dict["outliers_col"+column]=outliers_col
        
        outliers = np.argwhere(
                return_dict["norms"][column] >= return_dict["norms"][column].quantile(1 - outlier_percentage)
        )
        # save as dataframe with different size, not all distances detect the same number of outliers
        return_dict["outliers"+column]=outliers 
    
    return return_dict


def visualizeOutlierInfo_1(distance_dict):
    for column in distance_dict['norms'].columns:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.scatter(np.arange(distance_dict['norms'][column].shape[0]), distance_dict['norms'][column],
                 c=distance_dict["outliers_col"+column], s=20, linewidth=0,marker="o",label=column + ' Outliers')
        
        plt.legend(loc='upper left');
        plt.show()

