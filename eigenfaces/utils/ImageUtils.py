import cv2 # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

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
        outlier_percentage=0.10
    ):
    return_dict = {}
    outliers_dict = {}
    outliers_dict_iqr = {}
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

        Li=return_dict["norms"][column].quantile(0.25)-1.5*(return_dict["norms"][column].quantile(0.75)-return_dict["norms"][column].quantile(0.25))
        Ls=return_dict["norms"][column].quantile(0.75)+1.5*(return_dict["norms"][column].quantile(0.75)-return_dict["norms"][column].quantile(0.25))
        outliersiqr = np.argwhere(
                (return_dict["norms"][column] < Li) | (return_dict["norms"][column] > Ls)
        )
        outliers_dict_iqr[column] = {'indices': np.squeeze(outliersiqr)}
        
        
    if labels.size != 0:
        false_observations_dict = {}
        for column in return_dict["norms"].columns:
            counter_false = Counter(labels[outliers_dict[column]["indices"]])
            false_observations_dict[column] = {"true_negatives": counter_false[0.0], "false_negatives": counter_false[1.0]}
        
        return_dict["falsitude_metrics"] = false_observations_dict
            
    return_dict["outliers"] = outliers_dict
    
    if labels.size != 0:
        false_observations_dict_iqr = {}
        for column in return_dict["norms"].columns:
            counter_false = Counter(labels[outliers_dict_iqr[column]["indices"]])
            false_observations_dict_iqr[column] = {"true_negatives_iqr": counter_false[0.0], "false_negatives_iqr": counter_false[1.0]}
        
        return_dict["falsitude_metrics_iqr"] = false_observations_dict_iqr
            
    return_dict["outliersiqr"] = outliers_dict_iqr
    
    return return_dict

def visualizeOutlierInfo(distance_dict,labels):
    for column in distance_dict['norms'].columns:
        labels2=np.ones(labels.shape[0])
        labels2[distance_dict["outliers"][column]['indices']]=0
        
        fig = plt.figure(figsize=(15,4))
        ax1 = fig.add_subplot(1,2,1)
        
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

        plt.subplot(1,2,2)
        plt.title("Heatmap "+str(column))
        data = {'y_Predicted': labels2,'y_Actual': labels}
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        ax=sns.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='.0f');
        ax.invert_yaxis()
        ax.invert_xaxis()
        
def visualizeOutlierInfo2(distance_dict,dataset,labels):
    
    for column in distance_dict['norms'].columns:
        labels2=np.ones(labels.shape[0])
        labels2[distance_dict["outliersiqr"][column]['indices']]=0
        
        plt.figure(figsize=(15,4))
        plt.subplot(1,2,1)
        plt.title('Histogram '+str(column))
        plt.grid(True)
        plt.hist(distance_dict['norms'][column]);
        plt.subplot(1,2,2)
        plt.title('Boxplot '+str(column))
        plt.boxplot(distance_dict['norms'][column], 0, 'rs', 0);
        plt.show()

        Distance=distance_dict["norms"][column][distance_dict["outliersiqr"][column]['indices']]
        Ind=distance_dict["outliersiqr"][column]['indices']
        Distance, Ind =zip(*sorted(zip(Distance, Ind)))
        fig = plt.figure(figsize=(14,16))
        ax1 = fig.add_subplot(1,3,1)
        plt.title("last "+str(column))
        ax1.imshow(dataset[Ind[-1]], plt.cm.gray)
        ax2 = fig.add_subplot(1,3,2)
        plt.title("second-to-last "+str(column))
        ax2.imshow(dataset[Ind[-2]], plt.cm.gray)
        ax3 = fig.add_subplot(1,3,3)
        plt.title("third-to-last "+str(column))
        ax3.imshow(dataset[Ind[-3]], plt.cm.gray)

        plt.figure()
        plt.title("Heatmap "+str(column))
        data = {'y_Predicted': labels2,'y_Actual': labels}
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        ax=sns.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='.0f');
        ax.invert_yaxis()
        ax.invert_xaxis()

 # ============

#==== Check parametric distribution of distances
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats

def check_parametricDistribu_distances(y):
    x = np.arange(len(y))
    size = len(y)
    y_df = pd.DataFrame(y)
    sc=StandardScaler() 
    sc.fit(y_df)
    y_std =sc.transform(y_df)
    y_std = y_std.flatten()
    
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'weibull_min', 
              'weibull_max',
             'chi2']
    
    # Set up empty lists to stroe results
    chi_square = []
    p_values = []
    
    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0,100,51)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    
    # Loop through candidate distributions
    
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)
        
        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],scale=param[-1])
        expected_frequency = []
        
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
            
       # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)
        
    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)
    
    # Report results

    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)

