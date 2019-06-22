import cv2 # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

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

def facespace(percentage_variance,dataset,data,mean_all,u,dataset_N,height,width,start,step,stop,low):

    representation_percentage = np.arange(start = start, stop = stop, step = step)
    sumvar=[]
    numvar=[]

    for i in range(len(representation_percentage)):
        sum_var_plot = 0
        num_var_plot = 0
        for j in np.arange(percentage_variance.shape[0]):
            if sum_var_plot >= representation_percentage[i]:
                num_var_plot = j
                break;
    
            sum_var_plot += percentage_variance[j]
        
        sumvar=np.append(sumvar,sum_var_plot*100)
        numvar=np.append(numvar,num_var_plot)

    Eigenvectors_plot=np.dot(data.T,u[:,0:int(numvar[3])])
    NormEigenvectors_plot = preprocessing.normalize(Eigenvectors_plot,axis=0, norm='l2')

    image_index = np.random.randint(low, high=dataset_N, size=1)[0]
    original_image = dataset[image_index]


    fig = plt.figure(figsize=(20,45))
    ax1 = fig.add_subplot(1,5,1)
    plt.title("Original Image")
    ax1.imshow(original_image, plt.cm.gray)

    for i in range(4):

        example_image = np.dot(np.dot(data[image_index],NormEigenvectors_plot[:,0:int(numvar[i])]),NormEigenvectors_plot[:,0:int(numvar[i])].T)+mean_all.reshape(height*width)
        ax2 = fig.add_subplot(1,5,i+2)
        plt.title("Reconstructed PCAs "+str(int(numvar[i]))+' - '+str(int(sumvar[i]))+'%')
        ax2.imshow(example_image.reshape(height,width), plt.cm.gray)

def specificimage(data,dataset,NormEigenvectorsA,mean_all,N_image,dataset_N,height,width):

    Image=data[N_image]
    w=np.dot(Image,NormEigenvectorsA)#weigth w of each Eigenface in generate subspace
    Reconstructed=np.dot(w,NormEigenvectorsA.T)+mean_all.reshape(height*width)#es mas claro w*vectores propios transpuestos
    example_image = Reconstructed
    original_image = dataset[N_image]
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(1,2,1)
    plt.title("Original Image")
    ax1.imshow(original_image, plt.cm.gray)
    ax2 = fig.add_subplot(1,2,2)
    plt.title("Reconstructed Image")
    ax2.imshow(example_image.reshape(height,width), plt.cm.gray)
    
def randomimage(data,dataset,NormEigenvectorsA,mean_all,dataset_N,height,width):

    image_index = np.random.randint(0, high=dataset_N, size=1)[0]
    example_image = np.dot(np.dot(data[image_index],NormEigenvectorsA),NormEigenvectorsA.T)+mean_all.reshape(height*width)
    original_image = dataset[image_index]
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(1,2,1)
    plt.title("Original Image")
    ax1.imshow(original_image, plt.cm.gray)
    ax2 = fig.add_subplot(1,2,2)
    plt.title("Reconstructed Image")
    ax2.imshow(example_image.reshape(height,width), plt.cm.gray)

def histbox(edistance):    
    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    plt.title('Histogram')
    plt.grid(True)
    plt.hist(edistance);
    plt.subplot(1,2,2)
    plt.title('Boxplot')
    plt.boxplot(edistance, 0, 'rs', 0);
    plt.show()
    
def outlierseigenfaces(edistance,threshold):
    
    z = np.abs(stats.zscore(edistance))
    outliersindex=np.where(z > threshold)
    outliers=edistance[outliersindex]
    zsort=z[outliersindex]
    indexsortout=np.argsort(outliers)
    outliers=outliers[indexsortout]
    zsort=zsort[indexsortout]
    indexsort=np.argsort(edistance) 
    edistancesort=edistance[indexsort] 
    return threshold, outliers, zsort, indexsort, z

def landimages(landscapes,height,width,mean_all,NormEigenvectorsA,ordn,outliers):

    N_land= np.random.randint(0, high=landscapes.shape[0], size=1)[0]
    landimage=landscapes[N_land].reshape(height*width)-mean_all.reshape(height*width)#seleccionar imagen individual
    wland=np.dot(landimage,NormEigenvectorsA)#pesos w de cada Eigenface en subespacio generado
    Reconstland=np.dot(wland,NormEigenvectorsA.T)+mean_all.reshape(height*width)#es mas claro w*vectores propios transpuestos
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(1,2,1)
    plt.title("Land image")
    ax1.imshow(landscapes[N_land], plt.cm.gray)
    ax2 = fig.add_subplot(1,2,2)
    plt.title("Reconstructed land Image")
    ax2.imshow(Reconstland.reshape(height, width), plt.cm.gray)
    edistanceland = np.linalg.norm(np.subtract(landscapes[N_land].reshape(height*width), Reconstland), ord=ordn, axis=0)

    if edistanceland>outliers[0]:
        print('No pertenece al dataset')
    else:
        print('error')
    print(edistanceland)

def kfold(y_true,landscapes,dataset,height,width,ordn):    
    
    accuracy=[]
    tncv=[]
    fpcv=[]
    fncv=[]
    tpcv=[]

    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    kf.get_n_splits(dataset)
    print(kf)  
    KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index] 
        datasetcv_N=X_train.shape[0]
        mean_all_cv = np.mean(X_train.reshape(datasetcv_N, height*width), axis=0).reshape(height, width)
        datacv=X_train.reshape(datasetcv_N, height*width) - np.mean(X_train.reshape(datasetcv_N, height*width), axis=0)
        datasetmeancv=(1/(datasetcv_N-1))*(np.dot(datacv,datacv.T))
        u,s,vh = np.linalg.svd(datasetmeancv)
        representation_percentage = 0.80
        sum_eig = np.sum(s)
        percentage_variance = np.divide(s, sum_eig)
        sum_var = 0
        num_var = 0
        for i in np.arange(percentage_variance.shape[0]):
            if sum_var >= representation_percentage:
                num_var = i
                break;
    
            sum_var += percentage_variance[i]
        num_var_select=num_var    
        Eigenvectors_cv=np.dot(datacv.T,u[:,0:num_var_select])
        NormEigenvectors_cv = preprocessing.normalize(Eigenvectors_cv,axis=0, norm='l2')
        dataReconstructedcv=np.dot(np.dot(datacv,NormEigenvectors_cv),NormEigenvectors_cv.T)+mean_all_cv.reshape(height*width)
        edistancecv = np.linalg.norm(np.subtract(dataReconstructedcv, X_train.reshape(datasetcv_N, height*width)), ord=ordn, axis=1)
        maxcv=np.max(edistancecv)
        X_test = np.vstack((X_test,landscapes))
        y_test=np.append(y_test,np.zeros(landscapes.shape[0]))
        X_test_mean=X_test.reshape(X_test.shape[0],height*width)-mean_all_cv.reshape(height*width)
        dataReconstructedX_test=np.dot(np.dot(X_test_mean,NormEigenvectors_cv),NormEigenvectors_cv.T)+mean_all_cv.reshape(height*width)
        edistanceX_test = np.linalg.norm(np.subtract(dataReconstructedX_test, X_test.reshape(X_test.shape[0], height*width)), ord=ordn, axis=1)
        y_pred=(edistanceX_test<=maxcv)*1
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy=np.append(accuracy,(tp+tn)/(tp+tn+fp+fn))
        tncv=np.append(tncv,tn)
        fpcv=np.append(fpcv,fp)
        fncv=np.append(fncv,fn)
        tpcv=np.append(tpcv,tp)

    print('test: ',X_test.shape[0])
    print('faces: ',test_index.shape[0])
    print('landscapes: ',landscapes.shape[0])

    return accuracy, tncv, fpcv, fncv, tpcv

def kmeansplit(datasetfull,y_label,datasetfull_N,height,width,representation_percentage,k):

    # Create training and test sets
    indices = np.arange(datasetfull_N)
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(datasetfull,y_label,indices,test_size = 0.2, random_state=42)
    datasetcv_N=X_train.shape[0]
    datafull=X_train.reshape(datasetcv_N, height*width) - np.mean(X_train.reshape(datasetcv_N, height*width), axis=0)
    datasetmeanfull=(1/datasetcv_N-1)*(np.dot(datafull,datafull.T))
    u,s,vh = np.linalg.svd(datasetmeanfull)
    sum_eig = np.sum(s)
    percentage_variance = np.divide(s, sum_eig)
    sum_var = 0
    num_var = 0
    for i in np.arange(percentage_variance.shape[0]):
        if sum_var >= representation_percentage:
            num_var = i
            break;   
        sum_var += percentage_variance[i]    
    num_var_select=num_var 
    EigenvectorsAk=np.dot(datafull.T,u[:,0:num_var_select])
    NormEigenvectorsAk = preprocessing.normalize(EigenvectorsAk,axis=0, norm='l2')
    omegaw=np.dot(datafull,NormEigenvectorsAk)
    kmeansk = KMeans(n_clusters=k, random_state=42).fit(omegaw)
    datasetest_N=X_test.shape[0]
    datafulltest=X_test.reshape(datasetest_N, height*width) - np.mean(X_train.reshape(datasetcv_N, height*width), axis=0)
    omegawtest=np.dot(datafulltest,NormEigenvectorsAk)
    y_predk=kmeansk.predict(omegawtest)
    Y=kmeansk.transform(omegawtest)
    
    cols = 3
    rows = 1
    plt.figure(figsize=(12,8))
    for i in np.arange(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.title("Class "+str(i+1))
        plt.imshow((np.dot(kmeansk.cluster_centers_[i],NormEigenvectorsAk.T)+np.mean(X_train.reshape(datasetcv_N, height*width), axis=0)).reshape(height, width), plt.cm.gray)
    
    
    plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    ax.scatter(Y[np.where(y_predk==0),0],Y[np.where(y_predk==0),1] ,Y[np.where(y_predk==0),2], cmap='viridis', linewidth=1);
    ax.scatter(Y[np.where(y_predk==1),0],Y[np.where(y_predk==1),1] ,Y[np.where(y_predk==1),2], cmap='viridis', linewidth=1);
    ax.scatter(Y[np.where(y_predk==2),0],Y[np.where(y_predk==2),1] ,Y[np.where(y_predk==2),2], cmap='viridis', linewidth=1);
    plt.gca().legend(('class 1','class 2','class 3'))
    
    
    plt.figure(figsize=(7,5))
    plt.scatter(Y[np.where(y_predk==0),0], Y[np.where(y_predk==0),1])
    plt.scatter(Y[np.where(y_predk==1),0], Y[np.where(y_predk==1),1])
    plt.scatter(Y[np.where(y_predk==2),0], Y[np.where(y_predk==2),1])
    plt.gca().legend(('class 1','class 2','class 3'))
    plt.xlabel('x')
    plt.ylabel('Y')
    plt.title('centroid transform')
    plt.show()
    
    return kmeansk, NormEigenvectorsAk, X_train, X_test, y_predk
    