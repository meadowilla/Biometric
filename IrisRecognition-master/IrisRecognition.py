# Import necessary libraries and modules
import numpy as np
import cv2
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
# import IrisMatching as IM
# import PerformanceEvaluation as PE
import datetime

# Define the root path to the dataset
rootpath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognition-master/CASIA Iris Image Database (version 1.0)/"

# Initialize arrays to hold training and testing features and classes
train_features = np.zeros((324,1536)) # there are 324 training images, each represented by a 1536-dimensional feature vector
train_classes = np.zeros(324, dtype = np.uint8) 
test_features = np.zeros((432,1536)) # 108 subjects * 4 images
test_classes = np.zeros(432, dtype = np.uint8)

# Record the start time of the process
starttime = datetime.datetime.now()

# Loop through each subject in the dataset
for i in range(1,109):
    # Define paths for training and testing images for the current subject
    filespath = rootpath + str(i).zfill(3)
    trainpath = filespath + "/1/"
    testpath = filespath + "/2/"

    # Loop through each training image for the current subject
    for j in range(1,4):
        # Construct the file path for the current training image
        irispath = trainpath + str(i).zfill(3) + "_1_" + str(j) + ".bmp"

        # Read the image in grayscale mode
        img = cv2.imread(irispath, 0)

        # Perform iris localization to find the iris and pupil regions
        iris, pupil = IrisLocalization(img)

        # Normalize the iris region
        normalized = IrisNormalization(img, pupil, iris)

        # Enhance the normalized iris image
        ROI = ImageEnhancement(normalized)

        iris_code = FeatureExtraction(ROI)
        # print("iris code shape: ", iris_code.shape)
        template_storage = np.zeros((48, 4096))
        template_storage[i-1] = iris_code
        print(f"User {i-1} enrolled successfully.")
        # Extract features from the enhanced image (not shown in the provided code)
        train_features[(i-1)*3+j-1, :] = iris_code
        train_classes[(i-1)*3+j-1] = i
    for k in range(1,5):
        irispath = testpath + str(i).zfill(3) + "_2_" + str(k) + ".bmp"
        img = cv2.imread(irispath, 0)
        iris, pupil = IrisLocalization(img)
        normalized = IrisNormalization(img, pupil, iris)
        ROI = ImageEnhancement(normalized)
        irisCode = FeatureExtraction(ROI)
        print("iris code shape: ", irisCode.shape)
        test_features[(i-1)*4+k-1, :] = irisCode
        test_classes[(i-1)*4+k-1] = i

endtime = datetime.datetime.now()

print('image processing and feature extraction takes '+str((endtime-starttime).seconds)+' seconds')


# PE.table_CRR(train_features, train_classes, test_features, test_classes)
# PE.performance_evaluation(train_features, train_classes, test_features, test_classes)
# #thresholds_2=[0.74,0.76,0.78]


# # this part is for bootsrap
# starttime = datetime.datetime.now() 
# thresholds_3=np.arange(0.6,0.9,0.02)
# times = 10 #running 100 times takes about 1 to 2 hours
# total_fmrs, total_fnmrs, crr_mean, crr_u, crr_l = IM.IrisMatchingBootstrap(train_features, train_classes, test_features, test_classes,times,thresholds_3)
# fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u = IM.calcROCBootstrap(total_fmrs, total_fnmrs)

# endtime = datetime.datetime.now()

# print('Bootsrap takes '+str((endtime-starttime).seconds) + ' seconds')

# fmrs_mean *= 100  #use for percent(%)
# fmrs_l *= 100
# fmrs_u *= 100
# fnmrs_mean *= 100
# fnmrs_l *= 100
# fnmrs_u *= 100
# PE.FM_FNM_table(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u, thresholds_3)
# PE.FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)
# PE.FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)

