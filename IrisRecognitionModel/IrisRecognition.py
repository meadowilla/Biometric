# Import necessary libraries and modules
import numpy as np
import cv2
from IrisEnhancement import IrisEnhancement
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from FeatureExtraction import FeatureExtraction
from HammingDistance import HammingDistance, find_min_hamming_distance
import datetime
import os
import PerformanceEvaluation as PE

# Define the root path to the dataset and normalizedpath to save the normalized images
rootpath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/CASIA Iris Image Database (version 1.0)/"
normalizedpath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/Normalized Images/"
iriscodepath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/Iris Codes/"

# Create the new folder if it doesn't exist
if not os.path.exists(normalizedpath):
    os.makedirs(normalizedpath)
if not os.path.exists(iriscodepath):
    os.makedirs(iriscodepath)

# Initialize arrays to hold training and testing features and classes
train_features = np.zeros((324,1024)) # there are 324 training images, each represented by a 1536-dimensional feature vector
train_classes = np.zeros(324, dtype = np.uint8) 
test_features = np.zeros((432,1024)) # 108 subjects * 4 images
test_classes = np.zeros(432, dtype = np.uint8)

# Record the start time of the process
starttime = datetime.datetime.now()

# Loop through 3 first users in dataset
for i in range(1,4):
    # Define paths
    filespath = rootpath + str(i).zfill(3)
    normalizedFilesPath = normalizedpath + str(i).zfill(3)
    iriscodeFilesPath = iriscodepath + str(i).zfill(3)
    if not os.path.exists(normalizedFilesPath):
        os.makedirs(normalizedFilesPath)
    if not os.path.exists(iriscodeFilesPath):
        os.makedirs(iriscodeFilesPath)

    trainpath = filespath + "/1/"
    testpath = filespath + "/2/"
    normalizedTrainPath = normalizedFilesPath + "/1/"
    normalizedTestPath = normalizedFilesPath + "/2/"
    iriscodeTrainPath = iriscodeFilesPath + "/1/"
    iriscodeTestPath = iriscodeFilesPath + "/2/"
    if not os.path.exists(normalizedTrainPath):
        os.makedirs(normalizedTrainPath)
    if not os.path.exists(normalizedTestPath):
        os.makedirs(normalizedTestPath)
    if not os.path.exists(iriscodeTrainPath):
        os.makedirs(iriscodeTrainPath)
    if not os.path.exists(iriscodeTestPath):
        os.makedirs(iriscodeTestPath)

    
    # Loop through all training and testing images of the current user
    for j in range(1,4):
        # Construct the file path for the current training image
        irispath = trainpath + str(i).zfill(3) + "_1_" + str(j) + ".bmp"

        eye = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
        iris, pupil = IrisLocalization(eye)
        normalized = IrisNormalization(eye, pupil, iris)
        # cv2.imwrite(os.path.join(normalizedTrainPath, str(i).zfill(3) + "_1_" + str(j) + ".png"), normalized)
        enhanced = IrisEnhancement(normalized)
        cv2.imwrite(os.path.join(normalizedTrainPath, str(i).zfill(3) + "_1_" + str(j) + ".png"), enhanced)

        block_size = 16
        train_iris_code = FeatureExtraction(enhanced, block_size)
        np.savetxt(os.path.join(iriscodeTrainPath, str(i).zfill(3) + "_1_" + str(j) + ".txt"), train_iris_code, fmt='%d')

    for k in range(1,5):
        # Construct the file path for the current training image
        irispath = testpath + str(i).zfill(3) + "_2_" + str(k) + ".bmp"

        eye = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
        iris, pupil = IrisLocalization(eye)
        normalized = IrisNormalization(eye, pupil, iris)
        # cv2.imwrite(os.path.join(normalizedTestPath, str(i).zfill(3) + "_2_" + str(k) + ".png"), normalized)
        enhanced = IrisEnhancement(normalized)
        cv2.imwrite(os.path.join(normalizedTestPath, str(i).zfill(3) + "_2_" + str(k) + ".png"), enhanced)

        block_size = 16
        test_iris_code = FeatureExtraction(normalized, block_size)
        np.savetxt(os.path.join(iriscodeTestPath, str(i).zfill(3) + "_2_" + str(k) + ".txt"), test_iris_code, fmt='%d')
    

    ''' Do the same things as above but without saving the images, just save into temperary arrays
    for j in range(1,4):
        # Construct the file path for the current training image
        irispath = trainpath + str(i).zfill(3) + "_1_" + str(j) + ".bmp"

        eye = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
        iris, pupil = IrisLocalization(eye)
        normalized = IrisNormalization(eye, pupil, iris)
        enhanced = IrisEnhancement(normalized)
        train_features[(i-1)*3+j-1, :] = FeatureExtraction(enhanced, 16)
        train_classes[(i-1)*3+j-1] = i
    
    for k in range(1,5):
        # Construct the file path for the current training image
        irispath = testpath + str(i).zfill(3) + "_2_" + str(k) + ".bmp"

        eye = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
        iris, pupil = IrisLocalization(eye)
        normalized = IrisNormalization(eye, pupil, iris)
        enhanced = IrisEnhancement(normalized)
        test_features[(i-1)*4+k-1, :] = FeatureExtraction(enhanced, 16)
        test_classes[(i-1)*4+k-1] = i
    '''

    ''' This this the code to compare a template of registerd user with a template of test user
    We can modify this code to compare all templates of registerd users (001, 002, 003) with a random template of test user (008)'''
    # train_tmp = np.loadtxt(iriscodeTestPath + str(i).zfill(3) + "_2_1.txt", dtype=int)
    # test_tmp = np.loadtxt(iriscodeTestPath + str(i).zfill(3) + "_2_2.txt", dtype=int)
    # min_distance, best_shift = find_min_hamming_distance(train_tmp.tolist(), test_tmp.tolist())
    # print(f"Test {i}: min_distance = {min_distance}, best_shift = {best_shift}")

    """ After achieving the min distance, we can compare it with a threshold (0.373) to determine if the test image is the same as the registerd image"""

endtime_1 = datetime.datetime.now()
print('image processing and feature extraction takes ' + str((endtime_1-starttime).seconds) + ' seconds')

thresholds = [0.373]
med_max = 0
threshold_max = 0

for threshold in thresholds:
    print("threshold: ", threshold)
    med = 0
    for i in range(1, 4): # 3 first users
        filespath = rootpath + str(i).zfill(3)
        normalizedFilesPath = normalizedpath + str(i).zfill(3)
        iriscodeFilesPath = iriscodepath + str(i).zfill(3)

        trainpath = filespath + "/1/"
        testpath = filespath + "/2/"
        normalizedTrainPath = normalizedFilesPath + "/1/"
        normalizedTestPath = normalizedFilesPath + "/2/"
        iriscodeTrainPath = iriscodeFilesPath + "/1/"
        iriscodeTestPath = iriscodeFilesPath + "/2/"

        count = 0
        for j in range(1, 4):
            for k in range (1, 5):
                train_tmp = np.loadtxt(iriscodeTrainPath + str(i).zfill(3) + "_1_" + str(j) + ".txt", dtype=int)
                test_tmp = np.loadtxt(iriscodeTestPath + str(i).zfill(3) + "_2_" + str(k) + ".txt", dtype=int)
                min_distance, best_shift = find_min_hamming_distance(train_tmp.tolist(), test_tmp.tolist())
                if (min_distance <= threshold):
                    count += 1
        print(f"Test {i}: ", count /12 *100, "%")
        med = med + count

    print(f"overall of threshold {threshold}: ", med /3 /12 *100, "%")  # 3 first users
    # if (med /25 /12 *100 > med_max):
    #     med_max = med /25 /12 *100
    #     threshold_max = threshold

# print(f"max overall: {med_max}%, threshold: {threshold_max}")
endtime_2 = datetime.datetime.now()
print('feature matching and performance evaluation takes '+ str((endtime_2-starttime).seconds) + ' seconds')


''' The below part is not used in the project'''
# PE.table_CRR(train_features, train_classes, test_features, test_classes)
# PE.performance_evaluation(train_features, train_classes, test_features, test_classes)
#thresholds_2=[0.74,0.76,0.78]


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