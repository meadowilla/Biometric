# Import necessary libraries and modules
import numpy as np
import cv2
import os
from IrisEnhancement import IrisEnhancement
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from FeatureExtraction import FeatureExtraction
from HammingDistance import find_min_hamming_distance

n = 3 # number of users to loop through

rootpath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/CASIA Iris Image Database (version 1.0)/"

for i in range(1,4): # loop through 3 first users in dataset
    # Define paths
    filespath = rootpath + str(i).zfill(3)
    trainpath = filespath + "/1/"

    iriscodepath = "D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/IrisCodesForApp/"
    if not os.path.exists(iriscodepath):
        os.makedirs(iriscodepath)

    iriscodeFilesPath = iriscodepath + str(i).zfill(3)
    if not os.path.exists(iriscodeFilesPath):
        os.makedirs(iriscodeFilesPath)

    iriscodeTrainPath = iriscodeFilesPath + "/1/"
    if not os.path.exists(iriscodeTrainPath):
        os.makedirs(iriscodeTrainPath)

    # Loop through all training images of the current user
    for j in range(1,4):
        # Construct the file path for the current training image
        irispath = trainpath + str(i).zfill(3) + "_1_" + str(j) + ".bmp"

        eye = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
        iris, pupil = IrisLocalization(eye)
        normalized = IrisNormalization(eye, pupil, iris)
        enhanced = IrisEnhancement(normalized)

        # block_size = 16
        train_iris_code = FeatureExtraction(enhanced, 16)
        np.savetxt(os.path.join(iriscodeTrainPath, str(i).zfill(3) + "_1_" + str(j) + ".txt"), train_iris_code, fmt='%d')
        

# Test the model with a new image (image 1 of user 8)
testpath = rootpath + "008/2/"
testirispath = testpath + "008_2_1.bmp"
testeye = cv2.imread(testirispath, cv2.IMREAD_GRAYSCALE)
testiris, testpupil = IrisLocalization(testeye)
testnormalized = IrisNormalization(testeye, testpupil, testiris)
testenhanced = IrisEnhancement(testnormalized)
test_feature = FeatureExtraction(testenhanced, 16)
test_class = 8

# Calculate the Hamming distance between the test image and the first training image of user 1
train_feature = np.loadtxt("D:/at school/2024.1/Biometric Authentication System/Project/Biometric/IrisRecognitionModel/IrisCodesForApp/001/1/001_1_1.txt", dtype=int)
min_distance, best_shift = find_min_hamming_distance(train_feature.tolist(), test_feature.tolist())
print(min_distance, best_shift)

# Calculate the Hamming distance between the test image and all training images
found = False
for i in range(1, 4):
    iriscodeTrainPath = iriscodepath + str(i).zfill(3) + "/1/"
    for j in range(1, 4):
        train_feature = np.loadtxt(iriscodeTrainPath + str(i).zfill(3) + "_1_" + str(j) + ".txt", dtype=int)
        min_distance, best_shift = find_min_hamming_distance(train_feature.tolist(), test_feature.tolist())
        if (min_distance <= 0.32):
            found = True
            print("The test image is matched with the training image of user" + str(i // 3 + 1))
            print("The prediction is: ", (i // 3 + 1) == test_class)

if not found:
    print("The test image is not matched with any training images")
    print("The prediction is: ", test_class in [1, 2, 3])

