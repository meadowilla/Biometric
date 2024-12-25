from IrisMatching import IrisMatching, IrisMatchingRed, calcROC
from tabulate import tabulate #13min+
import matplotlib.pyplot as plt
import numpy as np

# thresholds_2=[0.076,0.085,0.1]
def table_CRR(train_features, train_classes, test_features, test_classes):
    thresholds = np.arange(0.3,0.4,0.01)
    L1_1,_,_  = IrisMatching(train_features, train_classes, test_features, test_classes, 1)
    print("L1_1 = ",L1_1*100)
    L2_1,_,_ = IrisMatching(train_features, train_classes, test_features, test_classes, 2)
    print("L2_1 = ",L2_1*100)
    C_1,distsm,distsn = IrisMatching(train_features, train_classes, test_features, test_classes, 3)
    L1_2,L2_2,C_2=IrisMatchingRed(train_features, train_classes, test_features, test_classes, 105)
    print ("Correct recognition rate (%)")
    print(tabulate([['L1 distance measure',L1_1*100 ,L1_2*100],['L2 distance measure', L2_1*100,L2_2*100], ['Cosine similarity measure', C_1*100,C_2*100]], headers=['Similartiy measure', 'Original feature set',"Reduced feature set"]))
    fmrs, fnmrs = calcROC(distsm,distsn, thresholds)
    plt.plot(fmrs,fnmrs)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non_match Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.show()
    
#table_CRR(train_features, train_classes, test_features, test_classes)

def performance_evaluation(train_features, train_classes, test_features, test_classes):
    n = list(range(40, 201, 20))
    # n = list(range(4, 20, 2))
    cos_crr=[]
    for i in range(len(n)):
        l1crr, l2crr, coscrr=IrisMatchingRed(train_features, train_classes, test_features, test_classes, n[i])
        cos_crr.append(coscrr*100)
    plt.plot(n,cos_crr,marker="*",color='navy')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('figure_10.png')
    plt.show()
#performance_evaluation(train_features, train_classes, test_features, test_classes)

# Function to calculate metrics at a given threshold
def calculate_metrics(hamming_distances, labels, threshold):
    genuine = labels == 1  # Genuine pairs
    impostor = labels == 0  # Impostor pairs
    # Metrics
    false_matches = np.sum(hamming_distances[impostor] <= threshold)
    false_non_matches = np.sum(hamming_distances[genuine] > threshold)
    true_matches = np.sum(hamming_distances[genuine] <= threshold)
    true_non_matches = np.sum(hamming_distances[impostor] > threshold)

    # Calculate rates
    total_genuine = np.sum(genuine)
    total_impostor = np.sum(impostor)

    fmr = false_matches / total_impostor  # False Match Rate
    fnmr = false_non_matches / total_genuine  # False Non-Match Rate
    tpr = true_matches / total_genuine  # True Positive Rate (1 - FNMR)
    fpr = false_matches / total_impostor  # False Positive Rate (same as FMR for binary case)
    
    return fmr, fnmr, tpr, fpr

def calculate_eer(fmr, fnmr, thresholds):
    # Calculate the absolute difference between FMR and FNMR
    differences = np.abs(fmr - fnmr)
    
    # Find the index of the smallest difference
    eer_index = np.argmin(differences)
    
    # Retrieve the EER and the corresponding threshold
    eer = (fmr[eer_index] + fnmr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold