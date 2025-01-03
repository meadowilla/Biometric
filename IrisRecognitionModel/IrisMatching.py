from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np

def CalcTest(train_features, train_classes, test_sample, test_class, dist):
    dists = np.zeros(len(train_classes))
    distsm = []
    distsn = []
    offset = np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    for i in range(len(train_classes)):
        if dist == 1:
            distoff = np.ones(11)
            for j in range(len(offset)):
                distoff[j] = distance.cityblock(train_features[i,:],np.roll(test_sample, offset[j]))
            dists[i] = np.min(distoff)
            #dists[i] = distance.cityblock(train_features[i,:],test_sample)
            if train_classes[i] == test_class:
                distsm.append(dists[i])
            else:
                distsn.append(dists[i])
        if dist == 2:
            distoff = np.ones(11)
            for j in range(len(offset)):
                distoff[j] = distance.euclidean(train_features[i,:],np.roll(test_sample, offset[j]))
            dists[i] = np.min(distoff)
            if train_classes[i] == test_class:
                distsm.append(dists[i])
            else:
                distsn.append(dists[i])
        if dist == 3:
            distoff = np.ones(11)
            for j in range(len(offset)):
                distoff[j] = distance.cosine(train_features[i,:],np.roll(test_sample, offset[j]))
            dists[i] = np.min(distoff)
            #dists[i] = distance.cosine(train_features[i,:],test_sample)
            if train_classes[i] == test_class:
                distsm.append(dists[i])
            else:
                distsn.append(dists[i])
    sample_class = train_classes[np.argmin(dists)]
    return sample_class, distsm, distsn

def IrisMatching(train_features, train_classes, test_features, test_classes, dist):
    
    total = float(len(test_classes))
    num = 0.0
    distancesm = []
    distancesn = []

    for i in range(len(test_classes)):
        test_class, distsm, distsn = CalcTest(train_features, train_classes, test_features[i,:], test_classes[i], dist)
        distancesm.extend(distsm)
        distancesn.extend(distsn)
        if test_class == test_classes[i]:
            num += 1.0
    crr = num/total
      
    return crr, distancesm, distancesn

def IrisMatchingRed(train_features, train_classes, test_features, test_classes, n):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))
    if n < 108:
        lda = LinearDiscriminantAnalysis(n_components = n)
        lda.fit(train_features,train_classes)
        train_redfeatures = lda.transform(train_features)
        test_redfeatures = lda.transform(test_features)
    if n >= 108 and n < 323:
        lle = LocallyLinearEmbedding(n_neighbors=n+1,n_components=n)
        lle.fit(train_features)
        train_redfeatures = lle.transform(train_features)
        test_redfeatures = lle.transform(test_features)
    
    l1knn = KNeighborsClassifier(n_neighbors = 1, metric = 'l1')
    l1knn.fit(train_redfeatures, train_classes)
    l1classes = l1knn.predict(test_redfeatures)
    l1crr = float(np.sum(l1classes == test_classes))/total
    
    l2knn = KNeighborsClassifier(n_neighbors = 1, metric = 'l2')
    l2knn.fit(train_redfeatures, train_classes)
    l2classes = l2knn.predict(test_redfeatures)
    l2crr = float(np.sum(l2classes==test_classes))/total
    
    cosknn = KNeighborsClassifier(n_neighbors = 1, metric = 'cosine')
    cosknn.fit(train_redfeatures, train_classes)
    cosclasses = cosknn.predict(test_redfeatures)
    coscrr = float(np.sum(cosclasses==test_classes))/total
    #table_CRR()
    return l1crr, l2crr, coscrr

def calcROC(distancesm,distancesn,thresholds):
    distancesm = np.array(distancesm)
    distancesn = np.array(distancesn)
    numm = float(len(distancesm))
    numn = float(len(distancesn))
    #thresholds = [0.04,0.043,0.046,0.049,0.052,0.055,0.058,0.061,0.064,0.067,0.07,0.073,0.076,0.079,0.082,0.085,0.088,0.091,0.094,0.097,0.1,0.103,0.106,0.109]
    fmrs = []
    fnmrs = []

    for t in thresholds:
        fm = 0.0
        fnm = 0.0
        for dm in distancesm:
            if dm > t:
                fnm += 1.0
        for dn in distancesn:
            if dn < t:
                fm += 1.0
        
        fnmr = fnm/numm
        fmr = fm/numn
        
        fmrs.append(fmr)
        fnmrs.append(fnmr)
    #fmr_fnmr(fmr,fnmr)
    return fmrs, fnmrs #two list
