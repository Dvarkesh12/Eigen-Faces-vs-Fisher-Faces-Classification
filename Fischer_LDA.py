import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Creating the Pattern matrix
# reading faces of first 36 persons
no_faces = 38
imgs = os.listdir('CroppedYale')
imgs = imgs[0:no_faces]
imgs = ['CroppedYale/'+x for x in imgs]

# pattern (func.)
# Input:1. Dataset (images of persons)
#       2. Percent of dataset of each person to be used for training (fci - start index, fcf - end index of sample images)
#
# Working:
# vectorizes the images using raster scan order and stacks them together in a row to form a pattern matrix.
# Separate pattern matrices are formed for each person and all those matrices are collected in one array.
#
# Returns: An array cointaining pattern matrices for each person

def pattern(fci, fcf):
    # pattern matrix
    i = 0
    pco = []
    while i < len(imgs):
        ind = os.listdir(imgs[i])
        ind = ind[fci:fcf]
        pc = []
        for name in ind:
            img = cv.imread(f'{imgs[i]}/{name}', 0)
            pc.append(np.array(np.ndarray.flatten(img), dtype=float))
        pco.append(pc)
        i += 1
    return np.array(pco)


Pc = pattern(0, 50)     # Train set Pattern matrix
Ts = pattern(50, 64)    # Test set Pattern matrix

# class means
Pcm = []
i = 0
while i < len(Pc):
    Pcm.append(np.mean(Pc[i], axis=0))
    i += 1
Pcm = np.array(Pcm)


# Dimensionality reduction
# dim_red (func.)
# Reduces the dimension of original Pattern Matrix using SVD.
#
# Input - 1. Labelled Train Pattern matrix
#         2. Labelled Test Pattern matrix
#         3. Matrix of class means
#         4. Number of components to kept after dimensionality reduction.
#         (Number of columns of Left singular value matrix to be used.)
#
# Retruns: Dimensionally reduced Test and Train Pattern matrix, class means matrix and normalised reduced pattern matrix


# pc - labelled train pattern matrix
# ts - labelled test pattern matrix
# pcm - class means matrix
# n_coms - dimension of new space

def dim_red(pc, ts, pcm, n_coms):
    p = pc[0]
    i = 1
    while i< len(pc):
        p = np.vstack((p, pc[i]))
        i += 1

    # mean of data
    mn = np.mean(p, axis=0)

    # mean normalization
    pm = p - mn

    # matrix decomposition using SVD
    u1, c1, v1t = np.linalg.svd(pm.T, full_matrices=False)

    # projections of train-pattern matrix under SVD basis
    ic = []
    i = 0
    while i < len(pc):
        ic.append(np.matmul(pc[i], u1[:, 0:n_coms]))
        i += 1
    ic = np.array(ic)

    # projections of test-pattern matrix under SVD basis
    ti = []
    i = 0
    while i < len(ts):
        ti.append(np.matmul(ts[i], u1[:, 0:n_coms]))
        i += 1
    ti = np.array(ti)

    # projections of class means under SVD basis
    im = []
    i = 0
    while i < len(pcm):
        im.append(np.matmul(pcm[i], u1[:, 0:n_coms]))
        i += 1
    im = np.array(im)

    # class mean normalised train-pattern matrix
    npc = []
    i = 0
    while i < len(pcm):
        npc.append(pc[i] - pcm[i])
        i += 1
    npc = np.array(npc)

    # projections of mean normalised train-pattern matrix
    nic = []
    i = 0
    while i < len(npc):
        nic.append(np.matmul(npc[i], u1[:, 0:n_coms]))
        i += 1
    nic = np.array(nic)

    # projections of overall mean
    mnn = np.matmul(mn, u1[:, 0:n_coms])

    return ic, ti, im, nic, mnn

Ic, It, Im, Icm, Im_all = dim_red(Pc, Ts, Pcm, 1000)

#Calculating within class and between class covariance.
# within class covarinace
swc = []
i = 0
while i<len(Ic):
    swc.append(np.matmul( (Ic[i] - Im[i]).T, (Ic[i]-Im[i])))
    i += 1
Sw = np.sum(swc, axis=0)

# between class covariance matrix
Sb = 0
i = 0
while i < len(Im):
    Sb = Sb + Ic.shape[1] * np.matmul((Im[i] - Im_all).reshape(len(Im_all), 1), (Im[i] - Im_all).reshape(1, len(Im_all)))
    i += 1

# Calculating eigen values and eigen vectors of product of Sw^-1 * Sb
ei_val, ei_vec = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
# Selecting first few required number of eigen vectors after sorting eigen values from highest to lowest
vv = np.argsort(ei_val)
wt = []
i = 1
while i < no_faces+1:
    wt.append(np.array(ei_vec[:, vv[-i]], dtype=float))
    i += 1
wt = np.array(wt)

# Projections of training images onto selected eigen vectors or weight vectors
II1 = []
i = 0
while i < len(Ic):
    II1.append(Ic[i] @ wt[0].reshape(len(wt[0]), 1))
    i += 1
II1 = np.array(II1)

II2 = []
i = 0
while i < len(Ic):
    II2.append(Ic[i] @ wt[1].reshape(len(wt[0]), 1))
    i += 1
II2 = np.array(II2)

# Projections of to be tested images onto selected eigen vectors or weight vectors
T1 = []
i = 0
while i < len(It):
    T1.append(It[i] @ wt[0].reshape(len(wt[0]), 1))
    i += 1

T2 = []
i = 0
while i < len(It):
    T2.append(It[i] @ wt[1].reshape(len(wt[1]), 1))
    i += 1

# Some plots showing projection of high dimensional faces onto 2 dimensions using fischer face algorithm
plt.figure(figsize=(15,15))
i = 0
while i < len(Ic):
    plt.scatter(II1[i], II2[i])
    i += 1
plt.show()

# Some plots showing projection of high dimensional faces onto 1 dimension using fischer face algorithm
y = np.ones((50, 1))
plt.figure(figsize=(15,5))
i = 0
while i < no_faces:
    plt.scatter(II1[i], y)
    i += 1
plt.show()

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num

ta = []
i = 0
while i < no_faces:
    xx = np.argsort(Ic[i] @ wt.T, axis=1)
    xx1 = [ll for ll in xx[:, 0]]
    xx2 = [ll for ll in xx[:, -1]]
    mf1 = most_frequent(xx1)
    mf2 = most_frequent(xx2)
    t = 0
    for c in xx1:
        if c == mf1 or c == mf2:
            t += 1
    ta.append(t)
    i += 1

Train_accuracy = sum(ta)/1900
print(f'Training accuracy is {Train_accuracy}*100')
