#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import operator
import numpy as np
import random
import matplotlib.pyplot as plt
from mldata import parse_c45


# In[2]:


def stratCrossValid(data): # stratified 5-fold-validation for both discrete and continuous cases
    subset0 = []
    subset1 = []
    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
    fold5 = []
    for i in range(0, len(data)):
        if  1.0 == data[i,-1]:
            subset1.append(data[i])
        else:
            subset0.append(data[i])
    subset1 = np.array(subset1)
    subset0 = np.array(subset0)
    
    np.random.seed(12345) # set random seed to 12345
    np.random.shuffle(subset1)
    np.random.shuffle(subset0)  
    line1 = int(len(subset1) / 5)
    line2 = int(len(subset0) / 5)
    
    temp1 = subset1[0:line1]
    temp2 = subset0[0:line2]
    fold1 = np.concatenate((temp1,temp2), axis = 0)
    temp3 = subset1[line1:line1 * 2]
    temp4 = subset0[line2:line2 * 2]
    fold2 = np.concatenate((temp3,temp4), axis = 0)
    temp5 = subset1[line1 * 2:line1 * 3]
    temp6 = subset0[line2 * 2:line2 * 3]
    fold3 = np.concatenate((temp5,temp6), axis = 0)
    temp7 = subset1[line1 * 3:line1 * 4]
    temp8 = subset0[line2 * 3:line2 * 4]
    fold4 = np.concatenate((temp7,temp8), axis = 0)
    temp9 = subset1[line1 * 4:len(subset1)]
    temp10 = subset0[line2 * 4:len(subset0)]
    fold5 = np.concatenate((temp9,temp10), axis = 0)
    
    return fold1,fold2,fold3,fold4,fold5

def getLabelFrequency(data):
    posCounts = 0
    negCounts = 0
    if len(data) == 0:
        return 0, 0
    else:
        for i in range(0, len(data)):
            if  1.0 == data[i,-1]:
                posCounts += 1 # number of positive labels
            else:
                negCounts += 1 # number of negative labels
    return negCounts, posCounts

def getLikelihood(binNum, attribute, data): # Note: the input dataset should be discretized first.
    likelihood = []
    for j in range(1, binNum + 1):
        posCounts = 0
        negCounts = 0
        temp = []
        for i in range(0,len(data)):          
            if data[i, attribute] == j and data[i, -1] == 1.0:
                posCounts += 1
            elif data[i, attribute] == j and data[i, -1] == 0.0:
                negCounts += 1
        temp.append(negCounts)
        temp.append(posCounts)
        likelihood.append(temp)
    likelihood = np.array(likelihood)
    return likelihood


def partition(binNum, attribute, data): # helper function of discretize
    output = []
    maxValue = max(data[:,attribute])
    threshold = min(data[:,attribute])
    output.append(threshold)
    
    for i in range(0, binNum):
        if threshold > maxValue:
            break
        else:
            threshold = threshold + maxValue / binNum
            output.append(threshold)
                                        
    return output

def discretize(binNum, attribute, data): # This function should discretize an attribute of a dataset based on the input binNum.
    valueRange = partition(binNum, attribute, data)
    output = np.copy(data)
    
    for i in range(0, len(data)):
        for j in range(1, binNum):
            if data[i, attribute] >= valueRange[j - 1] and data[i, attribute] < valueRange[j]:
                output[i, attribute] = j
    return output

def discretizeAll(binNum, data): # Discretize all attributes of a dataset based on the input binNUm.
    output = np.copy(data)
    for i in range(1, data.shape[1] - 1): # do not include index(the first column) and label(the last column)
        output = discretize(binNum, i, output)      
    return output
        
def mEstimate(mValue, binNum, binIndex, label, attribute, data): # Note: the input dataset should be discretized first.
    n = getLabelFrequency(data)[label]   
    likelihood = getLikelihood(binNum, attribute, data)
    nc = likelihood[binIndex,label]
    p = 1 / binNum
    mEstimate = (nc + mValue * p) / (n + mValue)
    return mEstimate

def LaplaceSmoothing(binNum, binIndex, label, attribute, data): # Note: the input dataset should be discretized first.
    a = 1
    n = getLabelFrequency(data)[label]   
    likelihood = getLikelihood(binNum, attribute, data)
    x = likelihood[binIndex,label]
    LS = (x + a) / (n + binNum * a)
    return LS
    
def naiveBayes(mValue, binNum, data):
    dataset = discretizeAll(binNum, data)
    output1 = [] # to store classified labels
    output2 = [] # to store confidences
    posFrequency = getLabelFrequency(dataset)[1]
    negFrequency = getLabelFrequency(dataset)[0]
    p = 1 / binNum
    
    for i in range(0, len(dataset)): # go through all samples
        pTrue = 0.0
        pFalse = 0.0
        for j in range(1, dataset.shape[1]): # go through all attributes
            likelihood = getLikelihood(binNum, j, dataset)
            for k in range(1, binNum + 1): # go through all bins
                if mValue >= 0 and dataset[i, j] == k: # if m >= 0 , use m-estimate
                    mTrue = (likelihood[k - 1, 1] + mValue * p) / (posFrequency + mValue)
                    mFalse = (likelihood[k - 1, 0] + mValue * p) / (negFrequency + mValue)
                    if mTrue == 0 and mFalse == 0: # help to solve log(0) errors
                        break
                    elif mTrue == 0 and mFalse != 0:
                        pFalse += math.log(mFalse, 2)
                    elif mTrue != 0 and mFalse == 0:
                        pTrue += math.log(mTrue, 2)
                    else:
                        pTrue += math.log(mTrue, 2)
                        pFalse += math.log(mFalse, 2)
                elif mValue < 0 and dataset[i, j] == k: # if m < 0, use Laplace Smoothing
                    pTrue += math.log((likelihood[k - 1, 1] + 1) / (posFrequency + binNum), 10)
                    pFalse += math.log((likelihood[k - 1, 0] + 1) / (negFrequency + binNum), 10)
        if posFrequency == 0:
            pFalse += math.log(negFrequency / len(dataset), 2)
        elif negFrequency == 0:
            pTrue += math.log(posFrequency / len(dataset), 2)
        else:
            pTrue += math.log(posFrequency / len(dataset), 2)
            pFalse += math.log(negFrequency / len(dataset), 2)
             
        confidence = pFalse / (pTrue + pFalse)
        if confidence > 0.5:
            output1.append(1.0)
            output2.append(confidence)
        else:
            output1.append(0.0)
            output2.append(confidence)
    output1 = np.array(output1)
    output2 = np.array(output2)
    return output1, output2

def calcAccPreRec(preds, data): # This funtion should return the accuracy, precision and recall of the preds dataset.
    truePos = 0.0
    trueNeg = 0.0
    falsePos = 0.0
    falseNeg = 0.0
    for i in range(0, len(data)):
        if preds[i] == data[i,-1] == 1.0:
            truePos = truePos + 1
        elif preds[i] == data[i,-1] == 0.0:
            trueNeg = trueNeg + 1
        elif preds[i] == 1.0 and data[i,-1] == 0.0:
            falsePos = falsePos + 1
        elif preds[i] == 0.0 and data[i,-1] == 1.0:
            falseNeg = falseNeg + 1
            
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    return accuracy, precision, recall


# In[3]:


def calcSd(inputs):
    total = 0
    sd = 0.0
    for i in range(inputs.shape[0]):
        total = total + inputs[i]
    average = total/inputs.shape[0]
    for i in range(inputs.shape[0]):
        dif = average - inputs[i]
        sd = sd + math.pow(dif,2)
    sd = sd/inputs.shape[0]
    sd = math.pow(sd, 0.5)
    return sd
    


# In[4]:


def rocArea(confi, label):
    area = 0
    fpr = np.array([])
    tpr = np.array([])
    for i in range(10):
        con = np.array([])
        real = np.array([])
        for x in range(label.shape[0]):
            if confi[x] <= 0.1*(i+1):
                con = np.append(con, confi[x])
                real = np.append(real, label[x])
        fpr = np.append(fpr, calcFPR(con,real))
        tpr = np.append(tpr, calcTPR(con,real))
    sfpr = sort(fpr,tpr)[0]
    stpr = sort(fpr,tpr)[1]
    for i in range(sfpr.shape[0] - 1):
        area = area + (sfpr[i+1]-sfpr[i])*(stpr[i]+0.5*abs(stpr[i]-stpr[i+1]))
    area = area + (1-sfpr[-1])*(stpr[-1]+0.5*abs(stpr[i]-1))
    area = area + (sfpr[0])*(0.5*stpr[0])
    return area
        


# In[5]:


def calcTPR(pred,real):
    tp = 0
    fn = 0
    for i in range(real.shape[0]):
        if real[i] == 1:
            if pred[i] >= 0.5:
                tp = tp + 1
            else:
                fn = fn + 1
    if (tp + fn) == 0:
        tpr = 0
    else:
        tpr = tp/(tp + fn)
    return tpr


# In[6]:


def calcFPR(pred,real):
    fp = 0
    tn = 0
    for i in range(real.shape[0]):
        if real[i] == 0:
            if pred[i] >= 0.5:
                fp = fp + 1
            else:
                tn = tn + 1
    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp/(fp+tn)
    return fpr
    


# In[7]:


def sort(ar1, ar2):
    new1 = np.array([])
    new2 = np.array([])
    index = np.argsort(ar1)
    for i in range(ar1.shape[0]):
        new1 = np.append(new1, ar1[index[i]])
        new2 = np.append(new2, ar2[index[i]])
    return new1,new2


# In[8]:


def calcAve(ar):
    total = 0;
    for i in range(ar.shape[0]):
        total = total + ar[i]
    return total/ar.shape[0]


# In[ ]:


path = input('Enter the path to the data:')
cv = int(input('Cross Validation? 0 for cv, 1 for full sample'))
numbin = int(input('Enter the number of bins for any continuous feature:'))
mvalue = int(input('Enter the value of m for the m-estimate:'))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
data = np.array(parse_c45(path).to_float())
acc = np.array([])
prec = np.array([])
recall = np.array([])
roc = np.array([])
if cv == 1:
    Bayes = naiveBayes(mvalue,numbin,data)
    pred = Bayes[0]
    confi = Bayes[1]
    acc = np.append(acc, calcAccPreRec(pred,data)[0])
    prec = np.append(prec, calcAccPreRec(pred,data)[1])
    recall = np.append(recall, calcAccPreRec(pred,data)[2])
    roc = np.append(roc, rocArea(confi, data[:,-1]))
else:
    cvdata = stratCrossValid(data)
    for i in range(5):
        Bayes = naiveBayes(mvalue,numbin,cvdata[i])
        pred = Bayes[0]
        confi = Bayes[1]
        acc = np.append(acc, calcAccPreRec(pred,cvdata[i])[0])
        prec = np.append(prec, calcAccPreRec(pred,cvdata[i])[1])
        recall = np.append(recall, calcAccPreRec(pred,cvdata[i])[2])
        roc = np.append(roc, rocArea(confi, cvdata[i][:,-1])) 
print('Accuracy:' ,round(calcAve(acc),3), round(calcSd(acc),3))
print('Precision:', round(calcAve(prec),3), round(calcSd(prec),3))
print('Recall:', round(calcAve(recall),3), round(calcSd(recall),3))
print('Area under ROC:', round(calcAve(acc),3))


# In[ ]:





# In[ ]:




