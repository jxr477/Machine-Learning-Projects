#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from mldata import parse_c45


# In[ ]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

# for test purpose
#def likelihood(features, target, weights):
#    scores = np.dot(features,weights)
#    prediction = sigmoid(scores)
#    for i in prediction:
#        if i == 0:
#            i = 0.001
#    ll = -np.dot(target.T, np.log(prediction)) - np.dot((1-target).T, np.log(1-prediction))
#    ll = np.sum(ll)/len(target)
#    return ll

# for test purpose
#def norms(weight):
#    norm = 0.0
#    for i in weight:
#        norm += np.square(i)
#    return np.sqrt(norm)

def calcAccPreRec(data, weights): # This funtion should return the accuracy, precision and recall of the preds dataset.
    truePos = 0.0
    trueNeg = 0.0
    falsePos = 0.0
    falseNeg = 0.0
    
    features = np.append(data[:,1:-1],np.ones((data.shape[0],1)),1)
    score = np.dot(features, weights)
    predictions = sigmoid(score)
    
    for i in range(len(predictions)):
        if predictions[i] > 0.5:     # predict 1
            if (data[i,-1] == 1):
                truePos = truePos + 1
            else:
                falsePos = falsePos + 1
        else:                       # predict 0
            if(data[i,-1] == 0):
                trueNeg = trueNeg + 1
            else:
                falseNeg = falseNeg + 1
            
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    
    if(truePos+falsePos == 0):
        precision = 0
    else :
        precision = truePos / (truePos + falsePos)
    if(truePos+falseNeg == 0): 
        recall = 0
    else:
        recall = truePos / (truePos + falseNeg)
    return accuracy, precision, recall

# This function will return the finally predictions instead of weights for the ROC purpose
def logReg(data, lamda):
    
    weights = np.zeros(data.shape[1] - 1)   # initialized all the weights including b at last index
    steps = 2000                             # training steps. can be modified
    learningRate = 0.001        
    
    # add a column of ones at the end for better b calculation
    features = np.append(data[:,1:-1],np.ones((data.shape[0],1)),1)
                    
    for step in range(steps):
        score = np.dot(features, weights)     # w * x + b
        predictions = sigmoid(score)
        
        error =  predictions - data[:,-1]
        gradient = (np.dot(features.T,error) / len(error)) + ((weights) * lamda)
        weights -= learningRate * gradient
        #print(calcAccPreRec(data,weights))
        #print(likelihood(features,data[:,-1],weights))
        #ll.append(likelihood(features,data[:,-1],weights))

    return weights

def logReg_Cross(data,lamda,CV):
    if(CV == 1):
        weight = logReg(data,lamda)
        
        features = np.append(data[:,1:-1],np.ones((data.shape[0],1)),1)
        predictions = sigmoid(np.dot(features, weight))
        
        APR = calcAccPreRec(data,weight)
        roc = rocArea(predictions, data[:,-1])
        
        print("Accuracy:% .3f 0.000" %(APR[0]))
        print("Precision:% .3f 0.000" %(APR[1]))
        print("Recall:% .3f 0.000" %(APR[2]))
        print("Area under ROC:% .3f" %(roc))
        
    else:              # Do cross validation
        dataset = stratCrossValid(data)
        Accuracy = 0.0
        Precision = 0.0
        Recall = 0.0
        acc = np.array([])
        prec = np.array([])
        recall = np.array([])
        roc = np.array([])
        
        for i in dataset:
            weight = logReg(i,lamda)         
            
            features = np.append(i[:,1:-1],np.ones((i.shape[0],1)),1)
            predictions = sigmoid(np.dot(features, weight))
            
            APR = calcAccPreRec(i,weight)
            
            Accuracy += APR[0]
            Precision += APR[1]
            Recall += APR[2]
            acc = np.append(acc, APR[0])
            prec = np.append(prec, APR[1])
            recall = np.append(recall, APR[2])
            roc = np.append(roc, rocArea(predictions, i[:,-1]))
        
        Accuracy /= 5
        Precision /= 5
        Recall /= 5
        ROC = calcAve(roc)
        
        
        print("Accuracy:% .3f" %(Accuracy),round(calcSd(acc),3))
        print("Precision:% .3f" %(Precision),round(calcSd(prec),3))
        print("Recall:% .3f" %(Recall),round(calcSd(recall),3))
        print("Area under ROC:% .3f" %(ROC))

# for test purpose
#def plot(loss):
#    xs = []
#    ys = []
#    for i in range(len(loss)):
#        if(math.isnan(loss[i])):
#            continue 
#       else:
#            xs.append(i)
#            ys.append(loss[i])
#    plt.scatter(xs,ys)
#    plt.show()


# In[ ]:


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
    


# In[ ]:


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
        


# In[ ]:


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


# In[ ]:


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
    


# In[ ]:


def sort(ar1, ar2):
    new1 = np.array([])
    new2 = np.array([])
    index = np.argsort(ar1)
    for i in range(ar1.shape[0]):
        new1 = np.append(new1, ar1[index[i]])
        new2 = np.append(new2, ar2[index[i]])
    return new1,new2


# In[ ]:


def calcAve(ar):
    total = 0;
    for i in range(ar.shape[0]):
        total = total + ar[i]
    return total/ar.shape[0]


# In[ ]:


path = input('Enter the path to the data:')
cv = int(input('Cross Validation? 0 for cv, 1 for full sample'))
lamda = int(input('Enter the value of lamda:'))

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
dataType = []
data = parse_c45(path)
for i in data.schema:
    dataType.append(i.type)
data = np.array(data.to_float())

for k in range(len(dataType)):
    if(dataType[k] == 'NOMINAL'):
        for j in range(data.shape[1]):
            data[j,k] += 1.0 

logReg_Cross(data,lamda,cv)

