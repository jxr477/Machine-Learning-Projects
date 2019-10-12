#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import math
import operator
import numpy as np
import random
from mldata import parse_c45

# In[3]:


data = np.array(parse_c45('voting').to_float())
data2 = np.array(parse_c45('spam').to_float())
data3 = np.array(parse_c45('volcanoes').to_float())


# In[152]:



def calcEntropy(data): # This funtion will return the entropy of a given dataset (either discrete or continuous).
    
    posCounts = 0.0
    negCounts = 0.0
    if len(data) == 0:
        return 0
    else:
        for i in range(0, len(data)-1):
            if  1.0 == data[i,-1]:
                posCounts = posCounts + 1 # number of positive labels
            else:
                negCounts = negCounts + 1 # number of negative labels
        probPos = posCounts / len(data)
        probNeg = negCounts / len(data)
        entropy = 0.0 # initialize entropy 
        if (probPos != 0 and probNeg != 0) :
            pos = - probPos * math.log(probPos, 2)
            neg = - probNeg * math.log(probNeg, 2)
            entropy = pos + neg
        return entropy
    
def calcAttr(attribute, data): # help function of calcInfoGain
    label = []

    for i in range(0, len(data)):
        if data[i, attribute] == 2.0:
            label.append(1.0)
        elif data[i, attribute] == 0.0:
            label.append(0.0)
        else:
            label.append(-1.0)
    return label

def calcInfoGain(attribute, data): # This funtion should return the information gain of a specified 
	# attribute of a discrete dataset(voting.data).
    parentEnt = calcEntropy(data)
    child = calcAttr(attribute, data)
    rightEnt = 0.0
    leftEnt = 0.0
    correctPos = 0.0
    correctNeg = 0.0
    wrongPos = 0.0
    wrongNeg = 0.0
    for i in range(0, len(data)):
        if child[i] == data[i,-1] == 1.0:
            correctPos = correctPos + 1
        elif child[i] == data[i,-1] == 0.0:
            correctNeg = correctNeg + 1
        elif child[i] == 1.0 and data[i,-1] == 0.0:
            wrongPos = wrongPos + 1
        elif child[i] == 0.0 and data[i,-1] == 1.0:
            wrongNeg = wrongNeg + 1
            
    if correctPos == 0 or wrongPos == 0: # help solving dividing(0) or log(0) errors
        rightEnt = 0
    else:
        prob1 = correctPos / (correctPos + wrongPos)
        prob2 = wrongPos / (correctPos + wrongPos)
        rightEnt = - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2)
    
    if correctNeg == 0 or wrongNeg == 0:
        leftEnt = 0
        
    else:
        prob3 = correctNeg / (correctNeg + wrongNeg)
        prob4 = wrongNeg / (correctNeg + wrongNeg)
        leftEnt = - prob3 * math.log(prob3, 2) - prob4 * math.log(prob4, 2)  
    
    prob5 = (correctPos + wrongPos) / len(data)
    prob6 = (correctNeg + wrongNeg) / len(data)
    infoGain = parentEnt - (prob5 * rightEnt + prob6 * leftEnt)
    return infoGain

def bestAttr(data): # This funtion should return the index of the attribute with the highest information gain
	# of voting.
    maxInfoGain = 0.0
    attrChoice = 0
    
    i = 1
    for i in range(1, data.shape[1] - 1):
        temp = calcInfoGain(i, data)
        if temp > maxInfoGain:
            maxInfoGain = temp # find the highest information gain
            attrChoice = i # find the index of the attribute with the highest information gain
    return attrChoice

def bestAttr2(data): # This funtion should return the index of the attribute with the highest gain ratio
	# of voting.
    maxGainRatio = 0.0
    attrChoice = 0
    
    i = 1
    for i in range(1, data.shape[1] - 1):
        temp = calcGainRatio(i, data)
        if temp > maxGainRatio:
            maxGainRatio = temp # find the highest information gain
            attrChoice = i # find the index of the attribute with the highest information gain
    return attrChoice

def calcAccuracy(preds, data): #
    correctCounts = 0.0
    for i in range(1, len(preds) - 1):
        if preds[i] == data[i,-1]:
            correctCounts = correctCounts + 1
    accuracy = correctCounts / len(data)
    
    return accuracy

def partition(data, attr) : # This function should divide voting.data into three datasets, each corresponding to label "+", "-" and "0"
    attrNumb = attr
    output1 = [] # for "+" case
    output2 = [] # for "-" case
    output3 = [] # for "0" case
    x = calcAttr(attrNumb,data)
    for i in range(0, len(data)): # add data to each datasets
        if x[i] == 1.0:
            output1.append(data[i])
        elif x[i] == 0.0:
            output2.append(data[i])
        else:
            output3.append(data[i])
    output1 = np.array(output1) # convert to np array
    output2 = np.array(output2)
    output3 = np.array(output3)
    return output1,output2,output3

def calcGainRatio(attribute, data):
    Entropy = calcEntropy(data)
    InfoGain = calcInfoGain(attribute, data)
    
    return InfoGain / Entropy

def crossValid(data): # randomize the input data and divide it into five folds （normal 5-fold-validation）  
    np.random.seed(12345)
    np.random.shuffle(data)
    line = int(len(data) / 5)
    fold1 = data[0:line]
    fold2 = data[line:line * 2]
    fold3 = data[line * 2:line * 3]
    fold4 = data[line * 3:line * 4]
    fold5 = data[line * 4:len(data)]
    return fold1,fold2,fold3,fold4,fold5  

def stratCrossValid(data): # stratified 5-fold-validation for both discrete and continuous case
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
    
    np.random.seed(12345)
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



def findThreshold(attribute, data): # This fundtion should return an array of the thresholds.
    dataset = sortByAttr(attribute, data)
    threshold = []
    switch = 0 # variable used to find the attribute value changed index
    end_thresh = 0 # Threshold ON/OFF for the special case: same attribute with different class label
    for i in range(1,len(data)):
        if dataset[i, attribute] != dataset[i - 1, attribute]:   # update the attribute value changed index
            if(end_thresh == 1) :
                valueToAdd = (dataset[i-1,attribute] + dataset[i,attribute])/ 2
                if (threshold != [] and threshold[-1] != valueToAdd) :    # Make sure no depulicate
                    threshold.append(valueToAdd)
                elif (threshold == []) :
                    threshold.append(valueToAdd)
                end_thresh = 0
            
            switch = i
            
        if dataset[i,-1] != dataset[i - 1, -1] and dataset[i, attribute] != dataset[i - 1, attribute]: # If the label of one row is not the same as the previous row, put a threashold there.
            valueToAdd = (dataset[i-1,attribute] + dataset[i,attribute])/ 2
            if (threshold != [] and threshold[-1] != valueToAdd) :    # Make sure no depulicate
                threshold.append(valueToAdd)
            elif (threshold == []) :
                threshold.append(valueToAdd)
        elif dataset[i,-1] != dataset[i - 1, -1] :  # same attribute value with different class label
            end_thresh = 1 
            if switch != 0 :
                valueToAdd = (dataset[switch,attribute] + dataset[switch-1,attribute]) / 2
                if (threshold != [] and threshold[-1] != valueToAdd) :    # Make sure no depulicate
                    threshold.append(valueToAdd)
                elif (threshold == []) :
                    threshold.append(valueToAdd)
             
    threshold = np.array(threshold)   
    
    if(len(threshold) > 100) :
        newThreshold = []
        index = math.floor(len(threshold) / 100)
        for x in range(0,100):
            newThreshold.append(threshold[x*index])
        return newThreshold
    return threshold
	
def sortByAttr(attribute, data) :  # This function returns a new array sorted by the required attribute. If the attribute has the same value, then sorted by the class label.
    result = sorted(data, key = lambda x: (x[attribute], x[-1]))
    result = np.array(result)
    return result

def splitAttr(thresholdNum, attribute, data): 
    
    correctPos = 0.0
    correctNeg = 0.0
    wrongPos = 0.0
    wrongNeg = 0.0
    
    for i in range(0, len(data) - 1):
        if data[i, attribute] <= thresholdNum:
            if data[i,-1] == 0.0 :
                correctNeg += 1
            else :
                wrongNeg += 1
        else:
            if data[i,-1] == 0.0 :
                wrongPos += 1
            else:
                correctPos += 1

    return correctPos,correctNeg,wrongPos,wrongNeg


def calcInfoGainC(thresholdNum, attribute, data, parentEnt): # This funtion should return the information gain of a specified 
	# threshold of an attribute of a continuous dataset.
    
    child = splitAttr(thresholdNum, attribute, data)
    rightEnt = 0.0
    leftEnt = 0.0
    correctPos = child[0]
    correctNeg = child[1]
    wrongPos = child[2]
    wrongNeg = child[3]

    if correctPos == 0 or wrongPos == 0: # help solving dividing(0) or log(0) errors
        rightEnt = 0
    else:
        prob1 = correctPos / (correctPos + wrongPos)
        prob2 = wrongPos / (correctPos + wrongPos)
        rightEnt = - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2)

    if correctNeg == 0 or wrongNeg == 0:
        leftEnt = 0

    else:
        prob3 = correctNeg / (correctNeg + wrongNeg)
        prob4 = wrongNeg / (correctNeg + wrongNeg)
        leftEnt = - prob3 * math.log(prob3, 2) - prob4 * math.log(prob4, 2)  

    prob5 = (correctPos + wrongPos) / len(data)
    prob6 = (correctNeg + wrongNeg) / len(data)
    infoGain = parentEnt - (prob5 * rightEnt + prob6 * leftEnt)
    return infoGain

def calcGainRatioC(thresholdNum, attribute, data):
    entropy = calcEntropy(data)
    infoGain = calcInfoGainC(thresholdNum, attribute, data, entropy)
    gainRatio = infoGain / entropy
    return gainRatio

def findMaxInfoGainC(attribute, data): # TThis funtion should return a matrix of max information gain of a specified attribute 
                                       # among all thresholds of a continuous dataset, and the selected thresholds.
    maxInfoGain = 0
    threshold = findThreshold(attribute,data)
    if len(threshold) == 0:
        maxInfGain = 0 
        thresholdChoice = 0
    else:
        thresholdChoice = threshold[0]
        parentEnt = calcEntropy(data)
        for i in range(0, len(threshold)):
            temp = calcInfoGainC(threshold[i], attribute, data, parentEnt)
            if temp > maxInfoGain:
                maxInfoGain = temp
                thresholdChoice = threshold[i]
    return maxInfoGain,thresholdChoice

def findMaxGainRatioC(attribute, data): # TThis funtion should return a matrix of max information gain of a specified attribute 
                                       # among all thresholds of a continuous dataset, and the selected thresholds.
    maxGainRatio = 0
    threshold = findThreshold(attribute,data)
    if len(threshold) == 0:
        return 0,0
    else:
        thresholdChoice = threshold[0]
    for i in range(0, len(threshold)):
        temp = calcGainRatioC(threshold[i], attribute, data)
        if temp > maxGainRatio:
            maxGainRatio = temp
            thresholdChoice = threshold[i]
    return maxGainRatio,thresholdChoice


def bestThreAttr(data):# This funtion should return the threshold index and the attribute 
# with the largest information gain of a continuous dataset.
    maxInfoGain = 0
    bestThre = 0
    bestAttr = 0
    
    for i in range(1, data.shape[1] - 1): # find maxInfoGain among all thresholds of all attributes
        temp = findMaxInfoGainC(i, data)
        if temp[0] > maxInfoGain:
            maxInfoGain = temp[0]
            bestAttr = i
            bestThre = temp[1]
    return bestThre, bestAttr

def bestThreAttr2(data):# This funtion should return the threshold index and the attribute 
# with the largest gain ratio of a continuous dataset.
    maxGainRatio = 0
    bestThre = 0
    bestAttr = 0
    
    for i in range(1, data.shape[1] - 1): # find maxInfoGain among all thresholds of all attributes
        temp = findMaxGainRatioC(i, data)
        if temp[0] > maxGainRatio:
            maxGainRatio = temp[0]
            bestAttr = i
            bestThre = temp[1]
    return bestThre, bestAttr

def partitionC(data, attribute, thresholdNum): # This funtion should divide a continuous dataset into
    # two datasets, according to a specified threshold of an attribute.
    output1 = []
    output2 = []
    for i in range(data.shape[0]):
        if data[i, attribute] <= thresholdNum:
            output1.append(data[i])
        else:
            output2.append(data[i])
    output1 = np.array(output1)
    output2 = np.array(output2)
    return output1, output2


# In[153]:


class TreeNode:
    leaf = False
    leafValue = None
    ifVisited = False
    haveSon = False
    son = np.array([])
    def __init__(self, attribute, value, parentNode, depth, dataset, index, opThre):
        self.attribute = attribute
        self.value = value
        self.parentNode = parentNode
        self.dataset = dataset
        self.index = index
        self.depth = depth
        self.opThre = opThre


# In[3]:


def countAtt(attribute):
    count = 0
    for x in range(attribute.shape[0]):
        if attribute[x]:
            count = count + 1
    return count


# In[4]:


def ifPrun(nodeList):
    prun = True
    for x in range(nodeList.shape[0]):
        if nodeList[x].ifVisited == False:
            prun = False
    return prun



def SplitF(option,attribute, data):
    if option == 0:
        calcInfoGain(attribute, data)
    if option == 1:
        calcGainRatio(attribute, data)


def attTransfer(att):
    attT = np.array([])
    for x in range(att.shape[0]):
        if att[x]:
            ind = int(x)
            attT = np.append(attT,ind)
    attT = attT.astype('int32')
    return attT


# In[154]:


def IfLeaf(treenode):
    if calcEntropy(treenode.dataset) == 0:
        return True
    else:
        return False


# In[155]:


def nextNode(Node):
    maxDepth = 0
    nodeIndex = -1
    for x in range(Node.shape[0]):
        if Node[x].depth > maxDepth and Node[x].ifVisited == False:
            maxDepth = Node[x].depth
            nodeIndex = x
    return nodeIndex


# In[156]:


def decideValue(TreeNode):
    posCount = 0
    negCount = 0
    for x in range(TreeNode.dataset.shape[0]):
        if TreeNode.dataset[x,-1] == 1:
            posCount = posCount + 1
        elif TreeNode.dataset[x,-1] == 0:
            negCount = negCount + 1
    if posCount > negCount:
        return 1
    elif posCount < negCount:
        return 0
    else:
        return random.randint(0,1)


# In[157]:


def createDTreeD(dataset,maxDepth):
    root = TreeNode(None,None, None,0, dataset,0, None)
    Node = np.array([root])
    att = np.arange(np.size(dataset,1))
    att[:] = True
    currentNodeIndex = 0;
    while countAtt(att) > 2 and ifPrun(Node) == False:
        if maxDepth < 0 or Node[currentNodeIndex].depth < maxDepth:
            if IfLeaf(Node[currentNodeIndex]) == False:
                Node[currentNodeIndex].haveSon = True
                currentData = Node[currentNodeIndex].dataset[:,attTransfer(att)]
                currentAtt = attTransfer(att)[bestAttr(currentData)]
                att[currentAtt] = False
                Node = np.append(Node, TreeNode(currentAtt, 1, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[0], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 0, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[1], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 2, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[2], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node[currentNodeIndex].son = Node[currentNodeIndex].son.astype(int)
        Node[currentNodeIndex].ifVisited = True
        currentNodeIndex = nextNode(Node)
    for x in range(Node.shape[0]):
        if Node[x].haveSon == False:
            Node[x].leafValue = decideValue(Node[x])
    return Node


# In[158]:


def testD(trainingD ,testD, maxDepth):
    correctCount = 0
    Tree = createDTreeD(trainingD, maxDepth)
    for x in range(testD.shape[0]):
        predV = None
        getValue = False
        currentNode = Tree[0]
        while getValue == False:
            if currentNode.haveSon == False:
                getValue = True
                predV = currentNode.leafValue
            else:
                nextIndex = 0
                attr = Tree[currentNode.son[0]].attribute
                for y in range(currentNode.son.shape[0]):
                    if testD[x,attr] == Tree[currentNode.son[y]].value:
                        nextIndex = currentNode.son[y]
                currentNode = Tree[nextIndex]
        if predV == testD[x,-1]:
            correctCount = correctCount + 1
    accuracy = correctCount / (testD.shape[0])
    accuracy = correctCount / (testD.shape[0])
    maxD = getMaxDepth(Tree)
    size = Tree.shape[0] - 1
    fF = Tree[1].attribute
    return accuracy, maxD, size, fF


# In[159]:


def createDTreeC(dataset,maxDepth):
    root = TreeNode(None,None, None,0, dataset, 0, None)
    Node = np.array([root])
    att = np.arange(np.size(dataset,1))
    att[:] = True
    currentNodeIndex = 0;
    while countAtt(att) > 2 and ifPrun(Node) == False:
        if maxDepth < 0 or Node[currentNodeIndex].depth < maxDepth:
            if IfLeaf(Node[currentNodeIndex]) == False:
                Node[currentNodeIndex].haveSon = True
                currentData = Node[currentNodeIndex].dataset[:,attTransfer(att)]
                currentAtt = attTransfer(att)[bestThreAttr(currentData)[1]]
                att[currentAtt] = False
                Node = np.append(Node, TreeNode(currentAtt, 1, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partitionC(Node[currentNodeIndex].dataset, currentAtt, bestThreAttr(currentData)[0])[0], np.size(Node), 0 ))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 0, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partitionC(Node[currentNodeIndex].dataset, currentAtt, bestThreAttr(currentData)[0])[1], np.size(Node), 1 ))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node[currentNodeIndex].son = Node[currentNodeIndex].son.astype(int)
        Node[currentNodeIndex].ifVisited = True
        currentNodeIndex = nextNode(Node)
    for x in range(Node.shape[0]):
        if Node[x].haveSon == False:
            Node[x].leafValue = decideValue(Node[x])
    return Node


# In[160]:


def testC(trainingD ,testD, maxDepth):
    correctCount = 0
    Tree = createDTreeC(trainingD, maxDepth)
    for x in range(testD.shape[0]):
        predV = None
        getValue = False
        currentNode = Tree[0]
        while getValue == False:
            if currentNode.haveSon == False:
                getValue = True
                predV = currentNode.leafValue
            else:
                nextIndex = 0
                attr = Tree[currentNode.son[0]].attribute
                for y in range(currentNode.son.shape[0]):
                    if testD[x,attr] == Tree[currentNode.son[y]].value:
                        nextIndex = currentNode.son[y]
            currentNode = Tree[nextIndex]
        if predV == testD[x,-1]:
            correctCount = correctCount + 1
    accuracy = correctCount / (testD.shape[0])
    maxD = getMaxDepth(Tree)
    size = Tree.shape[0] - 1
    fF = Tree[1].attribute
    return accuracy, maxD, size, fF
            
    


# In[161]:


def getMaxDepth(Tree):
    maxDepth = 0
    for x in range(Tree.shape[0]):
        if Tree[x].depth > maxDepth:
            maxDepth = Tree[x].depth
    return maxDepth


# In[ ]:


path = input("Please enter the name of the data:")
CV = input("0 for CV, 1 for full")
maxDepth = input("Please enter the max depth of the tree, -1 will grow the full tree")
Op = input("0 for information gain, 1  for gain ratio to split the tree")
accuracy = 0
size = 0
maxD = 0
firstFeature = 0
if path == "voting":
    if Op == 0:
        if CV == 0:
            folds = stratCrossValid(data)
            totalA = 0
            for x in range(4):
                totalA = totalA + testD(folds[x],folds[4],maxDepth)[0]
                size = testD(folds[x],folds[4],maxDepth)[2]
                firstFeature = testD(folds[x],folds[4],maxDepth)[3]
                if testD(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testD(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testD(data,data,maxDepth)[0]
            size = testD(data,data,maxDepth)[2]
            maxD = testD(data,data,maxDepth)[1]
            firstFeature = testD(data,data,maxDepth)[3]
    else:
        if CV == 0:
            folds = stratCrossValid(data)
            totalA = 0
            for x in range(4):
                totalA = totalA + testD2(folds[x],folds[4],maxDepth)[0]
                size = testD2(folds[x],folds[4],maxDepth)[2]
                firstFeature = testD2(folds[x],folds[4],maxDepth)[3]
                if testD2(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testD2(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testD2(data,data,maxDepth)[0]
            size = testD2(data,data,maxDepth)[2]
            maxD = testD2(data,data,maxDepth)[1]
            firstFeature = testD2(data,data,maxDepth)[3]
if path == "spam":
    if Op == 0:
        if CV == 0:
            folds = stratCrossValid(data2)
            totalA = 0
            for x in range(4):
                totalA = totalA + testC(folds[x],folds[4],maxDepth)[0]
                size = testC(folds[x],folds[4],maxDepth)[2]
                firstFeature = testC(folds[x],folds[4],maxDepth)[3]
                if testC(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testC(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testC(data2,data2,maxDepth)[0]
            size = testC(data2,data2,maxDepth)[2]
            maxD = testC(data2,data2,maxDepth)[1]
            firstFeature = testC(data2,data2,maxDepth)[3]
    else:
        if CV == 0:
            folds = stratCrossValid(data2)
            totalA = 0
            for x in range(4):
                totalA = totalA + testC2(folds[x],folds[4],maxDepth)[0]
                size = testC2(folds[x],folds[4],maxDepth)[2]
                firstFeature = testC2(folds[x],folds[4],maxDepth)[3]
                if testC2(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testC2(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testC2(data2,data2,maxDepth)[0]
            size = testC2(data2,data2,maxDepth)[2]
            maxD = testC2(data2,data2,maxDepth)[1]
            firstFeature = testC2(data2,data2,maxDepth)[3]
if path == "volcanoes":
    if Op == 0:
        if CV == 0:
            folds = stratCrossValid(data3)
            totalA = 0
            for x in range(4):
                totalA = totalA + testC(folds[x],folds[4],maxDepth)[0]
                size = testC(folds[x],folds[4],maxDepth)[2]
                firstFeature = testC(folds[x],folds[4],maxDepth)[3]
                if testC(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testC(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testC(data3,data3,maxDepth)[0]
            size = testC(data3,data3,maxDepth)[2]
            maxD = testC(data3,data3,maxDepth)[1]
            firstFeature = testC(data3,data3,maxDepth)[3]
    else:
        if CV == 0:
            folds = stratCrossValid(data3)
            totalA = 0
            for x in range(4):
                totalA = totalA + testC2(folds[x],folds[4],maxDepth)[0]
                size = testC2(folds[x],folds[4],maxDepth)[2]
                firstFeature = testC2(folds[x],folds[4],maxDepth)[3]
                if testC2(folds[x],folds[4],maxDepth)[1] > maxD:
                    maxD = testC2(folds[x],folds[4],maxDepth)[1]
        else:
            accuracy = testC2(data3,data3,maxDepth)[0]
            size = testC2(data3,data3,maxDepth)[2]
            maxD = testC2(data3,data3,maxDepth)[1]
            firstFeature = testC2(data3,data3,maxDepth)[3]
            
print("Accuracy:", accuracy,"\n")
print("Size:", size, "\n")
print("Maximum Depth:", maxD, "\n")
print("First Feature:", firstFeature, "\n")


# In[183]:


def createDTreeD2(dataset,maxDepth):
    root = TreeNode(None,None, None,0, dataset,0, None)
    Node = np.array([root])
    att = np.arange(np.size(dataset,1))
    att[:] = True
    currentNodeIndex = 0;
    while countAtt(att) > 2 and ifPrun(Node) == False:
        if maxDepth < 0 or Node[currentNodeIndex].depth < maxDepth:
            if IfLeaf(Node[currentNodeIndex]) == False:
                Node[currentNodeIndex].haveSon = True
                currentData = Node[currentNodeIndex].dataset[:,attTransfer(att)]
                currentAtt = attTransfer(att)[bestAttr2(currentData)]
                att[currentAtt] = False
                Node = np.append(Node, TreeNode(currentAtt, 1, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[0], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 0, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[1], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 2, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partition(Node[currentNodeIndex].dataset, currentAtt)[2], np.size(Node), None))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node[currentNodeIndex].son = Node[currentNodeIndex].son.astype(int)
        Node[currentNodeIndex].ifVisited = True
        currentNodeIndex = nextNode(Node)
    for x in range(Node.shape[0]):
        if Node[x].haveSon == False:
            Node[x].leafValue = decideValue(Node[x])
    return Node


# In[1]:


def testD2(trainingD ,testD, maxDepth):
    correctCount = 0
    Tree = createDTreeD2(trainingD, maxDepth)
    for x in range(testD.shape[0]):
        predV = None
        getValue = False
        currentNode = Tree[0]
        while getValue == False:
            if currentNode.haveSon == False:
                getValue = True
                predV = currentNode.leafValue
            else:
                nextIndex = 0
                attr = Tree[currentNode.son[0]].attribute
                for y in range(currentNode.son.shape[0]):
                    if testD[x,attr] == Tree[currentNode.son[y]].value:
                        nextIndex = currentNode.son[y]
                currentNode = Tree[nextIndex]
        if predV == testD[x,-1]:
            correctCount = correctCount + 1
    accuracy = correctCount / (testD.shape[0])
    accuracy = correctCount / (testD.shape[0])
    maxD = getMaxDepth(Tree)
    size = Tree.shape[0] - 1
    return accuracy, maxD, size


# In[2]:


def createDTreeC2(dataset,maxDepth):
    root = TreeNode(None,None, None,0, dataset, 0, None)
    Node = np.array([root])
    att = np.arange(np.size(dataset,1))
    att[:] = True
    currentNodeIndex = 0;
    while countAtt(att) > 2 and ifPrun(Node) == False:
        if maxDepth < 0 or Node[currentNodeIndex].depth < maxDepth:
            if IfLeaf(Node[currentNodeIndex]) == False:
                Node[currentNodeIndex].haveSon = True
                currentData = Node[currentNodeIndex].dataset[:,attTransfer(att)]
                currentAtt = attTransfer(att)[bestThreAttr2(currentData)[1]]
                att[currentAtt] = False
                Node = np.append(Node, TreeNode(currentAtt, 1, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partitionC(Node[currentNodeIndex].dataset, currentAtt, bestThreAttr2(currentData)[0])[0], np.size(Node), 0 ))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node = np.append(Node, TreeNode(currentAtt, 0, Node[currentNodeIndex], Node[currentNodeIndex].depth + 1, partitionC(Node[currentNodeIndex].dataset, currentAtt, bestThreAttr2(currentData)[0])[1], np.size(Node), 1 ))
                Node[currentNodeIndex].son = np.append(Node[currentNodeIndex].son, np.size(Node)-1)
                Node[currentNodeIndex].son = Node[currentNodeIndex].son.astype(int)
        Node[currentNodeIndex].ifVisited = True
        currentNodeIndex = nextNode(Node)
    for x in range(Node.shape[0]):
        if Node[x].haveSon == False:
            Node[x].leafValue = decideValue(Node[x])
    return Node


# In[3]:


def testC2(trainingD ,testD, maxDepth):
    correctCount = 0
    Tree = createDTreeC2(trainingD, maxDepth)
    for x in range(testD.shape[0]):
        predV = None
        getValue = False
        currentNode = Tree[0]
        while getValue == False:
            if currentNode.haveSon == False:
                getValue = True
                predV = currentNode.leafValue
            else:
                nextIndex = 0
                attr = Tree[currentNode.son[0]].attribute
                for y in range(currentNode.son.shape[0]):
                    if testD[x,attr] == Tree[currentNode.son[y]].value:
                        nextIndex = currentNode.son[y]
            currentNode = Tree[nextIndex]
        if predV == testD[x,-1]:
            correctCount = correctCount + 1
    accuracy = correctCount / (testD.shape[0])
    maxD = getMaxDepth(Tree)
    size = Tree.shape[0] - 1
    fF = Tree[1].attribute
    return accuracy, maxD, size, fF


# In[ ]:




