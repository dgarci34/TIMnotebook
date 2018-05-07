# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

#define function for parsing csv
def loadCsv(filename):
    #grab first line
    lines = csv.reader(open(filename, "rb"))
    #have a list with each element being a line
    dataset = list(lines)
    #loop through each line
    for i in range(len(dataset)):
        #only save the decimal value
        dataset[i] = [float(x) for x in dataset[i]]
    #return the output
    return dataset

#function for spliting the csv into ratios
def splitDataset(dataset, splitRatio):
    #set training size
    trainSize = int(len(dataset) * splitRatio)
    #initialize empty list
    trainSet = []
    #save copy of original dataset
    copy = list(dataset)
    #loop while in range of training set
    while len(trainSet) < trainSize:
        #grab a random value
        index = random.randrange(len(copy))
        #add to training set
        trainSet.append(copy.pop(index))
    #return tuple of training set and the original copy
    return [trainSet, copy]

#used to break up dataset by class
def separateByClass(dataset):
    #empty init
    separated = {}
    #loop through all elements in dataset
    for i in range(len(dataset)):
        #make a list of the line passed in
        vector = dataset[i]
        #if not seperated, first part empty
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        #is seperated, break up
        separated[vector[-1]].append(vector)
    #return result after loop
    return separated

#define function for getting the mean
def mean(numbers):
    #return arithmatic
    return sum(numbers)/float(len(numbers))
#define function for standard deviation
def stdev(numbers):
    #get the average
    avg = mean(numbers)
    #calculate the difference between input and mean
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    #return result
    return math.sqrt(variance)

#define function for sumerizing
def summarize(dataset):
    #list the mean and stddev for each attibute from dataset
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    #delete any previous sumaries
    del summaries[-1]
    #return result
    return summaries

#function for sumerizing by class
def summarizeByClass(dataset):
    #first seperate dataset by classes into list
    separated = separateByClass(dataset)
    #have an empty summary set
    summaries = {}
    #loop through each class
    for classValue, instances in separated.iteritems():
        #add summary of class to output
        summaries[classValue] = summarize(instances)
    #return output
    return summaries

#define function for finding probability
def calculateProbability(x, mean, stdev):
    #get exponent
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    #return eponent over squared root of 2pi times standard deviation
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#define probability of belonging to certatin class
def calculateClassProbabilities(summaries, inputVector):
    #initialize empty probability
    probabilities = {}
    #loop through classes
    for classValue, classSummaries in summaries.iteritems():
        #if it exists assign 1
        probabilities[classValue] = 1
        #loop through class summaries
        for i in range(len(classSummaries)):
            #get the mean and standard dev
            mean, stdev = classSummaries[i]
            #prepare a list of x
            x = inputVector[i]
            #multiply the probabilities by a the new probability
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    #return output
    return probabilities

#new function for making predictions
def predict(summaries, inputVector):
    #obtain proability
    probabilities = calculateClassProbabilities(summaries, inputVector)
    #init assumptions
    bestLabel, bestProb = None, -1
    #loop through all probability items
    for classValue, probability in probabilities.iteritems():
        #if new probability is higher then previous, replace
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    #return output
    return bestLabel

#function for obtaining a prediction
def getPredictions(summaries, testSet):
    #init empty predictions
    predictions = []
    #loop through test set
    for i in range(len(testSet)):
        #make predition based on current test set
        result = predict(summaries, testSet[i])
        #add to prediction vector
        predictions.append(result)
    #return output
    return predictions

#function for obtaining accuracy
def getAccuracy(testSet, predictions):
    #init with no correct
    correct = 0
    #loop through test set
    for x in range(len(testSet)):
        #if a prediction matched the actual
        if testSet[x][-1] == predictions[x]:
            #increment correct count
            correct += 1
    #return the percentage of correct
    return (correct/float(len(testSet))) * 100.0

#for 67/33 split
def main():
    #load filename
    filename = 'pima-indians-diabetes.data.csv'
    #preditermined ratio
    splitRatio = 0.67
    #include dataset
    dataset = loadCsv(filename)
    #define training set and testing set
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    #show statistics on input
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    #obtain accuracy of predictions
    accuracy = getAccuracy(testSet, predictions)
    #output the accuracy found
    print('Accuracy: {0}%').format(accuracy)

main()
