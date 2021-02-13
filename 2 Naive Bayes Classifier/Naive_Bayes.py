# Implementing Gaussian Naïve Bayes Classifier
# Author: Manas Trivedi, 181CO231

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Reading the dataset, setting class label
dataFrame = pd.read_csv("Iris.csv")
inputs = dataFrame.drop(columns = ["Id", "Species"])
outputs = dataFrame["Species"]

# Splitting dataset into training and testing data
trainingInputs, testingInputs, trainingOutputs, testingOutputs = train_test_split(inputs, outputs, test_size = 0.3)

# Creating the model
naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(trainingInputs, trainingOutputs)

# Applying the model on the testing data and finding performance measures
predictedOutputs = naiveBayesClassifier.predict(testingInputs)
confusionMatrix = confusion_matrix(testingOutputs, predictedOutputs)
accuracy = accuracy_score(testingOutputs, predictedOutputs) * 100
precision = precision_score(testingOutputs, predictedOutputs, average = 'micro') * 100
recall = recall_score(testingOutputs, predictedOutputs, average = 'micro') * 100
f1Score = f1_score(testingOutputs, predictedOutputs, average = 'micro') * 100
print('\nConfusion Matrix:\n', confusionMatrix, '\n')
print('Accuracy:\t', accuracy, '%\n')
print('Precision:\t', precision, '%\n')
print('Recall:\t\t', recall, '%\n')
print('F1-Score:\t', f1Score, '%\n')