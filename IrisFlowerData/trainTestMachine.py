from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import visualization

multiClassLogistic = LogisticRegression(multi_class='ovr')

def trainMachine():
    print("******************** T R A I N I N G    I N     P R O G R E S S ******************")
    data = visualization.my_iris_data
    print("=========================Training Data=========================")
    print(data)
    selectedFeatureSet = data[:,2:4]
    selectedLabel = data[:,4]
    print("=========================Feature Set=========================")
    print(selectedFeatureSet)
    print("=======Labels (Setosa:0, Versicolor:1, Virginica: 2)=========")
    print("Label: ", selectedLabel)
    #Fitting the data -> finding required parameters/coefficients
    multiClassLogistic.fit(selectedFeatureSet, selectedLabel)
    # The coefficients/parameters
    print("Coefficients (b0 and b1) of the three logistic regression models used in One vs Rest (ovr) scheme:\n "+ str(multiClassLogistic.coef_))


def testMachine():
    print("\n\n******************** T E S T I N G    I N     P R O G R E S S ******************")
    testData= np.genfromtxt('IrisDataSetTest.csv', delimiter=',')
    print("=========================Test Data=========================")
    print(testData)
    testSet = testData[:,2:4]
    print("=========================Test Features=========================")
    print(testSet)
    trueCategory = testData[:,4]
    print("=========================True Class=========================")
    print(trueCategory)
    print("=========================Predicted Class=========================")
    predictedCategory = multiClassLogistic.predict(testSet)
    print(predictedCategory)
    print("Accuracy Score: ", accuracy_score(trueCategory, predictedCategory))

trainMachine()
testMachine()