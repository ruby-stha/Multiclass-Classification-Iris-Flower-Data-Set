from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import visualization

multiClassLogistic = LogisticRegression(multi_class='ovr')
plt = visualization.pyplot


"""Train the machine with training data to create trained model"""
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
    print("Coefficients (b1 and b2) of the three logistic regression models used in One vs Rest (ovr) scheme:\n "+ str(multiClassLogistic.coef_))
    print("Intercept (b0) of the three logistic regression models used in One vs Rest (ovr) scheme:\n "+ str(multiClassLogistic.intercept_))


"""Test the trained model with test data"""
def testMachine():
    print("\n\n******************** T E S T I N G    I N     P R O G R E S S ******************")
    testData= np.genfromtxt('IrisDataSetTest.csv', delimiter=',')
    print("=========================Test Data=========================")
    print(testData)
    plotTestDataAndDeciBoundary(testData)
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


"""This function simply teaches how to create a meshgrid that will be used when creating the 3d graph of hyperplane"""
def meshGrid():
    fig = plt.figure()
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    plt.plot(X, Y, marker='.', color='k', linestyle='none')
    fig.savefig("visualizations/meshGrid.png")


def getZ(x,y, params):
    return params[0] + params[1] * x + params[2] * y

"""The model graphs that are being made here are the graphs of hyperplanes,
where hyperplane is given by z = intercept_ + coeff_1 * feature_1 + coeff_2*feature2"""
def makeModelGraph():
    print("\n\n******************** M A K I N G    G R A P H    M O D E L ******************")
    print("=========================Each Logistic Regression Model =========================")
    print(eachModel)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ind, params in enumerate(eachModel):
        x = np.arange(visualization.my_iris_data[:,2].min() - 1, visualization.my_iris_data[:,2].max() + 1, 0.1)
        y = np.arange(visualization.my_iris_data[:,3].min() - 1, visualization.my_iris_data[:,3].max() + 1, 0.1)
        X, Y = np.meshgrid(x, y)
        z = np.array([getZ(x,y,params) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z=z.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Z= b0 + b1 * X1 + b2 * X2')
        fig.savefig("visualizations/" + 'plot' + str(ind) + ".png")


"""Simply put, Decision Boundary is the edge of hyperplane on 2d axes, that is, when the third dimension z=0
 =>intercept_ + coeff_1 * feature_1 + coeff_2*feature2 = 0
 =>feature2 = -(intercept_ + coeff_1*feature1)/coeff_2
 This function plots test data along with decision boundaries."""
def plotTestDataAndDeciBoundary(data):
    fig = plt.figure()
    petlenSetosa = data[:10, 2]
    petwidthSetosa = data[:10, 3]
    petlenVersicolor = data[10:20, 2]
    petwidthVersicolor = data[10:20, 3]
    petlenVirginica = data[20:30, 2]
    petwidthVirginica = data[20:30, 3]
    plt.plot(petlenSetosa, petwidthSetosa, 'ro', petlenVersicolor, petwidthVersicolor, 'bo', petlenVirginica, petwidthVirginica, 'go')
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    colors=['r--','b--', 'g--']
    for ind, params in enumerate(eachModel):
        def getx2(x1):
            return (-(params[0]+params[1]*x1)/params[2])
        plt.plot([visualization.my_iris_data[:,2].min() - 1, visualization.my_iris_data[:,2].max() + 1],
                 [getx2(visualization.my_iris_data[:,2].min() - 1), getx2(visualization.my_iris_data[:,2].max() + 1)],
                 colors[ind])
        fig.savefig("visualizations/" + 'decide' + str(ind) + ".png")


# Series of operations required from training to testing to necessary visualizations
trainMachine()
coefficients = multiClassLogistic.coef_
intercepts = np.transpose([multiClassLogistic.intercept_])
eachModel = np.concatenate((intercepts, coefficients),1)
# meshGrid()
makeModelGraph()
testMachine()


