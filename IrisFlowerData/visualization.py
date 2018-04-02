import numpy as np
from matplotlib import pyplot


def visualize(x1, y1, x2, y2, x3, y3, xLabel, yLabel, figName):
    fig = pyplot.figure()
    pyplot.plot(x1, y1, 'ro', x2, y2, 'bo', x3, y3, 'go')
    pyplot.xlabel(xLabel)
    pyplot.ylabel(yLabel)
    pyplot.show()
    fig.savefig("visualizations/" + figName + ".png")

# Extracting the training data set to develop a predictive model
# Fields (from left):SepalLength, SepalWidth, PetalLength, PetalWidth, FlowerName
# Codes Provided to Iris Flowers in Test Data: Setosa = 0 ; Versicolor = 1 ; Virginica = 2
my_iris_data = np.genfromtxt('IrisDataSetTrain.csv', delimiter=',')

#individual features extraction
seplenSetosa=my_iris_data[:40, 0]
sepwidthSetosa=my_iris_data[:40, 1]
petlenSetosa=my_iris_data[:40, 2]
petwidthSetosa=my_iris_data[:40, 3]

seplenVersicolor=my_iris_data[40:80, 0]
sepwidthVersicolor=my_iris_data[40:80, 1]
petlenVersicolor=my_iris_data[40:80, 2]
petwidthVersicolor=my_iris_data[40:80, 3]

seplenVirginica=my_iris_data[80:120, 0]
sepwidthVirginica=my_iris_data[80:120, 1]
petlenVirginica=my_iris_data[80:120, 2]
petwidthVirginica=my_iris_data[80:120, 3]

#Visualizing the data

# y= Sepal Length
visualize(sepwidthSetosa, seplenSetosa, sepwidthVersicolor, seplenVersicolor, sepwidthVirginica, seplenVirginica, "Sepal Width", "Sepal Length", "sepLength_vs_sepWidth")
visualize(petwidthSetosa, seplenSetosa, petwidthVersicolor, seplenVersicolor, petwidthVirginica, seplenVirginica, "Petal Width", "Sepal Length", "sepLength_vs_petalWidth")
visualize(petlenSetosa, seplenSetosa, petlenVersicolor, seplenVersicolor, petlenVirginica, seplenVirginica, "Petal Length", "Sepal Length", "sepLength_vs_petalLength")

# y= Sepal Width
visualize(petwidthSetosa, sepwidthSetosa, petwidthVersicolor, sepwidthVersicolor, petwidthVirginica, sepwidthVirginica, "Petal Width", "Sepal Width", "sepWidth_vs_petalWidth")
visualize(petlenSetosa, sepwidthSetosa, petlenVersicolor, sepwidthVersicolor, petlenVirginica, sepwidthVirginica, "Petal Length", "Sepal Width", "sepWidth_vs_petalLength")

# y= Petal Width
visualize(petlenSetosa, petwidthSetosa, petlenVersicolor, petwidthVersicolor, petlenVirginica, petwidthVirginica, "Petal Length", "Petal Width", "petalWidth_vs_petalLength")

