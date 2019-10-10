import numpy as np

class NeuralNetwork:
    def __init__(self, NNodes):
        self.NNodes = NNodes  # the number of nodes in the hidden layer
        self.activate = sigmoid  # a function used to activate
        self.deltaActivate = sigmoidDeriv  # the derivative of activate 这个是sigmoidDeriv

        self.input_nodes = 2  # number of input nodes
        self.output_nodes = 2  # number of output nodes

        self.W1 = np.random.randn(self.input_nodes, NNodes)/np.sqrt(self.input_nodes)
        self.W2 = np.random.randn(NNodes+1, self.output_nodes)/np.sqrt(NNodes)


    def fit(self, X, Y, learningRate, epochs, regLambda):
        """
        This function is used to train the model.
        Parameters
        ----------
        X : numpy matrix
            The matrix containing sample features for training.
        Y : numpy array
            The array containing sample labels for training.
        Returns
        -------
        None
        """
        # Initialize your weight matrices first.
        # (hint: check the sizes of your weight matrices first!)
        # For each epoch, do
        for e in range(epochs):
            for i in range(X.shape[0]):
                self.backpropagate(X[i], Y[i],learningRate)
            learningRate = learningRate * (1 - regLambda)

    def predict(self, X):
        """
        Predicts the labels for each sample in X.
        Parameters
        X : numpy matrix
            The matrix containing sample features for testing.
        Returns
        -------
        YPredict : numpy array
            The predictions of X.
        ----------
        """
        YPredict = []
        for x in X:
            YPredict.append(self.forward(x))
        return YPredict

    def forward(self, X):
        # Perform matrix multiplication and activation twice (one for each layer).
        # (hint: add a bias term before multiplication)

        # X = matrix of input nodes
        # W1 = matrix of weight between input and first hidden layer
        # W2 = matrix of weight between second hidden layer and output layer
        a2 = self.activate(np.dot(X, self.W1))
        a2 = np.concatenate((np.ones(1).T, np.array(a2)))
        yhat = self.activate(np.dot(a2, self.W2))

        return yhat

    def backpropagate(self, X, YTrue, learningRate):
        a2 = self.activate(np.dot(X, self.W1))
        a2 = np.concatenate((np.ones(1).T, np.array(a2)))
        a3 = self.activate(np.dot(a2, self.W2))

        costderiv3 = np.multiply(self.deltaActivate(a3), YTrue - a3)
        costderiv2 = np.multiply(self.deltaActivate(a2), np.dot(self.W2, costderiv3))

        self.update3 = np.multiply(learningRate, np.outer(a2, costderiv3))
        self.update2 = np.multiply(learningRate, np.outer(X, costderiv2[1:]))

        self.W2 += self.update3
        self.W1 += self.update2

# incase of use: activate, deltaActivate

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoidDeriv(x):
    return sigmoid(x)*(1.0-sigmoid(x))



def KFold(X, Y, K = 5):
    index = np.arange(len(X))
    index = np.random.shuffle(index)
    return index


def test(XTest, model):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """

    YPredict = model.predict(XTest)
    YPredict = np.argmax(YPredict, axis=1).astype(int)
    return YPredict

def plotDecisionBoundary(model, X, Y):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    np.plt.contourf(x1_array, x2_array, Z, cmap=np.plt.cm.bwr)
    np.plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=np.plt.cm.bwr)
    np.plt.show()

def train(XTrain, YTrain, args):
    """
    This function is used for the training phase.
    Parameters
    ----------
    XTrain : numpy matrix
        The matrix containing samples features (not indices) for training.
    YTrain : numpy array
        The array containing labels for training.
    args : List
        The list of parameters to set up the NN model.
    Returns
    -------
    NN : NeuralNetwork object
        This should be the trained NN object.
    """
    # 1. Initializes a network object with given args.
    model = NeuralNetwork(args[0])

    # 2. Train the model with the function "fit".
    # (hint: use the plotDecisionBoundary function to visualize after training)
    model.fit(XTrain, YTrain, args[1], args[2], args[3])
    plotDecisionBoundary(model, XTrain, YTrain)

    # 3. Return the model.
    return model


def getConfusionMatrix(YTrue, YPredict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(YTrue)):
        if YTrue[i] == 1 and YPredict[i] == 1:
            TP += 1
        if YTrue[i] == 1 and YPredict[i] == 0:
            FN += 1
        if YTrue[i] == 0 and YPredict[i] == 1:
            FP += 1
        if YTrue[i] == 0 and YPredict[i] == 0:
            TN += 1
    CM = np.array([[TP, FP], [FN, TN]])
    return CM

    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """


def getPerformanceScores(YTrue, YPredict):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    CM = getConfusionMatrix(YTrue, YPredict)
    accuracy = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])
    precision = CM[0][0] / (CM[0][0] + CM[0][1])
    recall = CM[0][0] / (CM[0][0] + CM[1][0])
    f1 = (2 * recall * precision) / (recall + precision)
    return {"CM": CM,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1}

