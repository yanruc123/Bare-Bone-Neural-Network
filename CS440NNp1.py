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




