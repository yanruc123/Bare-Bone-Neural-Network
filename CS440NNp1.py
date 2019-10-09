import numpy as np
import os, sys

class NeuralNetwork:
    def __init__(self, NNodes, activate, deltaActivate):
        self.NNodes = NNodes # the number of nodes in the hidden layer
        self.activate = activate # a function used to activate
        self.deltaActivate = deltaActivate # the derivative of activate 这个是sigmoidDeriv


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
        self.input_nodes = 2  #number of input nodes
        self.output_nodes = 1  #number of output nodes
        self.learningRate = learningRate

        self.W1 = np.random.rand(input_nodes,NNodes)
        self.W2 = np.random.rand(NNodes,output_nodes)
        self.regLambda = regLambda

        #record 2 list of testing result
        forward_output = []
        back_output = []
        
        # For each epoch, do
        for e in epochs:
            # For each training sample (X[i], Y[i]), do
            for i in range(0,len(X)):
                # 1. Forward propagate once. Use the function "forward" here!
                forward_output.append(forward(X[i]))
                
                # 2. Backward progate once. Use the function "backpropagate" here!
                back_output.append(backpropagate(X[i],Y[i]))

        pass
        

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
        #不是已经iterate over X in fit了吗，为什么还要predict the labels for each sample in X
        YPredict = []
        for x in X:
            YPredict.append(forward(x))
        return YPredict

    def forward(self, X):
        # Perform matrix multiplication and activation twice (one for each layer).
        # (hint: add a bias term before multiplication)

        # X = matrix of input nodes
        # W1 = matrix of weight between input and first hidden layer
        # W2 = matrix of weight between second hidden layer and output layer
        self.z2 = np.dot(X,self.W1)
        self.a2 = activate(self.z2)
        self.z3 = np.dot(a2,W2)
        yhat = activate(self.z3)

        return yhat
        
    def backpropagate(self,X,YTrue):
        YPredict = predict(X)

        # Compute loss / cost using the getCost function.
        cost = getCost(X, YTrue, YPredict)
                
        # Compute gradient for each layer.
        costderiv3 = ((-1)*(YPredict-YTrue))*deltaActivate(self.z3)
        costderiv2 = np.dot(np.dot(costderiv3,np.transpose(W2)),deltaActivate(self.z2)) 

        #lambda*weight is added to djdw for regularization to prevent overfitting 
        # how to use learning rate: gradiant*learning_rate
        djdw2 = np.dot(np.transpose(self.a2),costderiv3) + self.regLambda*self.W2  
        djdw1 = np.dot(np.transpose(X),costderiv2) + self.regLambda*self.W1
        
        # Update weight matrices.cs
        self.W1 += djdw1*learningRate
        self.W2 += djdw2*learningRate

        pass
        
    def getCost(self, X, YTrue, YPredict):
        # Compute loss / cost in terms of crossentropy.
        # (hint: your regularization term should appear here)

        # for each term in (ypredict - ytrue)^2 *0.5
        result = 0.5*(YTrue-YPredict)**2/X.shape + (self.regLambda/2)*(sum(self.W1**2)+sum(self.W2**2))
        return result

def getData(dataDir):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''
    # TO-DO for this part:
    # Use your preferred method to read the csv files.
    # Write your codes here:
    
    
    # Hint: use print(X.shape) to check if your results are valid.
    return X, Y

def splitData(X, Y, K = 5):
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    '''
    
    # Make sure you shuffle each train list.
    pass

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
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

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
    model = NeuralNetwork(args[0],args[1],args[2])
    
    # 2. Train the model with the function "fit".
    # (hint: use the plotDecisionBoundary function to visualize after training)
    X = XTrain
    Y = YTrain
    learningRate = 0.1
    epochs = 10
    regLambda = 0.1
    model.fit(X, Y, learningRate, epochs, regLambda)

    plotDecisionBoundary(model,X,Y)
    
    # 3. Return the model.
    return model


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
    pass

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
    CM = [TP,FP,FN,TN]
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
    accuracy = (CM[0]+CM[3])/(CM[0]+CM[1]+CM[2]+CM[3])
    precision = CM[0]/(CM[0]+CM[1])
    recall = CM[0]/(CM[0]+CM[2])
    f1 = (2*recall*precision)/(recall+precision)
    return {"CM" : CM,
    "accuracy" : accuracy,
    "precision" : precision,
    "recall" : recall,
    "f1" : f1}

# incase of use: activate, deltaActivate
def sigmoid(self, z):
        return 1/(1+np.exp(-z))

def sigmoidDeriv(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)


