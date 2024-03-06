# neural network class skeleton code
import numpy as np
import matplotlib.pyplot as plt


class myNeuralNetwork(object):
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.01):
        """__init__
        Class constructor: Initialize the parameters of the network including
        the learning rate, layer sizes, and each of the parameters
        of the model (weights, placeholders for activations, inputs,
        deltas for gradients, and weight gradients). This method
        should also initialize the weights of your model randomly
            Input:
                n_in:          number of inputs
                n_layer1:      number of nodes in layer 1
                n_layer2:      number of nodes in layer 2
                n_out:         number of output nodes
                learning_rate: learning rate for gradient descent
            Output:
                none
        """
        # initialize input
        self.n_in = n_in
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_out = n_out
        self.learning_rate = learning_rate

        # initialize weights with small random values for each layer
        self.W1 = np.random.randn(self.n_in, self.n_layer1) * 0.01
        self.W2 = np.random.randn(self.n_layer1, self.n_layer2) * 0.01
        self.W_out = np.random.randn(self.n_layer2, self.n_out) * 0.01

        self.b1 = np.random.randn(1, self.n_layer1) * 0.01
        self.b2 = np.random.randn(1, self.n_layer2) * 0.01
        self.b_out = np.random.randn(1, self.n_out) * 0.01

    def forward_propagation(self, x):
        """forward_propagation
        Takes a vector of your input data (one sample) and feeds
        it forward through the neural network, calculating activations and
        layer node values along the way.
            Input:
                x: a vector of data representing 1 sample [n_in x 1]
            Output:
                y_hat: a vector (or scaler of predictions) [n_out x 1]
                (typically n_out will be 1 for binary classification)
        """
        # input to first hidden layer
        self.a1 = np.dot(x, self.W1) + self.b1
        self.z1 = self.sigmoid(self.a1)

        # First hidden layer to second hidden layer
        self.a2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = self.sigmoid(self.a2)

        # Second hidden layer to output layer
        self.z_out = np.dot(self.z2, self.W_out) + self.b_out
        y_hat = self.sigmoid(self.z_out)  # final output prediction

        # store activations for use in backpropagation
        # self.y_hat = y_hat

        return y_hat

    def compute_loss(self, y, y_hat):
        """compute_loss
        Computes the current loss/cost function of the neural network
        based on the weights and the data input into this function.
        To do so, it runs the X data through the network to generate
        predictions, then compares it to the target variable y using
        the cost/loss function
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                loss: a scalar measure of loss/cost
        """
        # Compute the cross-entropy loss
        m = y.shape[0]  # Number of samples
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
        return loss

    def backpropagate(self, x, y):
        """backpropagate
        Backpropagate the error from one sample determining the gradients
        with respect to each of the weights in the network. The steps for
        this algorithm are:
            1. Run a forward pass of the model to get the activations
               Corresponding to x and get the loss functionof the model
               predictions compared to the target variable y
            2. Compute the deltas (see lecture notes) and values of the
               gradient with respect to each weight in each layer moving
               backwards through the network

            Input:
                x: A vector of 1 samples of data [n_in x 1]
                y: Target variable [scalar]
            Output:
                loss: a scalar measure of th loss/cost associated with x,y
                      and the current model weights
        """

        # Step 1: Forward pass
        y_hat = self.forward_propagation(x)
        loss = self.compute_loss(y, y_hat)

        # Step 2: Backward pass to compute gradients
        # Output layer to second hidden layer
        dLoss_yHat = y_hat - y
        dZ3 = dLoss_yHat * self.sigmoid_derivative(self.z_out)
        dW_out = np.dot(self.a2.T, dZ3) / y.shape[0]
        db_out = np.sum(dZ3, axis=0, keepdims=True) / y.shape[0]

        # Second hidden layer to first hidden layer
        dA2 = np.dot(dLoss_yHat, self.W_out.T)
        dZ2 = dA2 * self.sigmoid_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dZ2) / y.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / y.shape[0]

        # First hidden layer to input layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(x.T, dZ1) / y.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / y.shape[0]

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.W_out -= self.learning_rate * dW_out
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2
        self.b_out -= self.learning_rate * db_out

        return loss

    # def stochastic_gradient_descent_step(self):
    #     """stochastic_gradient_descent_step [OPTIONAL - you may also do this
    #     directly in backpropagate]
    #     Using the gradient values computed by backpropagate, update each
    #     weight value of the model according to the familiar stochastic
    #     gradient descent update equation.

    #     Input: none
    #     Output: none
    #     """
    #     # Update weights and biases for each layer based on the gradients computed in backpropagation
    #     # and the learning rate
    #     # Check if gradients exist
    #     if (
    #         hasattr(self, "dW1")
    #         and hasattr(self, "dW2")
    #         and hasattr(self, "dW_out")
    #         and hasattr(self, "db1")
    #         and hasattr(self, "db2")
    #         and hasattr(self, "db_out")
    #     ):
    #         # Update weights for each layer
    #         self.W1 -= self.learning_rate * self.dW1
    #         self.W2 -= self.learning_rate * self.dW2
    #         self.W_out -= self.learning_rate * self.dW_out

    #         # Update biases for each layer
    #         self.b1 -= self.learning_rate * self.db1
    #         self.b2 -= self.learning_rate * self.db2
    #         self.b_out -= self.learning_rate * self.db_out
    #     else:
    #         print("Gradients not available. Please run backpropagation first.")

    def fit(
        self,
        X,
        y,
        max_epochs=10,
        learning_rate=0.01,
        get_validation_loss=False,
        X_val=None,
        y_val=None,
    ):
        """fit
        Input:
            X: A matrix of N samples of data [N x n_in]
            y: Target variable [N x 1]
        Output:
            training_loss:   Vector of training loss values at the end of each epoch
            validation_loss: Vector of validation loss values at the end of each epoch
                             [optional output if get_validation_loss==True]
        """
        training_loss = []
        validation_loss = []
        self.learning_rate = learning_rate
        for epoch in range(max_epochs):
            # Initialize total loss for the epoch
            epoch_loss = 0

            # Iterate over individual samples
            for i in range(X.shape[0]):
                # Perform a forward pass and backpropagation
                x_sample = X[i].reshape(1, -1)  # Reshape to ensure it's a column vector
                y_sample = y[i].reshape(1, -1)
                loss = self.backpropagate(x_sample, y_sample)
                epoch_loss += loss
            # Average the loss over all samples and store it
            avg_epoch_loss = epoch_loss / X.shape[0]
            training_loss.append(avg_epoch_loss)

            # Optionally, compute validation loss
            if get_validation_loss and X_val is not None and y_val is not None:
                val_loss = 0
                for i in range(X_val.shape[0]):
                    y_hat = self.forward_propagation(X_val[i].reshape(1, -1))
                    val_loss += self.compute_loss(y_val[i].reshape(1, -1), y_hat)
                validation_loss.append(val_loss / X_val.shape[0])

            # Print progress
            print(
                f"Epoch {epoch+1}/{max_epochs}, Training Loss: {avg_epoch_loss}", end=""
            )
            if get_validation_loss:
                print(f", Validation Loss: {validation_loss[-1]}", end="")

        # Return the training and optionally validation loss
        return training_loss, validation_loss

    def predict_proba(self, X):
        """predict_proba
        Compute the output of the neural network for each sample in X, with the last layer's
        sigmoid activation providing an estimate of the target output between 0 and 1
            Input:
                X: A matrix of N samples of data [N x n_in]
            Output:
                y_hat: A vector of class predictions between 0 and 1 [N x 1]
        """
        # Step 1: Forward pass through the first hidden layer
        z1 = np.dot(X, self.W1) + self.b1.T  # Adjust dimensions if necessary
        a1 = self.sigmoid(z1)

        # Step 2: Forward pass through the second hidden layer
        z2 = np.dot(a1, self.W2) + self.b2.T  # Adjust dimensions if necessary
        a2 = self.sigmoid(z2)

        # Step 3: Forward pass through the output layer
        z_out = np.dot(a2, self.W_out) + self.b_out.T  # Adjust dimensions if necessary
        y_hat = self.sigmoid(z_out)  # Final predictions

        return y_hat

    def predict(self, X, decision_thresh=0.5):
        """predict
        Compute the output of the neural network prediction for
        each sample in X, with the last layer's sigmoid activation
        providing an estimate of the target output between 0 and 1,
        then thresholding that prediction based on decision_thresh
        to produce a binary class prediction
            Input:
                X: A matrix of N samples of data [N x n_in]
                decision_threshold: threshold for the class confidence score
                                    of predict_proba for binarizing the output
            Output:
                y_hat: A vector of class predictions of either 0 or 1 [N x 1]
        """
        # Step 1: Obtain probability estimates for each sample
        proba = self.predict_proba(X)

        # Step 2: Threshold the probabilities to get binary class predictions
        y_hat = (proba >= decision_thresh).astype(
            int
        )  # Convert boolean array to integers (0 or 1)

        return y_hat

    def sigmoid(self, X):
        """sigmoid
        Compute the sigmoid function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid function
        """
        # Our activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        """sigmoid_derivative
        Compute the sigmoid derivative function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid derivative function
        """
        return self.sigmoid(X) * (1 - self.sigmoid(X))


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_train = 500
N_test = 100
X, y = make_moons(N_train, noise=0.20, random_state=42)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=N_test / N_train, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

nn = myNeuralNetwork(n_in=2, n_layer1=5, n_layer2=5, n_out=1, learning_rate=0.01)

# Train the model and collect training and validation losses
training_loss, validation_loss = nn.fit(
    X_train,
    y_train,
    max_epochs=100,
    learning_rate=0.01,
    get_validation_loss=True,
    X_val=X_val,
    y_val=y_val,
)

# Plotting the cost function

plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Curves")
plt.show()
