from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json

f = open('params.json')
params = json.load(f)
weights = params['weights']
biases = params['biases']
f.close()

(trainx, trainy), (testx, testy) = mnist.load_data()

# normalization of image data
trainx = (trainx * (0.99 / 255)) + 0.01
# flattening all 28x28 matrices into 784 long vectors. Result is 60000x784 input matrix of train data
modified_x = []
for a in trainx:
    s = np.array(a)
    modified_x.append(s.flatten())
modified_x = np.array(modified_x)

# one-hot encoding of all outputs. Mapping numbers to indices in a binary one-hot vector for classification
one_hot = []
for i in trainy:
    temp = np.zeros(10)
    temp[i] = np.float(1)
    one_hot.append(temp)


class Network:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = np.array(outputs)

        # initialize random weights and biases
        self.bias = biases
        self.weights = weights
        self.error_list = []
        self.epochs = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cr_loss(self, x, y):
        # multiclass cross-entropic loss: log sum for each output averaged over m=60000 outputs.
        sum = np.sum(np.multiply(y, np.log(x)))
        return (-1 / self.inputs.shape[0]) * sum

    def feedforward(self):
        self.layer = self.sigmoid((np.matmul(self.inputs, self.weights)) + self.bias)

    def backpropagation(self, learning_rate):
        self.error = self.cr_loss(self.layer, self.outputs)

        # final derivatives for both weights and biases.
        delta_w = (1 / self.inputs.shape[0]) * np.matmul(self.inputs.T, (self.layer - self.outputs))
        delta_b = (1 / self.inputs.shape[0]) * np.sum(self.layer - self.outputs)
        self.weights -= learning_rate * delta_w
        self.bias -= learning_rate * delta_b

    # add batches: feedforward multiple images, backprop once
    def train(self, epochs, learning_rate):
        for e in range(epochs):
            self.feedforward()
            self.backpropagation(learning_rate)
            self.error_list.append(np.average(self.error))
            self.epochs.append(e)
            if e % 100 == 0:
                print(e, ": ", self.error)
        print("Final error:", self.error)

    def predict(self, test_input):
        predicted = self.sigmoid(np.matmul(test_input, self.weights) + self.bias)
        return predicted


NN = Network(modified_x[0:40000], one_hot[0:40000])
# play around with: learning rate, different loss function (non cross-entropy), validation accuracy, overfitting and regularization, COMPARISONS
NN.train(1000, 1)

dict = {"weights": np.array(NN.weights).tolist(), "biases": NN.bias.tolist()}
with open('params.json', 'w') as out:
    json.dump(dict, out)


answers = []
for q in modified_x[40000:]:
    answers.append(np.argmax(NN.predict(q)))
# print(answers, np.array(trainy[40000:]).shape)
print(confusion_matrix(answers, np.array(trainy[40000:]).T))
print(classification_report(answers, np.array(trainy[40000:]).T))

plt.figure(figsize=(15, 5))
plt.plot(NN.epochs, NN.error_list)
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()
