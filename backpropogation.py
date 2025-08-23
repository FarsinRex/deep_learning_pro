import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
d = np.array([0, 1, 1,0])  # ground truth output of XOR

#declaring the network parameters and weights
def initialize_network():
    
    #network parameters
    inputSize = 2
    hiddenSize = 2
    outputSize = 1
    lr = 0.1
    epochs = 18000
    w1 = np.random.rand(hiddenSize,inputSize)*2-1
    b1 = np.random.rand(hiddenSize,1)*2-1
    w2 = np.random.rand(outputSize,hiddenSize)*2-1
    b2 = np.random.rand(outputSize,1)*2-1
    
    return w1,b1,w2,b2,lr,epochs

w1, b1, w2, b2, lr, epochs = initialize_network()
error_list = []
for epoch in range(epochs):
    z1 =np.dot(w1, X) + b1  # Weighted sum for hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

    z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

    # Error calculation and backpropagation
    error = d - a2  # Difference between expected and actual output
    da2 = error * (a2 * (1 - a2))  # Derivative for output layer
    dz2 = da2  # Gradient for output layer

    # Propagate error to hidden layer
    da1 = np.dot(w2.T, dz2)  # Gradient for hidden layer
    dz1 = da1 * (a1 * (1 - a1))
    
    w2 += lr * np.dot(dz2, a1.T)  # Update weights for output layer
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)  # Update biases for output layer
    
    w1 += lr * np.dot(dz1, X.T)  # Update weights for hidden layer
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)  # Update biases for hidden layer
    
    if (epoch+1)%10000 == 0:
        print("epoch: %d, average error: %0.05f" % (epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))
    
#testing the neural network
z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

# Print results
print('Final output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

#plotting the error
plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()