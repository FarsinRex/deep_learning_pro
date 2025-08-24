import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0,z)
def derivative_relu(z):
    return np.where(z>0,1,0)

def hyperbolic_tanget(z):
    return np.tanh(z)
def hyperbolic_derivative(z):
    #the derivative of tanh function is sech**(x) and its equal to (1-tanh(x)**2)
    return 1-np.tanh(z)**2

z = np.linspace(-10,10,400)
hyper_grad, relu_grad = hyperbolic_derivative(z), derivative_relu(z)

plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, hyperbolic_tanget(z), label='Hyperbolic tangent Activation', color='b')
plt.plot(z, hyper_grad, label="Hyperbolic Derivative", color='r', linestyle='--')
plt.title('hyperbolic tangent Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()
  
