import numpy as np
def sigmoid(x, deriv=False):
     if (deriv== True):
          return x*(1-x)
     else:
          return 1/(1+np.exp(-x))
     
# Input 
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

# Output
Y = np.array([[0,1,1,0]]).T


np.random.seed(1)

# Synaptic Weights
syn0= 2 * np.random.random((3,4))-1
syn1= 2 * np.random.random((4,1))-1

for iter in range(60000):
     # Forward Propagation
     l0 = X
     l1 = sigmoid(np.dot(l0,syn0))
     l2=sigmoid(np.dot(l1,syn1))
     
     # errors & deltas
     l2_error = Y - l2
     if iter % 10000 == 0:
          print("L2_error={0}".format(np.mean(np.abs(l2_error))))
          
     l2_delta = l2_error*sigmoid(l2,deriv=True)
     l1_error = l2_delta.dot(syn1.T) # Back Propagation
     l1_delta = l1_error * sigmoid(l1,deriv=True)
   
     #Update weights
     syn0+= np.dot(l0.T, l1_delta)
     syn1+= np.dot(l1.T, l2_delta)
     
new_guy = np.array([[1,1,0]])

def predictor(X):
     l1 = sigmoid(np.dot(X,syn0))
     l2 = sigmoid(np.dot(l1,syn1)) 
     print("Heart Disease Chance : {}".format(l2))