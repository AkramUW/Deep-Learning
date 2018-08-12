# My First Neural Net
This isn't actually mine, however, it's recreated from memory after reading Andrew Trask's blog [post](https://iamtrask.github.io/2015/07/12/basic-python-network/ "post") . I've also added some context and comments to help anybody who reads this to better understand the neural net.
> Our only dependency would be the [Numpy](http://www.numpy.org/) Library for Scientific Computing

## OKAY so let's get into it!

### Our Context
Let's say we have data from a Yes/No survey in terms of 1s and 0s, where 1 is yes and 0 is no to  the following 4 questions:

1. Do you drink at least 8 cups of water per day ?
2. Do you eat more than 1500 mg of salt per day ?
3. Are you above 40 years of age ?
4. Do you have any sort of heart disease ?

>*This is a purely hypothetical, just to add context to the data*

The survey gave us the following data, each row represents one person's answers:

|Water ?| Salt ?  | 40+ ?  | Heart Disease ?
| :-------------:|:-------------:| :-----:|:----:|
|  0    | 0 | 1 |0
| 1     | 1      |   1 |1
| 1 | 0 |    1 |1
|0|1|1|0

### Set-up

We aim to use this data to predict if someone has heart disease or not if we're given answers to the questions above.

We can first transform our table into input and output.

|Input 1| Input 2  | Input 3  | Output
| :-------------:|:-------------:| :-----:|:----:|
|  0    | 0 | 1 |0
| 1     | 1      |   1 |1
| 1 | 0 |    1 |1
|0|1|1|0

This table will serve as our training set. We'll be using [Python](https://www.python.org/downloads/) to build our model, so a good in-code representation would be a numpy array (i.e. a matrix).
   
``` python 
 Input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
 Output = np.array([[0,1,1,0]]).T 
 ```

We now would like to intialize random [synaptic weights](https://en.wikipedia.org/wiki/Synaptic_weight) for our neural net. Briefly, synaptic weights are weights associated with each input determining how much does it contribute to the output.

``` python 
syn0= 2 * np.random.random((3,4))-1
syn1= 2 * np.random.random((4,1))-1 
```

Now we got it all set, except for one part which is the [activation function](https://en.wikipedia.org/wiki/Activation_function) that's going to map the weighted sum of the input to the range between 0 and 1, transforming the input (in a way) to a probability. Here we use the sigmoid function as our activation function, since our data is binary.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/2000px-Sigmoid-function-2.svg.png)


``` python 
 def sigmoid(x, deriv=False):
     if (deriv== True):
          return x*(1-x)
     else:
          return 1/(1+np.exp(-x)) 
```
> Notice that we embedded our function's [derivative](https://www.youtube.com/watch?v=9vKqVkMQHKk) into the function body, as it's going to be used later on

### Training

We now have everything we need to train our model, we're going to do use by feeding forward the input to our model, calculating the error then adjusting our weights to minimize the error at each instance. Let's take it step by step.

We're going to use 60,000 training instances to train our model, this means we're going to make our currently random synaptic weights better 60,000 times by minimizing the error at each instance. Then we'll have 3 layers in our neural net. 
The first layer is our input
The second layer is going to predict the output, by applying the sigmoid function to weighted sum of our input.
The third layer is going to refine the prediction and spit out the output
```python 
 for iter in range(60000):
     l0 = Input
     l1 = sigmoid(np.dot(l0,syn0)) 
     l2 = sigmoid(np.dot(l1,syn1))
```

Now we have predicted the output, but did we really ? Let's see the error
` l2_error = Output - l2 `
We're going to use this error to calculate an adjustment matrix or delta that we're going to use to see how far are we from the output and  to update our synaptic weights at each instance, this is called **Back Propagation**
``` python 
	 l2_delta = l2_error*sigmoid(l2,deriv=True)
     l1_error = l2_delta.dot(syn1.T)
     l1_delta = l1_error * sigmoid(l1,deriv=True)

     syn0+= np.dot(l0.T, l1_delta)
     syn1+= np.dot(l1.T, l2_delta)

```

### Testing

Now we have done all the training for our model, we gave all the protein shakes, all the running, everything. Let's see how it would predict for an unseen case, which is a person who drinks 8 cups of water per day, eats more than 1500mg of salt per day but is under the age of 40. Let's make a matrix.

` new_guy = np.array([[1,1,0]])`

In order to predict we must first build a predictor function 
```python
def predictor(X):
     l1 = sigmoid(np.dot(X,syn0))
     l2 = sigmoid(np.dot(l1,syn1)) 
     print("Heart Disease Chance : {}".format(l2))
```

When I ran ` predictor(new_guy)` in Python, it has predicted that `new_guy` has an 0.006 chance of having heart disease, and I'm not a doctor so I don't know if this good or bad. But the main idea here is the framework of building, training and testing a neural net. But a useful metric to see if our neural net is good is determining if the `l2_error` decreases over time or not, you can do that by running the .py file in this repository.

Wish you the best,
Akram


