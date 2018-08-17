# INCEPTION MODULE CODING IN KERAS
# Dependcies 
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Input, Conv2D, MaxPooling2D,Flatten,Dense
from keras.models import Model, model_from_json
from keras.optimizers import SGD
import os

# Data preprocessing
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32') # converting entries to floats
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train) # converting the outputs to binary matrix
y_test = np_utils.to_categorical(y_test)

# Building the NN
input_img=Input(shape=(32,32,3))

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
# Notice here the different ways that we have altered the input and that we have recorded each in a global  variable.
# Normally, in CNNs, you just have one global variable (input) that you keep processing and changing in the network to achieve the output.
# Padding is kept same throughout to be able to concatenate the layers

output=keras.layers.concatenate([tower_1,tower_2,tower_3],axis=3)
output= Flatten()(output)
out=Dense(10,activation='softmax')(output)

# Model
model=Model(inputs=input_img,outputs=out)
#print(model.summary()) for visualizing the network and its parameters

# Setting parameters
epochs =25
lrate=0.01
decay=lrate/epochs
sgd= SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)

# Compiling
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Storage of model in json file, and  result in HDF5 file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(os.getcwd(), 'model.h5'))

# Evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
