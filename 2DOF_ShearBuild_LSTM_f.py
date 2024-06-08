"""Import Dependencies"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
import os
import joblib  # save scaler

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, Activation  # CuDNNLSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
from random import shuffle

# # Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

# Set the GPU device
# gpu_device = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_device[0], True)

"""Loading the Data from matlab already preprocessed or setted up data"""
# data directory
current_directory = os.getcwd()  # Get the current directory
dataDir = os.path.dirname(current_directory)  # Get the parent directory

# dataDir = os.getcwd()
# dataDir = r"C:\Users\aaziz\Documents\MATLAB\PhD Python Code\AASC_DeepLSTM"  # Replace the directory

# mat folder ( matlab)
# mat = scipy.io.loadmat(dataDir + '/data/OLD_data/data_2DOF_SB_BWWN.mat')
mat = scipy.io.loadmat(dataDir + '/LSTMsForNonlinearStructuralSystems/data_2DOF_SB_BWWN.mat')

"""Known Data"""

# Get all data from "input_tf" input data: Dim: Number or Records x Time Steps x DOF (for input, earthquake so DOF = 1)
X_data = mat['input_tf']
# Get all output data from "input_tf" output data: Number or Records x Time Steps x DOF
y_data = mat['target_tf']
# Specify training indices to use for training the LSTM
train_indices = mat['trainInd'] - 1
# Specify testing indices to use for testing the LSTM
test_indices = mat['valInd'] - 1
# Flatten the Data: reshape by multiplying the dimensions, make it linear an array
X_data_flatten = np.reshape(X_data, [X_data.shape[0] * X_data.shape[1], 1])
# Create a MinMax Scaler between -1 and 1
scaler_X = MinMaxScaler(feature_range=(-1, 1))
# Fit the scaler_
scaler_X.fit(X_data_flatten)
# Transform through the scaler
X_data_flatten_map = scaler_X.transform(X_data_flatten)
# Reshape the data back to original dimension before scalling
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], 1])
# Flatten the output data
y_data_flatten = np.reshape(y_data, [y_data.shape[0] * y_data.shape[1], y_data.shape[2]])
# Create a MinMax Scaler between -1 and 1 for y
scaler_y = MinMaxScaler(feature_range=(-1, 1))
# Fit the scaler_
scaler_y.fit(y_data_flatten)
# Transform through the scaler
y_data_flatten_map = scaler_y.transform(y_data_flatten)
# Reshape the data back to original dimension before scaling
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

# """UnKnown Data"""
#
# # New Data to Predict
# X_pred = mat['input_pred_tf']
# # Reference Solution or Recorded values
# y_pred_ref = mat['target_pred_tf']
# # Scaling Process
# X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0] * X_pred.shape[1], 1])
# X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
# X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], 1])
#
# y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0] * y_pred_ref.shape[1], y_pred_ref.shape[2]])
# y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
# y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

X_data_new = X_data_map
y_data_new = y_data_map

" SPECIFYING TRAINING DATA"
X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]

" SPECIFYING TESTING DATA"
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

# X_pred = X_pred_map
# y_pred_ref = y_pred_ref_map

" Number of Input features and Number of file_name_out features"
data_dim = X_train.shape[2]  # number of input features
timesteps = X_train.shape[1]
num_classes = y_train.shape[2]  # number of output features
batch_size = 5

# Optimizers to use
# rms = RMSprop(learning_rate=0.001, decay=0.0001)
# adam = Adam(learning_rate=0.001, decay=0.0001)
rms = RMSprop(learning_rate=0.001)
adam = Adam(learning_rate=0.001)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""SETTING UP THE LONG SHORT TERM MEMORY NN"""

# Type of model: Sequential
model = Sequential()

# LSTM first Layer: LSTM 1
model.add(LSTM(100, return_sequences=True, stateful=False, input_shape=(None, data_dim)))
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNLSTM
# Activation Layer (have relu)
model.add(Activation('relu'))
# model.add(Dropout(0.2))

# LSTM first Layer: LSTM 2
model.add(LSTM(100, return_sequences=True, stateful=False))
# Activation Layer (have relu)
model.add(Activation('relu'))
# model.add(Dropout(0.2))

# FC 1 :  adds a fully connected layer with 100 neurons to the model.
model.add(Dense(100))
# model.add(Activation('relu'))

# FC 2 :  adds a fully connected layer with 5 neurons to the model: SAME AS NDOF/num_classes etc
model.add(Dense(num_classes))

# Print Summary of the model
model.summary()

# Compiling the model
model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_squared_error, mean_absolute_error
              optimizer=adam,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])

best_loss = 100
train_loss = []
test_loss = []
history = []

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Limit GPU memory growth
#     tf.config.experimental.set_memory_growth(gpus[0], True)

# Configure TensorFlow to allow dynamic GPU memory growth
# gpus = tf.config.list_physical_devices('CPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# with tf.device('/device:GPU:1'):

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# # tf.Session(config=tf.ConfigProto(log_device_placement=True))

start = time.time()
# Setting the Number of Epochs
epochs = 10

# Model
dataSpec = "BWWNGATest"
# Strcuturre
structSpec = "2SSB"

strings = [dataSpec, structSpec, 'epochs', str(epochs)]
# Model Name
nameMod = "_".join(strings)

for e in range(epochs):  # Loop through the epochs

    # print the epoch number
    print('epoch = ', e + 1)

    # create a list number of samples
    Ind = list(range(len(X_data_new)))

    # shuffles this list
    shuffle(Ind)

    # create a split ratio ( 70% in this case)
    ratio_split = 0.7

    # Indice the train and testing
    Ind_train = Ind[0:round(ratio_split * len(X_data_new))]
    Ind_test = Ind[round(ratio_split * len(X_data_new)):]

    X_train = X_data_new[Ind_train]
    y_train = y_data_new[Ind_train]
    X_test = X_data_new[Ind_test]
    y_test = y_data_new[Ind_test]

    # Fitting the model to the training Data
    model.fit(X_train, y_train,
              batch_size=batch_size,
              # validation_split=0.2,
              validation_data=(X_test, y_test),
              shuffle=True,
              epochs=1)
    score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    train_loss.append(score0[0])
    test_loss.append(score[0])

    if test_loss[e] < best_loss:
        best_loss = test_loss[e]
        model.save(dataDir + 'results/2DOF_SB (LSTM-f)/' + nameMod + '.h5')

    end = time.time()
    running_time = (end - start) / 3600
    print('Running Time: ', running_time, ' hour')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




'''
A/B Testing
'''

def build_model_A(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, stateful=False, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dense(output_shape))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mse'])
    return model

def build_model_B(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, stateful=False, input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(LSTM(50, return_sequences=True, stateful=False))
    model.add(Activation('tanh'))
    model.add(Dense(50))
    model.add(Dense(output_shape))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.001), metrics=['mse'])
    return model

input_shape = (None, data_dim)
output_shape = num_classes

# initialization
model_A = build_model_A(input_shape, output_shape)
model_B = build_model_B(input_shape, output_shape)

# training
history_A = model_A.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test), shuffle=True)
history_B = model_B.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test), shuffle=True)

# eval
score_A = model_A.evaluate(X_test, y_test, batch_size=batch_size)
print(f'Model A - Test loss: {score_A[0]} / Test mse: {score_A[1]}')
score_B = model_B.evaluate(X_test, y_test, batch_size=batch_size)
print(f'Model B - Test loss: {score_B[0]} / Test mse: {score_B[1]}')

# '''
# Drift Detection
# '''

# def detect_drift(data, threshold=0.01):
#     """
#     using CUSUM
#     """
#     meanLoss = np.mean(data)
#     pos = np.zeros(len(data))
#     neg = np.zeros(len(data))
#     theDrift = []

#     for i in range(1, len(data)):
#         pos[i] = max(0, pos[i-1] + (data[i] - meanLoss))
#         neg[i] = min(0, neg[i-1] + (data[i] - meanLoss))
        
#         if pos[i] > threshold or neg[i] < -threshold:
#             theDrift.append(i)
#             pos[i] = 0
#             neg[i] = 0

#     return theDrift


