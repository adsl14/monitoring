import ee
import time
import os
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from functions import addNDVI, normalizedDifference, cleanExperimentFolder, defineArgParsers

# Fijar semillas para reproducibilidad de resultados (antes de importar Keras)
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)

import keras
from keras.optimizers import adam
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard


def trainModel(args):

	# VARS
	trainFileName = 'Training_demo.csv'
	testFileName = 'Testing_demo.csv'

	# LOAD THE TABLES

	# Get the root path of the project
	main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	csv_path = os.path.join(main_path,'csv')

	# Load the csv tables and get the dataframes
	trainDataFrame = pd.read_csv(os.path.join(csv_path,trainFileName))
	print('csv for training loaded')
	testDataFrame = pd.read_csv(os.path.join(csv_path,testFileName))
	print('csv for testing loaded')

	# Get a numpy array of the dataframe
	train_n = trainDataFrame.to_numpy()
	test_n = testDataFrame.to_numpy()

	# shuffle only the train dataset
	np.random.shuffle(train_n)

	x_train = train_n[:,:-1]
	y_train = train_n[:,-1]
	x_test = test_n[:,:-1]
	y_test = test_n[:,-1]

	# Get the x_val and y_val
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) # 80% train, 20% val

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, args.num_classes)
	y_val = keras.utils.to_categorical(y_val, args.num_classes)
	y_test = keras.utils.to_categorical(y_test, args.num_classes)

	# Add NDVI
	x_train = addNDVI(x_train)
	x_test = addNDVI(x_test)
	x_val = addNDVI(x_val)

	# CREATE THE MODEL

	# Number of features
	num_features_input = x_train.shape[1]

	# hyperparameters
	lr = 1e-03
	batch_size = 16
	epochs = 80
	shuffle = True
	percentageDropout = 0.2
	HiddenLayers = 2
	HiddenNeurons = [4,4]
	loss_function = 'categorical_crossentropy'
	metrics = 'accuracy'

	# Experiment folder and name
	nameExperimentsFolder = 'experiments'
	nameExperiment = 'land_classification'
	nameModel = 'lr%.1e-bs%d-drop%.2f-hla%d-hne%s' % (lr,batch_size,percentageDropout,HiddenLayers,str(HiddenNeurons))
	fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
	experimentPath = os.path.join(nameExperimentsFolder,nameExperiment,nameModel)

	# Create the folder of experiments where we will save the models
	if not(os.path.exists(experimentPath)):
	  os.makedirs(experimentPath)

	# Callback parameters
	monitor_stop = 'acc' # What the model will check in order to stop the training
	monitor_reduce_lr = 'loss' # What the model will check in order to change the learning rate
	min_delta = 1e-03 # Minimum change in the monitored quantity to qualify as an improvement
	patience_stop = 50 # Number of epochs with no improvement after which training will be stopped.
	patience_reduce_lr = 8 # Number of epochs with no improvement after which lr will be reduced.

	callbacks = []
	callbacks.append(ModelCheckpoint(os.path.join(experimentPath,fileExtension),
		monitor='val_loss', save_best_only=True, mode='min', verbose=1))
	callbacks.append(TensorBoard(log_dir=os.path.join(experimentPath, 'logs'), 
	                             write_graph=True))
	callbacks.append(EarlyStopping(monitor=monitor_stop, min_delta=min_delta, patience=patience_stop, 
		verbose=1))
	callbacks.append(ReduceLROnPlateau(monitor=monitor_reduce_lr, factor=0.1, patience=patience_reduce_lr, 
		min_lr=1e-08))

	# Create the model
	model = Sequential()
	model.add(Dense(64, activation=tf.nn.relu, input_dim=num_features_input))
	model.add(Dropout(percentageDropout))
	model.add(Dense(args.num_classes, activation='softmax'))

	# Show the neural net
	print(model.summary())

	# Compiling the neural network
	model.compile(
	    optimizer=adam(lr=lr), 
	    loss='categorical_crossentropy', 
	    metrics =['accuracy'])

	# Training the model
	history = model.fit(
	    x=x_train,
	    validation_data=(x_val,y_val),
	    y=y_train,
	    batch_size=batch_size, 
	    epochs=epochs, 
	    shuffle=shuffle,
	    callbacks=callbacks,
	    verbose=1)

	cleanExperimentFolder(experimentPath)

def main():

	args = defineArgParsers()

	trainModel(args)

if __name__ == "__main__":
	main()
