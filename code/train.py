import os
import sys
import random as rn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from functions import addNDVI, normalizedDifference, cleanExperimentFolder, defineArgParsersTrain

# Fix the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)
import keras.backend as k
sess = tf.get_default_session()
k.set_session(sess)

import keras
from keras.optimizers import adam
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard


def trainModel(args):

	# Get the label for each class
	labels = args.labels.split(',')
	num_classes = len(labels)

	# dataset path
	parentPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	trainFilePath = os.path.join(parentPath,args.datasets_path,args.trainDataName)

	# Load the csv tables and get the dataframes
	trainDataFrame = pd.read_csv(trainFilePath)
	print('csv for training loaded')

	# Get a numpy array of the dataframe
	train_n = trainDataFrame.to_numpy()

	x_train = train_n[:,:-1]
	y_train = train_n[:,-1]

	# Get the x_val and y_val. It will do the shuffle
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
	test_size=0.2, random_state=seed) # 80% train, 20% val

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	# Add NDVI
	x_train = addNDVI(x_train)
	x_val = addNDVI(x_val)

	# Number of features
	num_features_input = x_train.shape[1]

	# Convert string list to int list
	args.nNeurons = list(map(int, args.nNeurons.split(',')))
	nLayers = len(args.nNeurons)

	# Experiment folder and name
	nameModel = 'lr%.1e-bs%d-drop%.2f-hla%d-hne%s-epo%d' % (args.learning_rate,args.batch_size,
		args.percentageDropout,nLayers,str(args.nNeurons),args.epochs)
	fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
	experimentPath = os.path.join(args.experiment_folder,args.experiment_name,nameModel)

	# Create the folder of experiments where we will save the models
	if not(os.path.exists(experimentPath)):
	  os.makedirs(experimentPath)
	else:
		print("experiment %s ignored. Aborting..." % (nameModel))
		sys.exit()

	callbacks = []
	callbacks.append(ModelCheckpoint(os.path.join(experimentPath,fileExtension),
		monitor=args.monitor_modelCheckPoint, save_best_only=True, mode='min', verbose=1))
	callbacks.append(TensorBoard(log_dir=os.path.join(experimentPath, 'logs'), 
	                             write_graph=True))
	callbacks.append(EarlyStopping(monitor=args.monitor_stop, min_delta=args.min_delta, 
		patience=args.patience_stop, verbose=1))
	callbacks.append(ReduceLROnPlateau(monitor=args.monitor_reduce_lr, factor=0.1, 
		patience=args.patience_reduce_lr, min_lr=1e-08))

	# Create the model
	model = Sequential()
	model.add(Dense(args.nNeurons[0], activation=tf.nn.relu, input_dim=num_features_input))

	# Check if the user has, in the input, more than one hidden layer
	if nLayers > 1:
		for i in range(1,nLayers):
			model.add(Dense(args.nNeurons[i], activation=tf.nn.relu))

	# Check if the user wants to use a dropout layer
	if args.percentageDropout > 0.0:
		model.add(Dropout(args.percentageDropout))
	model.add(Dense(num_classes, activation='softmax'))

	# Show the neural net
	print(model.summary())

	# Compiling the neural network
	model.compile(
	    optimizer=adam(lr=args.learning_rate), 
	    loss=args.loss_function, 
	    metrics =['accuracy'])

	# Training the model
	history = model.fit(
	    x=x_train,
	    validation_data=(x_val,y_val),
	    y=y_train,
	    batch_size=args.batch_size, 
	    epochs=args.epochs, 
	    shuffle=args.shuffle,
	    callbacks=callbacks,
	    verbose=1)

	cleanExperimentFolder(experimentPath)

def main():

	args = defineArgParsersTrain()

	trainModel(args)

if __name__ == "__main__":
	main()
