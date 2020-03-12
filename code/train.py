import pandas as pd
import time
import os
import re
import argparse
import numpy as np
import random as rn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing # For normalization

from pickle import dump # Save scaler
from pickle import load # Load scaler
from datetime import datetime as dateTime
from matplotlib import pyplot as plt

#Fix the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)
import keras.backend as k

import keras
from keras.optimizers import adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input, Flatten, LSTM, CuDNNLSTM, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, GlobalAveragePooling1D, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard



# Give to the GPU what percentage will use
def configureKerasForGPU(percentage):

	# Configure keras backend GPU management
	tf_config = tf.ConfigProto()
	# Limit gpu memory utilization to a third of total
	tf_config.gpu_options.per_process_gpu_memory_fraction = percentage
	# tf_config.gpu_options.allow_growth = True
	# Log on which device the operation runs
	tf_config.log_device_placement = False
	sess = tf.Session(config=tf_config)
	k.set_session(sess)

# Check if a string will return 'True' or 'False'
def str2bool(val):

	if val.lower() in ('yes', 'true', 't', 'y', '1'):
		return True

	elif val.lower() in ('no', 'false', 'f', 'n', '0'):
		return False

	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

# All the parameters that has the script
def defineArgParsers():

	parser = argparse.ArgumentParser(description='Generate a model.')
	parser.add_argument("--network",type=str, default='LSTM_p_CNN', help="Select the network you want to use (LSTM_p_CNN, LSTM+CNN, CNN+LSTM, LSTM)")
	parser.add_argument("--percentageGPU",type=float, default=0.0, help="Amount of use the memory of the GPU")
	parser.add_argument("--learning_rate",type=float, default=1e-4, help="Learning rate modifier.")
	parser.add_argument("--batch_size",type=int, default=16, help="Size of batch (number of samples) to evaluate")
	parser.add_argument("--epochs",type=int, default=100, help="Number of epochs.")
	parser.add_argument("--percentageDropout",type=float, default=0.2, help="How many links of the network will be ommited in order to avoid overfitting")
	parser.add_argument("--patience",type=int, default=30, help="Number of epochs with no improvement after which training will be stopped.")
	parser.add_argument("--patience_reduce_lr",type=int, default=8, help="Num epochs to reduce learning rate.")
	parser.add_argument("--nNeuronsSequence",type=str, default="128", help="Number of units in the LSTM layer and number of LSTM layers")
	parser.add_argument("--nNeuronsConv1D",type=str, default="64,64", help="Number of kernels in the Convolutional layer and number of Convolutionals layers")
	parser.add_argument("--nNeurons",type=str, default="64", help="Number of neurons at the end of the network (hidden layers)")
	parser.add_argument("--loss_function",type=str, default="categorical_crossentropy", help="loss function (categorical_crossentropy)")
	parser.add_argument("--shuffle",type=str2bool, default="y", help="Whether to shuffle the order of the batches at the beginning of each epoch.")
	parser.add_argument("--min_delta",type=float, default=1e-3, help="Minimum change in the monitored quantity to qualify as an improvement.")
	parser.add_argument("--campaingsFull",type=str2bool, default="n", help="using all the campaings to train (split train/test each campaing) or using a few for train, and one for test.")

	return parser.parse_args()

def show_confussionMatrix(matrix,labels):

	row = matrix.shape[0]
	cols = matrix.shape[1]

	print('Real | Predicted | Amount')

	for i in range(0,row):
		for j in range(0,cols):
			print("%s | %s | %d" % (labels[i],labels[j],matrix[i,j]))
   
def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

def atoi(text):

	return int(text) if text.isdigit() else text

def natural_keys(text):

	return [ atoi(c) for c in re.split('(\d+)',text) ]

# When the network has finished the training, all the generated models will be erased, unless the best one.
def cleanExperimentFolder(folderNameExperimentPath):

  models = os.listdir(folderNameExperimentPath)

  # Sort by natural order
  models.sort(key=natural_keys,reverse=True)

  # Remove from the list, the folder 'logs'
  models = models[1:]

  # We check if there is, at least, one min model saved
  if len(models) > 1:
    for i in range (1,len(models)):
      model_path = os.path.join(folderNameExperimentPath,models[i])
      os.remove(model_path)
      print("Experiment %s removed" %(model_path))
  else:
    print("Folder %s ignored" %(folderNameExperimentPath))

  # Get best model name
  return models[0]

def searchModelInFile(model_name,file_pointer):

  for row in file_pointer:
    if model_name == row[0]:
      return True, row[1:]

  return False, []

def WriteResultsModel(best_model_path,output_writer, x_test, y_test, labels):
  
  # Load the best model for that experiment
  model = load_model(best_model_path)

  # For CNN+LSTM, we had changed the input shape (n_samples, substeps, steps, features)
  if "CNN" == best_model_path.split("/")[-2][0:3]:
    x_test = x_test.reshape((x_test.shape[0], 1, steps, features))

  # Get the predictions
  predictions = model.predict(x_test)

  # Confusion matrix
  cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

  # Evaluate test data
  score = model.evaluate(x_test, y_test, verbose=0)

  print("RESULTS")
  print("------------------------")
  print("Confusion matrix")
  show_confussionMatrix(cm,labels)
  print("------------------------")
  print("Score")
  print("Test loss:", score[0])
  print("Test accuracy:", str(round(score[1]*100,2)) + ' %')
  print("------------------------")

  output_writer.writerow([best_model_path, score[0], str(round(score[1]*100,2)) + ' %'])
  print("Model %s results saved correctly \n \n" % (best_model_path))
 
def plot_history(history):

  plt.figure(figsize=(8, 5))
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.yscale('log')
  plt.plot(history.epoch,np.array(history.history['loss']),label='categorical_crossentropy (training)')
  plt.plot(history.epoch,np.array(history.history['val_loss']),label='categorical_crossentropy (validation)')
  plt.legend()

  max_y = max(max(np.array(history.history['loss'])), max(np.array(history.history['val_loss'])))

def showCorrelativeMatrix(df):
  plt.figure(figsize=(12,10))
  cor = df.corr()
  sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
  plt.show()

def add_Dense_Layer(number, data):
  
  data = Dense(number, activation='relu',
                kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data)  
  return data

def add_LSTM_Layer(number,return_sequence,data):
  
  data = LSTM(number, activation='relu', return_sequences=return_sequence,
              kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
              recurrent_initializer=keras.initializers.glorot_uniform(seed=seed),
              bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data) 
  return data

def add_CuDNNLSTM_Layer(number,return_sequence,data):

  data = CuDNNLSTM(number, return_sequences=return_sequence,
              kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
              recurrent_initializer=keras.initializers.glorot_uniform(seed=seed),
              bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data) 
  return data

def add_Conv1D_Layer(number,data):

  data = Conv1D(filters=number, kernel_size=1,data_format='channels_last', activation='relu', 
              kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
              bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data)
  
  return data

def add_Conv1DTimeDistributed_Layer(number,data):

  data = TimeDistributed(Conv1D(filters=number, kernel_size=1,data_format='channels_last', activation='relu', 
              kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
              bias_initializer=keras.initializers.glorot_uniform(seed=seed)))(data)
  
  return data

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

def showSamples(tags_name):

	num_trainSamples = 0
	num_testSamples = 0

	for campaing in campaings[:-1]:

	  # Read the train and test .csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,tags_name))
	  tagDataFrameTrain = tagDataFrameTrain[tagDataFrameTrain[labels_header] != -1]

	  num_trainSamples = num_trainSamples + tagDataFrameTrain[labels_header].value_counts()

	  print(campaing)

	# Read the train and test .csv
	tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaings[-1],tags_name))
	tagDataFrameTest = tagDataFrameTest[tagDataFrameTest[labels_header] != -1]
	nameColumns = tagDataFrameTest.columns

	num_testSamples = num_testSamples + tagDataFrameTest[labels_header].value_counts()

	print(campaings[-1])

	print("Entrenamiento")
	print(num_trainSamples)
	total_train = num_trainSamples.sum()
	print("Total %d" %(total_train))

	print("")
	print("Test")
	print(num_testSamples)
	total_test = num_testSamples.sum()
	print("Total %d" %(total_test))

def splitTrainTestCampaings(test_size=0.3,*,campaings,path_radar,tags_name, labels_header):

	for campaing in campaings:

		path_train = os.path.join(path_radar,campaing,"train.csv")
		path_test = os.path.join(path_radar,campaing,"test.csv")

		if os.path.exists(path_train) and os.path.exists(path_test):

			print("Campaña: %s. Split ya realizado." %(campaing))
			continue

		else:

			# Read 'tags.csv' and clean tag dataframe. Ignore the samples that were marked as -1
			tagDataFrame = pd.read_csv(os.path.join(path_radar,campaing,tags_name))
			tagDataFrame = tagDataFrame[tagDataFrame[labels_header] != -1]

			nameColumns = tagDataFrame.columns

			tagDataFrameName = tagDataFrame[nameColumns[0]]
			tagDataFrameActivity = tagDataFrame[nameColumns[1:]]

			# Get the train and test split. It will shuffle automatically. 70% train, 30% test  
			dfName_train, dfName_test, dfTarget_train, dfTarget_test = train_test_split(tagDataFrameName,tagDataFrameActivity, test_size=0.3, random_state=seed, stratify=tagDataFrameActivity)

			df_train = pd.concat([dfName_train, dfTarget_train], axis=1)
			pd.DataFrame.to_csv(df_train,index=False,path_or_buf=os.path.join(path_radar,campaing,"train.csv"))
			print("Conjunto de train guardado correctamente en %s" %(os.path.join(path_radar,campaing,"train.csv")))

			df_test = pd.concat([dfName_test, dfTarget_test], axis=1)
			pd.DataFrame.to_csv(df_test,index=False,path_or_buf=os.path.join(path_radar,campaing,"test.csv"))
			print("Conjunto de test guardado correctamente en %s" %(os.path.join(path_radar,campaing,"test.csv")))

def loadSamplesFull(tags_name, labels, indexes, campaings, path_radar,labels_header,interpolate):

	now = time.time()

	num_classes = len(labels)
	num_features = len(indexes)
	sequencesTrain = []
	sequencesTest = []
	targetsTrain = []
	targetsTest = []

	for campaing in campaings:

	  # Read the train and test .csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,"train.csv"))
	  print("'train.csv' cargado correctamente")

	  tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaing,"test.csv"))
	  print("'test.csv' cargado correctamente")

	  total_train = tagDataFrameTrain.shape[0]
	  total_test = tagDataFrameTest.shape[0]

	  # Get the sequence for each area
	  i = 1
	  for row in tagDataFrameTrain.values:

	    region_path = row[0]
	    areadf = pd.read_csv(os.path.join(path_radar,campaing,'dataset',region_path))
	    areadf = areadf[indexes].dropna(how='all')

	    if(interpolate):
	      areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()
	    
	    sequencesTrain.append(areadf.values)
	    targetsTrain.append(row[1:])

	    print("---Entrenamiento---")
	    print(campaing)
	    print("Progreso %d/%d" %(i,total_train))
	    print("Recinto %s cargado." %(region_path))
	    i=i+1

	  i = 1
	  for row in tagDataFrameTest.values:

	    region_path = row[0]
	    areadf = pd.read_csv(os.path.join(path_radar,campaing,'dataset',region_path))
	    areadf = areadf[indexes].dropna(how='all')

	    if(interpolate):
	      areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()

	    sequencesTest.append(areadf.values)
	    targetsTest.append(row[1:])

	    print("---Test---")
	    print(campaing)
	    print("Progreso %d/%d" %(i,total_test))
	    print("Recinto %s cargado." %(region_path))
	    i=i+1

	# Get the length of the time series for each area in order to get the max values of the time series
	len_sequences = []
	sequences = sequencesTrain + sequencesTest
	for one_seq in sequences:
	    len_sequences.append(len(one_seq))
	seriesDescription = pd.Series(len_sequences).describe()
	max_valueSerie = int(seriesDescription[-1])
	min_valueSerie = int(seriesDescription[3])

	# Change time_step value
	time_step = max_valueSerie
	print("Using %d time_step" %(time_step))

	#Padding the sequence with the values in last row to max length.
	# If all the sequences has the same time_step, we don't fix any custom time_step
	if max_valueSerie == min_valueSerie:

	  x_train = np.array(sequencesTrain)
	  x_test = np.array(sequencesTest)

	else:

	  print("Generating fix time_step for train")
	  new_seq = []
	  for one_seq in sequencesTrain:
	      len_one_seq = len(one_seq)
	      n = time_step - len_one_seq

	      to_concat = np.repeat(one_seq[-1], n).reshape(num_features, n).transpose()
	      new_one_seq = np.concatenate([one_seq, to_concat])
	      new_seq.append(new_one_seq)
	  x_train = np.array(new_seq)

	  print("Generating fix time_step for test")
	  new_seq = []
	  for one_seq in sequencesTest:
	      len_one_seq = len(one_seq)
	      n = time_step - len_one_seq

	      to_concat = np.repeat(one_seq[-1], n).reshape(num_features, n).transpose()
	      new_one_seq = np.concatenate([one_seq, to_concat])
	      new_seq.append(new_one_seq)
	  x_test = np.array(new_seq)                       

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(targetsTrain, num_classes)
	y_test = keras.utils.to_categorical(targetsTest, num_classes)

	end = time.time()

	elapsed = end - now
	time_convert(elapsed)

	# Get number of samples
	unique_samples_train, num_trainSamples = np.unique(y_train, axis=0, return_counts=True)
	unique_samples_test, num_testSamples = np.unique(y_test, axis=0, return_counts=True)

	print("")
	print("Entrenamiento")
	print(unique_samples_train)
	print(num_trainSamples)
	print("Total: %d" %(num_trainSamples.sum()))

	print("")
	print("Test")
	print(unique_samples_test)
	print(num_testSamples)
	print("Total: %d" %(num_testSamples.sum()))

	return x_train, y_train, x_test, y_test, time_step, num_features, num_classes

def loadSamples(tags_name, labels, indexes, campaings, path_radar,labels_header,interpolate):

	now = time.time()

	num_classes = len(labels)
	num_features = len(indexes)
	sequencesTrain = []
	sequencesTest = []
	targetsTrain = []
	targetsTest = []

	# --- TRAIN BLOCK ---
	for campaing in campaings[:-1]:

	  # Read the train and test .csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,tags_name))
	  tagDataFrameTrain = tagDataFrameTrain[tagDataFrameTrain[labels_header] != -1]
	  print("'tags.csv' de entrenamiento cargado correctamente")

	  total_train = tagDataFrameTrain.shape[0]

	  # Get the sequence for each area
	  i = 1
	  for row in tagDataFrameTrain.values:

	    region_path = row[0]
	    areadf = pd.read_csv(os.path.join(path_radar,campaing,'dataset',region_path))
	    areadf = areadf[indexes].dropna(how='all')

	    if(interpolate):
	      areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()
	    
	    sequencesTrain.append(areadf.values)
	    targetsTrain.append(row[1:])

	    print("---Entrenamiento---")
	    print(campaing)
	    print("Progreso %d/%d" %(i,total_train))
	    print("Recinto %s cargado." %(region_path))
	    i=i+1

	# --- TEST BLOCK ---
	# Read the test .csv
	tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaings[-1],tags_name))
	tagDataFrameTest = tagDataFrameTest[tagDataFrameTest[labels_header] != -1]
	print("'tags.csv' de test cargado correctamente")

	total_test = tagDataFrameTest.shape[0]

	# Get the sequence for each area (TEST)
	i = 1
	for row in tagDataFrameTest.values:

	  region_path = row[0]
	  areadf = pd.read_csv(os.path.join(path_radar,campaings[-1],'dataset',region_path))
	  areadf = areadf[indexes].dropna(how='all')

	  if(interpolate):
	    areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()
	  
	  sequencesTest.append(areadf.values)
	  targetsTest.append(row[1:])

	  print("---Test---")
	  print(campaings[-1])
	  print("Progreso %d/%d" %(i,total_test))
	  print("Recinto %s cargado." %(region_path))
	  i=i+1

	# Get the length of the time series for each area in order to get the max values of the time series
	len_sequences = []
	sequences = sequencesTrain + sequencesTest
	for one_seq in sequences:
	  len_sequences.append(len(one_seq))
	seriesDescription = pd.Series(len_sequences).describe()
	max_valueSerie = int(seriesDescription[-1])
	min_valueSerie = int(seriesDescription[3])

	# Change time_step value
	time_step = max_valueSerie
	print("Using %d time_step" %(time_step))

	#Padding the sequence with the values in last row to max length.
	# If all the sequences has the same time_step, we don't fix any custom time_step
	if max_valueSerie == min_valueSerie:

	  x_train = np.array(sequencesTrain)
	  x_test = np.array(sequencesTest)

	else:

	  print("Generating fix time_step for train")
	  new_seq = []
	  for one_seq in sequencesTrain:
	      len_one_seq = len(one_seq)
	      n = time_step - len_one_seq

	      to_concat = np.repeat(one_seq[-1], n).reshape(num_features, n).transpose()
	      new_one_seq = np.concatenate([one_seq, to_concat])
	      new_seq.append(new_one_seq)
	  x_train = np.array(new_seq)

	  print("Generating fix time_step for test")
	  new_seq = []
	  for one_seq in sequencesTest:
	      len_one_seq = len(one_seq)
	      n = time_step - len_one_seq

	      to_concat = np.repeat(one_seq[-1], n).reshape(num_features, n).transpose()
	      new_one_seq = np.concatenate([one_seq, to_concat])
	      new_seq.append(new_one_seq)
	  x_test = np.array(new_seq)                       

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(targetsTrain, num_classes)
	y_test = keras.utils.to_categorical(targetsTest, num_classes)


	end = time.time()

	elapsed = end - now
	time_convert(elapsed)

	# Get number of samples
	unique_samples_train, num_trainSamples = np.unique(y_train, axis=0, return_counts=True)
	unique_samples_test, num_testSamples = np.unique(y_test, axis=0, return_counts=True)

	print("")
	print("Entrenamiento")
	print(unique_samples_train)
	print(num_trainSamples)
	print("Total: %d" %(num_trainSamples.sum()))

	print("")
	print("Test")
	print(unique_samples_test)
	print(num_testSamples)
	print("Total: %d" %(num_testSamples.sum()))

	return x_train, y_train, x_test, y_test, time_step, num_features, num_classes

def normalize_data(x_train, y_train, x_test, y_test, nameExperimentsFolder, nameExperiment, experimentFolder):

	path_scaler = os.path.join(nameExperimentsFolder, nameExperiment, experimentFolder + '-scaler.pkl')

	if not os.path.exists(path_scaler):

		# Normalize data
		samples_train, steps, features = x_train.shape
		samples_test = x_test.shape[0]

		x_train = x_train.reshape((samples_train,steps*features),order='F')
		x_test = x_test.reshape((samples_test,steps*features),order='F')

		# Change for  MinMaxScaler, StandardScaler, Normalizer
		scaler = preprocessing.MinMaxScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)

		x_train = np.reshape(x_train, (samples_train, steps, features),order='F')
		x_test = np.reshape(x_test, (samples_test, steps, features),order='F')

		# Shuffle the x_train samples
		x_train, y_train = shuffle(x_train, y_train, random_state=seed)

		# Save scaler
		dump(scaler, open(path_scaler, 'wb'))
		print("Scaler saved correctly")

	else:

		print("Scaler already saved")

		# Normalize data
		samples_train, steps, features = x_train.shape
		samples_test = x_test.shape[0]

		scalerPath = os.path.join(nameExperimentsFolder, nameExperiment, experimentFolder + '-scaler.pkl')

		scaler = load(open(scalerPath, 'rb'))

		x_train = x_train.reshape((samples_train,steps*features),order='F')
		x_test = x_test.reshape((samples_test,steps*features),order='F')

		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)

		x_train = np.reshape(x_train, (samples_train, steps, features),order='F')
		x_test = np.reshape(x_test, (samples_test, steps, features),order='F')

	print("Samples normalized.")

	return x_train, y_train, x_test, y_test

# TRAIN MODELS FUNCTIONS
# LSTM || CNN
def TrainLSTM_p_CNN(lr=1e-03, batch_size=16, epochs=100, percentageDropout=0.0, nNeuronsSequence=[64,64],nNeuronsConv1D=[128,256,128], nNeurons=[16,8],
	shuffle=False, min_delta= 1e-03, patience_stop = 30, patience_reduce_lr = 8, loss_function = 'categorical_crossentropy', metrics = ['categorical_accuracy'], 
	*, x_train, y_train, x_test, y_test, time_step, num_features, num_classes, nameExperimentsFolder, nameExperiment, experimentFolder,campaingsFull):

	# hyperparameters
	#lr = 1e-02
	#batch_size = 16
	#epochs = 100
	#shuffle = False
	#percentageDropout = 0.3
	#nNeurons = [16,8]
	#nNeuronsSequence = [64,64]
	#nNeuronsConv1D = [128,256,128]

	nLayers = len(nNeurons)
	nLayersSequence = len(nNeuronsSequence)
	nLayersConv1D = len(nNeuronsConv1D)

	# date
	date = dateTime.now().strftime("%d:%m:%y:%H:%M:%S")

	# Experiment folder and name
	nameModel = 'LSTM_p_CNN-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-cF_%s' % (lr,batch_size,
	percentageDropout,str(nNeuronsSequence),str(nNeuronsConv1D), str(nNeurons),epochs,time_step,campaingsFull)

	fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
	path_experiment = os.path.join(nameExperimentsFolder,nameExperiment,experimentFolder,nameModel)

	# If the experiment folder already exists, we will ignore it.
	if os.path.exists(path_experiment):
		print('Ignored the experiment %s. This experiment has been used before.' % (path_experiment))

	# The experiment folder doesn't exists
	else:
		os.makedirs(path_experiment)

		# Callback parameters
		monitor_stop = 'val_loss' # What the model will check in order to stop the training
		monitor_reduce_lr = 'val_loss' # What the model will check in order to change the learning rate

		callbacks = []
		callbacks.append(ModelCheckpoint(os.path.join(path_experiment,fileExtension),monitor='val_loss',
		                                save_best_only=True, mode='min', verbose=1))
		callbacks.append(TensorBoard(log_dir=os.path.join(path_experiment,'logs'), write_graph=True))
		callbacks.append(EarlyStopping(monitor=monitor_stop, min_delta=min_delta, patience=patience_stop, verbose=1))
		callbacks.append(ReduceLROnPlateau(monitor=monitor_reduce_lr, factor=0.1, patience=patience_reduce_lr, min_lr=1e-08))

		# Create the model
		k.clear_session()
		input = Input(shape=(time_step,num_features,))

		#--------------
		# LSTM block
		#--------------
		# Check if the user has entered at least one hidden layer sequence
		if nLayersSequence > 0:
		  # Has two hidden LSTM layers
		  if nLayersSequence > 1:
		    x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], True, input)

		    for i in range(1,nLayersSequence-1):
		      x = add_CuDNNLSTM_Layer(nNeuronsSequence[i], True, x)

		    x = add_CuDNNLSTM_Layer(nNeuronsSequence[-1], False, x)

		  # Only one hidden LSTM layer
		  else:
		    x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], False, input)

		  if percentageDropout > 0.0:
		    x = Dropout(percentageDropout)(x)            

		else:
		  print("Please, insert at least one recurrent layer.")
		  assert False


		#--------------
		# CONV1D block
		#--------------
		# Check if the user has entered at least one hidden layer conv1D
		if nLayersConv1D > 0:
		    x_2 = add_Conv1D_Layer(nNeuronsConv1D[0], input)
		    #x_2 = BatchNormalization()(x_2) makes no reproducibility

		    for i in range(1,nLayersConv1D):
		      x_2 = add_Conv1D_Layer(nNeuronsConv1D[i], x_2)
		      #x_2 = BatchNormalization()(x_2) makes no reproducibility

		    x_2 = GlobalAveragePooling1D()(x_2)

		    if percentageDropout > 0.0:
		      x_2 = Dropout(percentageDropout)(x_2)

		else:
		  print("Please, insert at least one conv1D layer.")
		  assert False

		#--------------
		# CONCATENATE LSTM + Conv1D
		#--------------    
		x = Concatenate()([x,x_2])

		#--------------
		# Dense block
		#--------------
		# ADD dense layer
		if nLayers > 0:
		  for i in range(0,nLayers):
		    x = add_Dense_Layer(nNeurons[i], x)

		  # Add dropout before the output layer
		  #if percentageDropout > 0.0:
		    #x = Dropout(percentageDropout)(x)  

		# Output
		output = Dense(num_classes, activation='softmax',
		                kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)

		model = Model(input,output)

		# Show the neural net
		print(model.summary())

		# Compiling the neural network
		model.compile(
		    optimizer=adam(lr=lr), 
		    loss=loss_function, 
		    metrics =metrics)

		# Training the model
		history = model.fit(
		    x=x_train,
		    validation_data=(x_test,y_test),
		    y=y_train,
		    batch_size=batch_size, 
		    epochs=epochs, 
		    shuffle=shuffle,
		    callbacks=callbacks,
		    verbose=1)

		plt.figure(figsize=(10,5))
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.yscale('log')
		  
		# Error de entrenamiento
		plt.plot(history.epoch,np.array(history.history['loss']),label='Loss (train)')
		# Error de validación
		plt.plot(history.epoch,np.array(history.history['val_loss']),label='Loss (val)')

		plt.legend()

		print('|Precisión en Entrenamiento|')
		print("Máximo: ", max(np.array(history.history['categorical_accuracy'])))
		print("Mínimo: ", min(np.array(history.history['categorical_accuracy'])))
		print("Media: ", np.mean(np.array(history.history['categorical_accuracy'])))
		print("Desv. tipica: ", np.std(np.array(history.history['categorical_accuracy'])))

		print("")

		print('|Precisión en Validación|')
		print("Máximo:", max(np.array(history.history['val_categorical_accuracy'])))
		print("Mínimo:", min(np.array(history.history['val_categorical_accuracy'])))
		print("Media:", np.mean(np.array(history.history['val_categorical_accuracy'])))
		print("Desv. tipica:", np.std(np.array(history.history['val_categorical_accuracy'])))

		# Clean the folder where the models are saved
		best_model_name = cleanExperimentFolder(path_experiment)

		# Save figure
		plt.savefig(os.path.join(path_experiment, nameModel + ".png"))

# LSTM -> CNN
def TrainLSTM_CNN(lr=1e-03, batch_size=16, epochs=100, percentageDropout=0.0, nNeuronsSequence=[64,64],nNeuronsConv1D=[128,256,128], nNeurons=[16,8], shuffle=False, min_delta= 1e-03, patience_stop = 30,
	patience_reduce_lr = 8,loss_function = 'categorical_crossentropy', metrics = ['categorical_accuracy'], *, x_train, y_train, x_test, y_test, time_step, num_features, num_classes, 
	nameExperimentsFolder, nameExperiment, experimentFolder,campaingsFull):

  # hyperparameters
  #lr = 1e-02
  #batch_size = 16
  #epochs = 100
  #shuffle = False
  #percentageDropout = 0.3
  #nNeurons = [16,8]
  #nNeuronsSequence = [64,64]
  #nNeuronsConv1D = [128,256,128]

  nLayers = len(nNeurons)
  nLayersSequence = len(nNeuronsSequence)
  nLayersConv1D = len(nNeuronsConv1D)

  # date
  date = dateTime.now().strftime("%d:%m:%y:%H:%M:%S")

  # Experiment folder and name
  nameModel = 'LSTM_CNN-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-cF_%s' % (lr,batch_size,
  percentageDropout,str(nNeuronsSequence),str(nNeuronsConv1D), str(nNeurons),epochs,time_step,campaingsFull)
  
  fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
  path_experiment = os.path.join(nameExperimentsFolder,nameExperiment,experimentFolder,nameModel)

  # If the experiment folder already exists, we will ignore it.
  if os.path.exists(path_experiment):
    print('Ignored the experiment %s. This experiment has been used before.' % (path_experiment))

  # The experiment folder doesn't exists
  else:
    os.makedirs(path_experiment)

    # Callback parameters
    monitor_stop = 'val_loss' # What the model will check in order to stop the training
    monitor_reduce_lr = 'val_loss' # What the model will check in order to change the learning rate

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(path_experiment,fileExtension),monitor='val_loss',
                                    save_best_only=True, mode='min', verbose=1))
    callbacks.append(TensorBoard(log_dir=os.path.join(path_experiment,'logs'), write_graph=True))
    callbacks.append(EarlyStopping(monitor=monitor_stop, min_delta=min_delta, patience=patience_stop, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_reduce_lr, factor=0.1, patience=patience_reduce_lr, min_lr=1e-08))

    # Create the model
    k.clear_session()
    input = Input(shape=(time_step,num_features,))

    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[i], True, x)

      if percentageDropout > 0.0:
            x = Dropout(percentageDropout)(x)  
    else:
      print("Please, insert at least one recurrent layer.")
      assert False

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(nNeuronsConv1D[0], x)
        #x = BatchNormalization()(x) makes no reproducibility

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(nNeuronsConv1D[i], x)
          #x = BatchNormalization()(x) makes no reproducibility

        # Apply global average pooling and make the output only one dimension
        x = GlobalAveragePooling1D()(x)

        if percentageDropout > 0.0:
              x = Dropout(percentageDropout)(x)    

    else:
      print("Please, insert at least one conv1D layer.")
      assert False

    #--------------
    # Dense block
    #--------------
    # ADD dense layer
    if nLayers > 0:
      for i in range(0,nLayers):
        x = add_Dense_Layer(nNeurons[i], x)

      # Add dropout before the output layer
      #if percentageDropout > 0.0:
        #x = Dropout(percentageDropout)(x)  

    # Output
    output = Dense(num_classes, activation='softmax',
                    kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

    # Show the neural net
    print(model.summary())
  
    # Compiling the neural network
    model.compile(
        optimizer=adam(lr=lr), 
        loss=loss_function, 
        metrics = metrics)

    # Training the model
    history = model.fit(
        x=x_train,
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=batch_size, 
        epochs=epochs, 
        shuffle=shuffle,
        callbacks=callbacks,
        verbose=1)
    
    plt.figure(figsize=(10,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
      
    # Error de entrenamiento
    plt.plot(history.epoch,np.array(history.history['loss']),label='Loss (train)')
    # Error de validación
    plt.plot(history.epoch,np.array(history.history['val_loss']),label='Loss (val)')

    plt.legend()

    print('|Precisión en Entrenamiento|')
    print("Máximo: ", max(np.array(history.history['categorical_accuracy'])))
    print("Mínimo: ", min(np.array(history.history['categorical_accuracy'])))
    print("Media: ", np.mean(np.array(history.history['categorical_accuracy'])))
    print("Desv. tipica: ", np.std(np.array(history.history['categorical_accuracy'])))

    print("")

    print('|Precisión en Validación|')
    print("Máximo:", max(np.array(history.history['val_categorical_accuracy'])))
    print("Mínimo:", min(np.array(history.history['val_categorical_accuracy'])))
    print("Media:", np.mean(np.array(history.history['val_categorical_accuracy'])))
    print("Desv. tipica:", np.std(np.array(history.history['val_categorical_accuracy'])))

    # Clean the folder where the models are saved
    best_model_name = cleanExperimentFolder(path_experiment)

    # Save figure
    plt.savefig(os.path.join(path_experiment, nameModel + ".png"))

# CNN -> LSTM
def TrainCNN_LSTM(lr=1e-03, batch_size=16, epochs=100, percentageDropout=0.0,  nNeuronsSequence=[64,64],nNeuronsConv1D=[128,256,128], nNeurons=[16,8], shuffle=False, min_delta= 1e-03, patience_stop = 30,
 patience_reduce_lr = 8, substeps=1,loss_function = 'categorical_crossentropy',  metrics = ['categorical_accuracy'], *, x_train, y_train, x_test, y_test, time_step, num_features, num_classes,
  nameExperimentsFolder, nameExperiment, experimentFolder,campaingsFull):

  # hyperparameters
  #lr = 1e-02
  #batch_size = 16
  #epochs = 100
  #shuffle = False
  #percentageDropout = 0.3
  #nNeurons = [16,8]
  #nNeuronsSequence = [64,64]
  #nNeuronsConv1D = [128,256,128]

  nLayers = len(nNeurons)
  nLayersSequence = len(nNeuronsSequence)
  nLayersConv1D = len(nNeuronsConv1D)

  # date
  date = dateTime.now().strftime("%d:%m:%y:%H:%M:%S")

  # Experiment folder and name
  nameModel = 'CNN_LSTM-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-cF_%s' % (lr,batch_size,
  percentageDropout,str(nNeuronsSequence),str(nNeuronsConv1D), str(nNeurons),epochs,time_step,campaingsFull)
  
  fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
  path_experiment = os.path.join(nameExperimentsFolder,nameExperiment, experimentFolder,nameModel)

  # If the experiment folder already exists, we will ignore it.
  if os.path.exists(path_experiment):
    print('Ignored the experiment %s. This experiment has been used before.' % (path_experiment))

  # The experiment folder doesn't exists
  else:
    os.makedirs(path_experiment)

    # Callback parameters
    monitor_stop = 'val_loss' # What the model will check in order to stop the training
    monitor_reduce_lr = 'val_loss' # What the model will check in order to change the learning rate

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(path_experiment,fileExtension),monitor='val_loss',
                                    save_best_only=True, mode='min', verbose=1))
    callbacks.append(TensorBoard(log_dir=os.path.join(path_experiment,'logs'), write_graph=True))
    callbacks.append(EarlyStopping(monitor=monitor_stop, min_delta=min_delta, patience=patience_stop, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_reduce_lr, factor=0.1, patience=patience_reduce_lr, min_lr=1e-08))

    # Create the model
    k.clear_session()
    input = Input(shape=(substeps,time_step,num_features,))

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[0], input)

        for i in range(1,nLayersConv1D):
          x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[i], x)

          # Add a dropout and a Pooling each 2 conv1D layer
          if i % 2 == 1:
            if percentageDropout > 0.0:
              x = TimeDistributed(Dropout(percentageDropout))(x)
            x = TimeDistributed(MaxPooling1D())(x)

        # Apply flatten
        x = TimeDistributed(Flatten())(x)

    else:
      print("Please, insert at least one conv1D layer.")
      assert False
 
    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      # The user has entered two hidden layers
      if nLayersSequence > 1:
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], True, x)

        for i in range(1,nLayersSequence-1):
          x = add_CuDNNLSTM_Layer(nNeuronsSequence[i], True, x)

        x = add_CuDNNLSTM_Layer(nNeuronsSequence[-1], False, x)
  
      # The user has entered only one hidden layer
      else:
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], False, x)

      # Add dropout layer 
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)
    else:
      print("Please, insert at least one recurrent layer.")
      assert False

    #--------------
    # Dense block
    #--------------
    # ADD dense layer
    if nLayers > 0:
      for i in range(0,nLayers):
        x = add_Dense_Layer(nNeurons[i], x)

      # Add dropout before the output layer
      #if percentageDropout > 0.0:
        #x = Dropout(percentageDropout)(x)  

    # Output
    output = Dense(num_classes, activation='softmax',
                    kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

    # Show the neural net
    print(model.summary())
  
    # Compiling the neural network
    model.compile(
        optimizer=adam(lr=lr), 
        loss=loss_function, 
        metrics = metrics)

    x_train_2 = x_train.reshape((x_train.shape[0], substeps, time_step, num_features))
    x_test_2 = x_test.reshape((x_test.shape[0], substeps, time_step, num_features))

    # Training the model
    history = model.fit(
        x=x_train_2,
        validation_data=(x_test_2,y_test),
        y=y_train,
        batch_size=batch_size, 
        epochs=epochs, 
        shuffle=shuffle,
        callbacks=callbacks,
        verbose=1)
    
    plt.figure(figsize=(10,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
      
    # Error de entrenamiento
    plt.plot(history.epoch,np.array(history.history['loss']),label='Loss (train)')
    # Error de validación
    plt.plot(history.epoch,np.array(history.history['val_loss']),label='Loss (val)')

    plt.legend()

    print('|Precisión en Entrenamiento|')
    print("Máximo: ", max(np.array(history.history['categorical_accuracy'])))
    print("Mínimo: ", min(np.array(history.history['categorical_accuracy'])))
    print("Media: ", np.mean(np.array(history.history['categorical_accuracy'])))
    print("Desv. tipica: ", np.std(np.array(history.history['categorical_accuracy'])))

    print("")

    print('|Precisión en Validación|')
    print("Máximo:", max(np.array(history.history['val_categorical_accuracy'])))
    print("Mínimo:", min(np.array(history.history['val_categorical_accuracy'])))
    print("Media:", np.mean(np.array(history.history['val_categorical_accuracy'])))
    print("Desv. tipica:", np.std(np.array(history.history['val_categorical_accuracy'])))

    # Clean the folder where the models are saved
    best_model_name = cleanExperimentFolder(path_experiment)

    # Save figure
    plt.savefig(os.path.join(path_experiment, nameModel + ".png"))

# LSTM
def TrainLSTM(lr=1e-03, batch_size=16, epochs=100, percentageDropout=0.0, nNeuronsSequence = [64,64],nNeurons=[16,8], shuffle=False,  min_delta= 1e-03, patience_stop = 30, patience_reduce_lr = 8,
	loss_function = 'categorical_crossentropy', metrics = ['categorical_accuracy'], *, x_train, y_train, x_test, y_test, time_step, num_features, num_classes, nameExperimentsFolder, nameExperiment,
	experimentFolder,campaingsFull):

  # hyperparameters
  #lr = 1e-02
  #batch_size = 16
  #epochs = 100
  #shuffle = False
  #percentageDropout = 0.3
  #nNeurons = [16,8]
  #nNeuronsSequence = [64,64]

  nLayers = len(nNeurons)
  nLayersSequence = len(nNeuronsSequence)

  # date
  date = dateTime.now().strftime("%d:%m:%y:%H:%M:%S")

  # Experiment folder and name
  nameModel = 'LSTM-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d-seqLen%d-cF_%s' % (lr,batch_size,percentageDropout,str(nNeuronsSequence),str(nNeurons),epochs,time_step,campaingsFull)
  fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
  path_experiment = os.path.join(nameExperimentsFolder,nameExperiment,experimentFolder,nameModel)

  # If the experiment folder already exists, we will ignore it.
  if os.path.exists(path_experiment):
    print('Ignored the experiment %s. This experiment has been used before.' % (path_experiment))

  # The experiment folder doesn't exists
  else:
    os.makedirs(path_experiment)

    # Callback parameters
    monitor_stop = 'val_loss' # What the model will check in order to stop the training
    monitor_reduce_lr = 'val_loss' # What the model will check in order to change the learning rate

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(path_experiment,fileExtension),monitor='val_loss',
                                    save_best_only=True, mode='min', verbose=1))
    callbacks.append(TensorBoard(log_dir=os.path.join(path_experiment,'logs'), write_graph=True))
    callbacks.append(EarlyStopping(monitor=monitor_stop, min_delta=min_delta, patience=patience_stop, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_reduce_lr, factor=0.1, patience=patience_reduce_lr, min_lr=1e-08))

    # Create the model
    k.clear_session()
    input = Input(shape=(time_step,num_features,))

    #--------------
    # LSTM block
    #--------------
    # ADD Recurrent layer
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      # The user has entered two hidden layers
      if nLayersSequence > 1:
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], True, input)

        for i in range(1,nLayersSequence-1):
          x = add_CuDNNLSTM_Layer(nNeuronsSequence[i], True, x)

        x = add_CuDNNLSTM_Layer(nNeuronsSequence[-1], False, x)
  
      # The user has entered only one hidden layer
      else:
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], False, input)
          
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)
    else:
      print("Please, insert at least one recurrent layer.")
      assert False

    #--------------
    # Dense block
    #--------------
    # ADD dense layer
    if nLayers > 0:
      for i in range(0,nLayers):
        x = add_Dense_Layer(nNeurons[i], x)

      # Add dropout before the output layer
      #if percentageDropout > 0.0:
        #x = Dropout(percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
                    kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

    # Show the neural net
    print(model.summary())

    # Compiling the neural network
    model.compile(
        optimizer=adam(lr=lr), 
        loss=loss_function, 
        metrics =metrics)

    # Training the model
    history = model.fit(
        x=x_train,
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=batch_size, 
        epochs=epochs, 
        shuffle=shuffle,
        callbacks=callbacks,
        verbose=1)
    
    plt.figure(figsize=(10,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
      
    # Error de entrenamiento
    plt.plot(history.epoch,np.array(history.history['loss']),label='Loss (train)')
    # Error de validación
    plt.plot(history.epoch,np.array(history.history['val_loss']),label='Loss (val)')

    plt.legend()

    print('|Precisión en Entrenamiento|')
    print("Máximo: ", max(np.array(history.history['categorical_accuracy'])))
    print("Mínimo: ", min(np.array(history.history['categorical_accuracy'])))
    print("Media: ", np.mean(np.array(history.history['categorical_accuracy'])))
    print("Desv. tipica: ", np.std(np.array(history.history['categorical_accuracy'])))

    print("")

    print('|Precisión en Validación|')
    print("Máximo:", max(np.array(history.history['val_categorical_accuracy'])))
    print("Mínimo:", min(np.array(history.history['val_categorical_accuracy'])))
    print("Media:", np.mean(np.array(history.history['val_categorical_accuracy'])))
    print("Desv. tipica:", np.std(np.array(history.history['val_categorical_accuracy'])))

    # Clean the folder where the models are saved
    best_model_name = cleanExperimentFolder(path_experiment)

    # Save figure
    plt.savefig(os.path.join(path_experiment, nameModel + ".png"))

def main():

	# Modify percentageGPU for the experiment
	args = defineArgParsers()
	configureKerasForGPU(args.percentageGPU)

	tables_folder = 'tables'
	nameExperimentsFolder = 'experiments'
	radar_folder = 'radar'
	indexes_sentinel1_v2 = []

	# *** Modifiable ***
	nameExperiment = 'rice'

	sentinels = ["A"] # A, B or AB
	orbits = ["DESC", "ASC"] # ASC, DESC or ASC_DESC.
	indexes_sentinel1 = ['VH_Sum_VV'] # Rice VH_Sum_VV
	indexes_sentinel2 = ['ICEDEX','B11']
	# *** Modifiable ***

	# Update indexes_sentinel1 in other var
	for i in indexes_sentinel1:
	  for o in orbits:
	    for s in sentinels:
	      indexes_sentinel1_v2.append(i+"_"+s+"_"+o)

	# *** Modifiable ***
	# The indexes we will use in order to train the model (indexes = indexes_sentinel1_v2 + indexes_sentinel2)
	indexes = indexes_sentinel1_v2
	# *** Modifiable ***

	interpolate = False # IMPORTANT!!!! Interpolate samples if we're going to use sentinel 1 AND 2, OR We are going to use sentinel-1 A AND B Separately

	# Count the number of ocurrences of the indexes_sentinel1 (more than 2 means interpolate)
	for index_sentinel1 in indexes_sentinel1:
	  count = sum(index_sentinel1 in s for s in indexes)
	  if count > 1:
	    interpolate = True
	    break

	# If the 'for' before doesn't detect that we have to interpolate, we search if we're going to use sentinel2. If there is at least one index, interpolate = True
	if not interpolate:
	  for index_sentinel2 in indexes_sentinel2:
	    count = sum(index_sentinel2 in s for s in indexes)
	    if count > 0:
	      interpolate = True
	      break

	# *** Modifiable *** 
	labels_header = "water"

	labels = ['cumple', 'no_cumple']
	colors_label = ["cyan", "orange"]

	labels_correct = ['correcto', 'error']
	colors_correct = ["green", "red"]
	# *** Modifiable *** 

	experimentFolder = nameExperiment + "_" + ",".join(map(str,sentinels))  + "_" + ",".join(map(str,orbits)) + "-cF_" + str(args.campaingsFull)

	path_radar = os.path.join(tables_folder,radar_folder)

	# *** Modifiable *** 

	# Change this line in order to use other campaings (make sure the only difference is the date)
	campaings = ["rice_A,B_DESC,ASC_2017-11-01_2018-02-01", "rice_A,B_DESC,ASC_2018-11-01_2019-02-01", "rice_A,B_DESC,ASC_2016-11-01_2017-02-01"]

	# *** Modifiable ***

	if args.network in ["LSTM_p_CNN", "LSTM+CNN", "CNN+LSTM", "LSTM"]:

		# Create experiments folder
		if not os.path.exists(nameExperimentsFolder):
			os.mkdir(nameExperimentsFolder)

		# Create experiments folder
		if not os.path.exists(nameExperiment):
			os.mkdir(nameExperiment)

		# Preparing data
		tags_name = "tags_subarroz (2_classes).csv"

		if args.campaingsFull:
			splitTrainTestCampaings(test_size=0.3,campaings=campaings,path_radar=path_radar,tags_name=tags_name,labels_header=labels_header)
			x_train, y_train, x_test, y_test, time_step, num_features, num_classes = loadSamplesFull(tags_name,labels,indexes,campaings,path_radar,labels_header,interpolate)

		else:
			x_train, y_train, x_test, y_test, time_step, num_features, num_classes = loadSamples(tags_name,labels,indexes,campaings,path_radar,labels_header,interpolate)

		x_train, y_train, x_test, y_test = normalize_data(x_train, y_train, x_test, y_test, nameExperimentsFolder, nameExperiment, experimentFolder)

		# Convert string into int array
		nNeuronsSequence = [int(i) for i in args.nNeuronsSequence.split(",")]
		nNeuronsConv1D = [int(i) for i in args.nNeuronsConv1D.split(",")]
		nNeurons = [int(i) for i in args.nNeurons.split(",")]

		if args.network == "LSTM_p_CNN":
			TrainLSTM_p_CNN(lr=args.learning_rate,batch_size=args.batch_size,epochs=args.epochs,percentageDropout=args.percentageDropout,nNeuronsSequence=nNeuronsSequence,
				nNeuronsConv1D=nNeuronsConv1D,nNeurons=nNeurons, patience_stop = args.patience, patience_reduce_lr=args.patience_reduce_lr, loss_function = args.loss_function, shuffle=args.shuffle, 
				min_delta=args.min_delta, x_train = x_train, y_train=y_train, x_test=x_test, y_test=y_test, time_step=time_step, num_features = num_features, num_classes = num_classes, 
				nameExperimentsFolder=nameExperimentsFolder, nameExperiment=nameExperiment, experimentFolder=experimentFolder,campaingsFull=args.campaingsFull)

		elif args.network == "LSTM+CNN":
			TrainLSTM_CNN(lr=args.learning_rate,batch_size=args.batch_size,epochs=args.epochs,percentageDropout=args.percentageDropout, nNeuronsSequence=nNeuronsSequence,nNeuronsConv1D=nNeuronsConv1D,
				nNeurons=nNeurons, patience_stop = args.patience, patience_reduce_lr=args.patience_reduce_lr, loss_function = args.loss_function, shuffle=args.shuffle, min_delta=args.min_delta, 
				x_train = x_train, y_train=y_train, x_test=x_test, y_test=y_test, time_step=time_step, num_features = num_features, num_classes = num_classes, nameExperimentsFolder=nameExperimentsFolder, 
				nameExperiment=nameExperiment, experimentFolder=experimentFolder,campaingsFull=args.campaingsFull)

		elif args.network == "CNN+LSTM":
			TrainCNN_LSTM(lr=args.learning_rate,batch_size=args.batch_size,epochs=args.epochs,percentageDropout=args.percentageDropout, nNeuronsSequence=nNeuronsSequence,nNeuronsConv1D=nNeuronsConv1D,
				nNeurons=nNeurons, patience_stop = args.patience, patience_reduce_lr=args.patience_reduce_lr, loss_function = args.loss_function, shuffle=args.shuffle, min_delta=args.min_delta, 
				x_train = x_train, y_train=y_train, x_test=x_test, y_test=y_test, time_step=time_step, num_features = num_features, num_classes = num_classes,nameExperimentsFolder=nameExperimentsFolder, 
				nameExperiment=nameExperiment, experimentFolder=experimentFolder,campaingsFull=args.campaingsFull)

		elif args.network == "LSTM":
			TrainLSTM(lr=args.learning_rate,batch_size=args.batch_size,epochs=args.epochs,percentageDropout=args.percentageDropout, nNeuronsSequence=nNeuronsSequence,nNeurons=nNeurons, 
				patience_stop = args.patience, patience_reduce_lr=args.patience_reduce_lr, loss_function = args.loss_function, shuffle=args.shuffle, min_delta=args.min_delta, x_train = x_train, 
				y_train=y_train, x_test=x_test, y_test=y_test, time_step=time_step, num_features = num_features, num_classes = num_classes, nameExperimentsFolder=nameExperimentsFolder, 
				nameExperiment=nameExperiment, experimentFolder=experimentFolder,campaingsFull=args.campaingsFull)
	else:
		print("Error. That model is not defined.")

if __name__ == '__main__':
	main()