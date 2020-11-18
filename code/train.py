import pandas as pd
import time, os, csv, re, sys, argparse, shutil
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
rn.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
import keras.backend as k

import keras
from keras.optimizers import adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, GlobalAveragePooling1D, TimeDistributed
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

	# MANDATORY
	required = parser.add_argument_group('required arguments')
	required.add_argument("--nameExperiment",type=str, default='', help="Experiment name (activity,rice). Type -> string. Example -> --nameExperiment='activity'")
	required.add_argument("--sentinels",type=str, default='', help="Sentinel to be used (A, B or AB). Separator -> ','. Type -> string. Example -> --sentinels='A,B'")
	required.add_argument("--orbits",type=str, default='', help="Orbit to be used (ASC, DESC or ASC_DESC). Separator -> ','. Type -> string. Example -> --orbits='DESC,ASC'")
	required.add_argument("--indexes_sentinel1",type=str, default='', help="indexes of radar to be used (Rice: VH_Sum_VV). Separator -> ','. Type -> string. Example -> --indexes_sentinel1='VH_Sum_VV,VV'")
	required.add_argument("--labels",type=str, default='', help="Labels name for each class ('cumple','no_cumple'). Separator -> ','. Type -> string. Example -> --labels='cumple,no_cumple")
	required.add_argument("--colors_label",type=str, default='', help="Color for each class name ('cyan','orange'. Separator -> ','). Type -> string. Example -> --colors_label='cyan,orange'")
	required.add_argument("--campaings",type=str, default='', help="What campaings we will use (the order is important -> train and last val. Separator -> '|'). Type -> string. Example -> --campaings='rice_A,B_DESC,ASC_2016-11-15_2017-01-15|rice_A,B_DESC,ASC_2017-11-15_2018-01-15'")
	required.add_argument("--tags_name",type=str, default='', help="Tag filename of the regions (Rice -> tags_subarroz (2_classes).csv). Type -> string. Example -> --tags_name='tags.csv'")

	# OPTIONAL
	parser.add_argument("--network",type=str, default='LSTM_p_CNN', help="Select the network you want to use (LSTM_p_CNN, GRU_p_CNN, LSTM+CNN, GRU+CNN, CNN+LSTM, CNN+GRU, LSTM, GRU, CNN). Type -> string. Example -> --network='LSTM+CNN'")
	parser.add_argument("--percentageGPU",type=float, default=0.0, help="Amount of use the memory of the GPU. Type -> float. Example -> --percentageGPU=0.30")
	parser.add_argument("--learning_rate",type=float, default=1e-4, help="Learning rate modifier. Type -> float. Example -> --learning_rate=1e-03")
	parser.add_argument("--batch_size",type=int, default=16, help="Size of batch (number of samples) to evaluate. Type -> int. Example -> --batch_size=16")
	parser.add_argument("--epochs",type=int, default=100, help="Number of epochs. Type -> int. Example -> --epochs=100")
	parser.add_argument("--percentageDropout",type=float, default=0.2, help="How many links of the network will be ommited in order to avoid overfitting.  Type -> float. Example -> --percentageDropout=0.2")
	parser.add_argument("--patience",type=int, default=30, help="Number of epochs with no improvement after which training will be stopped. Type -> int. Example -> --patience=30")
	parser.add_argument("--patience_reduce_lr",type=int, default=8, help="Num epochs to reduce learning rate. Type -> int. Example -> --patience_reduce_lr=8")
	parser.add_argument("--nNeuronsSequence",type=str, default="128", help="Number of units in the LSTM layer and number of LSTM layers. Type -> string. Example -> --nNeuronsSequence='128,64'")
	parser.add_argument("--nNeuronsConv1D",type=str, default="64,64", help="Number of kernels in the Convolutional layer and number of Convolutionals layers. Type -> string. Example -> --nNeuronsConv1D='64,64'")
	parser.add_argument("--nNeurons",type=str, default="64", help="Number of neurons at the end of the network (hidden layers). Type -> string. Example -> --nNeurons='64,32'")
	parser.add_argument("--loss_function",type=str, default="categorical_crossentropy", help="loss function (categorical_crossentropy). Type -> string. Example -> --loss_function='categorical_crossentropy'")
	parser.add_argument("--shuffle",type=str2bool, default="y", help="Whether to shuffle the order of the batches at the beginning of each epoch. Type -> string. Example -> --shuffle='y'")
	parser.add_argument("--min_delta",type=float, default=1e-3, help="Minimum change in the monitored quantity to qualify as an improvement. Type -> float. Example -> --min_delta=1e-03")
	parser.add_argument("--campaingsFull",type=str2bool, default="n", help="using all the campaings to train (split train/val each campaing) or using a few for train, and one for val. Type -> string. Example -> --campaingsFull='n'")
	parser.add_argument("--indexes_sentinel2",type=str, default='', help="indexes of optic to be used (Rice: ICEDEX, B11). Separator -> ','. Type -> string. Example -> --indexes_sentinel2='NDVI,B11'")
	parser.add_argument("--kernelSize",type=int, default=3, help="kernel's size for convolutional 1D. Type -> int. Example -> --kernelSize=3")

	return parser.parse_args()
   
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
  models = models[2:]

  # We check if there is, at least, one min model saved
  if len(models) > 1:
    for i in range (1,len(models)):
      model_path = os.path.join(folderNameExperimentPath,models[i])
      os.remove(model_path)
      print("Experiment %s removed" %(model_path))
  else:
    print("Folder %s ignored to clean" %(folderNameExperimentPath))

  # Get best model name
  return models[0]

def write_model_structure(model, model_path):

	# Write model structure in a file
	stringlist = []
	model.summary(print_fn=lambda x: stringlist.append(x))
	short_model_summary = "\n".join(stringlist)
	out = open(os.path.join(model_path,"structure.txt"),'w')
	out.write(short_model_summary)
	out.close
 
def plot_history(history):

  plt.figure(figsize=(8, 5))
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.yscale('log')
  plt.plot(history.epoch,np.array(history.history['loss']),label='categorical_crossentropy (training)')
  plt.plot(history.epoch,np.array(history.history['val_loss']),label='categorical_crossentropy (validation)')
  plt.legend()
  plt.grid()

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

def add_CuDNNGRU_Layer(number,return_sequence,data):

  data = CuDNNGRU(number, return_sequences=return_sequence,
              kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
              recurrent_initializer=keras.initializers.glorot_uniform(seed=seed),
              bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data) 
  return data  

def add_Conv1D_Layer(number, kernelSize, data):

  data = Conv1D(filters=number, kernel_size=kernelSize,data_format='channels_last', activation='relu', padding='same', 
  	kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
  	bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data)
  
  return data

def add_Conv1DTimeDistributed_Layer(number, kernelSize, data):

  data = TimeDistributed(Conv1D(filters=number, kernel_size=kernelSize,data_format='channels_last', activation='relu', padding='same',
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

	  # Read the train and val.csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,tags_name))
	  labels_header = list(tagDataFrameTrain.columns[1:])
	  for label_header in labels_header:
	  	tagDataFrameTrain = tagDataFrameTrain[tagDataFrameTrain[label_header] != -1]

	  num_trainSamples = num_trainSamples + tagDataFrameTrain[labels_header].value_counts()

	  print(campaing)

	# Read the train and val.csv
	tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaings[-1],tags_name))
	for label_header in labels_header:
		tagDataFrameTest = tagDataFrameTest[tagDataFrameTest[label_header] != -1]
	nameColumns = tagDataFrameTest.columns

	num_testSamples = num_testSamples + tagDataFrameTest[labels_header].value_counts()

	print(campaings[-1])

	print("Entrenamiento")
	print(num_trainSamples)
	total_train = num_trainSamples.sum()
	print("Total %d" %(total_train))

	print("")
	print("Validación")
	print(num_testSamples)
	total_test = num_testSamples.sum()
	print("Total %d" %(total_test))

def splitTrainTestCampaings(test_size=0.3,*,campaings,path_radar,tags_name):

	for campaing in campaings:

		path_train = os.path.join(path_radar,campaing,"train.csv")
		path_test = os.path.join(path_radar,campaing,"val.csv")

		# Read 'tags.csv' and clean tag dataframe. Ignore the samples that were marked as -1
		tagDataFrame = pd.read_csv(os.path.join(path_radar,campaing,tags_name))

		name_region = tagDataFrame.columns[0]
		labels_header = list(tagDataFrame.columns[1:])

		if os.path.exists(path_train) and os.path.exists(path_test):

			print("Campaña: %s. Split ya realizado." %(campaing))
			continue

		else:

			for label_header in labels_header:
				tagDataFrame = tagDataFrame[tagDataFrame[label_header] != -1]

			tagDataFrameName = tagDataFrame[name_region]
			tagDataFrameActivity = tagDataFrame[labels_header]

			# Get the train and val split. It will shuffle automatically. 70% train, 30% val  
			dfName_train, dfName_test, dfTarget_train, dfTarget_test = train_test_split(tagDataFrameName,tagDataFrameActivity, test_size=0.3, random_state=seed, stratify=tagDataFrameActivity)

			df_train = pd.concat([dfName_train, dfTarget_train], axis=1)
			pd.DataFrame.to_csv(df_train,index=False,path_or_buf=os.path.join(path_radar,campaing,"train.csv"))
			print("Conjunto de train guardado correctamente en %s" %(os.path.join(path_radar,campaing,"train.csv")))

			df_test = pd.concat([dfName_test, dfTarget_test], axis=1)
			pd.DataFrame.to_csv(df_test,index=False,path_or_buf=os.path.join(path_radar,campaing,"val.csv"))
			print("Conjunto de val guardado correctamente en %s" %(os.path.join(path_radar,campaing,"val.csv")))

	return labels_header

def loadSamplesFull(labels, indexes, campaings, path_radar,interpolate):

	now = time.time()

	num_classes = len(labels)
	num_features = len(indexes)
	sequencesTrain = []
	sequencesTest = []
	targetsTrain = []
	targetsTest = []

	for campaing in campaings:

	  # Read the train and val.csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,"train.csv"))
	  print("'train.csv' cargado correctamente")

	  tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaing,"val.csv"))
	  print("'val.csv' cargado correctamente")

	  total_train = tagDataFrameTrain.shape[0]
	  total_test = tagDataFrameTest.shape[0]

	  # Get the sequence for each area
	  i = 1
	  for row in tagDataFrameTrain.values:

	    region_path = row[0]
	    areadf = pd.read_csv(os.path.join(path_radar,campaing,'dataset',region_path))
	    areadf = areadf[indexes].dropna(how='all')
	    areadf = areadf.drop_duplicates()

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
	    areadf = areadf.drop_duplicates()

	    if(interpolate):
	      areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()

	    sequencesTest.append(areadf.values)
	    targetsTest.append(row[1:])

	    print("---Validación---")
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

	  print("Generating fix time_step for val")
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
	print("Validación")
	print(unique_samples_test)
	print(num_testSamples)
	print("Total: %d" %(num_testSamples.sum()))

	return x_train, y_train, x_test, y_test, time_step, num_features, num_classes

def loadSamples(tags_name, labels, indexes, campaings, path_radar,interpolate):

	now = time.time()

	num_classes = len(labels)
	num_features = len(indexes)
	sequencesTrain = []
	sequencesTest = []
	targetsTrain = []
	targetsTest = []

	# --- TRAIN BLOCK ---
	for campaing in campaings[:-1]:

	  # Read the train and val.csv
	  tagDataFrameTrain = pd.read_csv(os.path.join(path_radar,campaing,tags_name))
	  labels_header = list(tagDataFrameTrain.columns[1:])
	  for label_header in labels_header:
	  	tagDataFrameTrain = tagDataFrameTrain[tagDataFrameTrain[label_header] != -1]
	  print("'tags.csv' de entrenamiento cargado correctamente")

	  total_train = tagDataFrameTrain.shape[0]

	  # Get the sequence for each area
	  i = 1
	  for row in tagDataFrameTrain.values:

	    region_path = row[0]
	    areadf = pd.read_csv(os.path.join(path_radar,campaing,'dataset',region_path))
	    areadf = areadf[indexes].dropna(how='all')
	    areadf = areadf.drop_duplicates()

	    if(interpolate):
	      areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()
	    
	    sequencesTrain.append(areadf.values)
	    targetsTrain.append(row[1:])

	    print("---Entrenamiento---")
	    print(campaing)
	    print("Progreso %d/%d" %(i,total_train))
	    print("Recinto %s cargado." %(region_path))
	    i=i+1

	# --- VAL BLOCK ---
	# Read the val .csv
	tagDataFrameTest = pd.read_csv(os.path.join(path_radar,campaings[-1],tags_name))
	labels_header = list(tagDataFrameTest.columns[1:])
	for label_header in labels_header:
		tagDataFrameTest = tagDataFrameTest[tagDataFrameTest[label_header] != -1]
	print("'tags.csv' de val cargado correctamente")

	total_test = tagDataFrameTest.shape[0]

	# Get the sequence for each area (VAL)
	i = 1
	for row in tagDataFrameTest.values:

	  region_path = row[0]
	  areadf = pd.read_csv(os.path.join(path_radar,campaings[-1],'dataset',region_path))
	  areadf = areadf[indexes].dropna(how='all')
	  areadf = areadf.drop_duplicates()

	  if(interpolate):
	    areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()
	  
	  sequencesTest.append(areadf.values)
	  targetsTest.append(row[1:])

	  print("---Validación---")
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

	  print("Generating fix time_step for val")
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
	print("Validación")
	print(unique_samples_test)
	print(num_testSamples)
	print("Total: %d" %(num_testSamples.sum()))

	return x_train, y_train, x_test, y_test, time_step, num_features, num_classes, labels_header

def normalize_data(x_train, y_train, x_test, y_test, path_folderScalers, nameExperiment, experimentFolder):

	path_scaler = os.path.join(path_folderScalers, experimentFolder + '-scaler.pkl')

	# Normalize data
	samples_train, steps, features = x_train.shape
	samples_test = x_test.shape[0]

	x_train = x_train.reshape((samples_train,steps*features),order='F')
	x_test = x_test.reshape((samples_test,steps*features),order='F')	

	if not os.path.exists(path_scaler):

		# Change for  MinMaxScaler, StandardScaler, Normalizer
		scaler = preprocessing.MinMaxScaler().fit(x_train)

		# Save scaler
		dump(scaler, open(path_scaler, 'wb'))
		print("Scaler saved correctly")

	else:

		print("Scaler already saved")

		# Load scaler
		scaler = load(open(path_scaler, 'rb'))

	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	x_train = np.reshape(x_train, (samples_train, steps, features),order='F')
	x_test = np.reshape(x_test, (samples_test, steps, features),order='F')

	# Shuffle the x_train samples
	x_train, y_train = shuffle(x_train, y_train, random_state=seed)

	print("Samples normalized.")

	return x_train, y_train, x_test, y_test

def writeOptions(path_optionsFile, nameExperiment, indexes, interpolate, labels_header, labels, colors_label, campaingsFull, tags_name, time_step, campaings):

	with open(path_optionsFile,mode="w",newline='') as output_file:
		output_writer = csv.writer(output_file, delimiter=',')
		output_writer.writerow(['nameExperiment', 'indexes', 'interpolate', 'labels_header', 'labels', 'colors_label', 'campaingFull', 'tags_name', 'time_step', 'campaings'])
		output_writer.writerow([nameExperiment, indexes, interpolate, labels_header, labels, colors_label, campaingsFull, tags_name, time_step, campaings])

def writeAccuracyResults(network,nameExperiment,path_experiment,loss,losses, accuracies, val_loss,val_losses, val_accuracies, num_modules_output):

	path_results = os.path.join("experiments",nameExperiment,"results")

	# Check if the folder 'results' exists. If not, we'll create it
	if not os.path.exists(path_results):
		os.mkdir(path_results)

	fileOutputName = os.path.join(path_results,network+"_loss.csv")
	header_file = ["Name","Loss", "Val_Loss"]
	accuracies_round = []
	val_accuracies_round = []
	for i in range(0, num_modules_output):
		header_file.append("Loss_" + str(i))
	for i in range(0, num_modules_output):
		header_file.append("Accuracy_" + str(i))
	for i in range(0, num_modules_output):
		header_file.append("Val_Loss_" + str(i))
	for i in range(0, num_modules_output):
		header_file.append("Val_Accuracy_" + str(i))
	for i in range(0, num_modules_output):
		accuracies_round.append(str(round(accuracies[i]*100,2)) + ' %')
	for i in range(0, num_modules_output):
		val_accuracies_round.append(str(round(val_accuracies[i]*100,2)) + ' %')

	# Check if the file 'network_loss.csv' exists
	if os.path.exists(fileOutputName):
		fileOutputNameAux = os.path.join(path_results,network+"_loss-temp.csv")

		# Update original file using temporal
		if os.path.exists(fileOutputNameAux):
			print("Replacing 'temp' to 'original")
			# Remove the original file
			os.remove(fileOutputName)
			# Rename the temporal file
			os.rename(fileOutputNameAux,fileOutputName)

		# Open original dataframe
		dataframe_o = pd.read_csv(fileOutputName)

		# Write the results in a temp file
		with open(fileOutputNameAux,mode='w',newline='') as output_file:
			output_writer = csv.writer(output_file, delimiter=',')
			output_writer.writerow(header_file)
			output_writer.writerow([path_experiment, loss, val_loss] + losses + accuracies_round + val_losses + val_accuracies_round)

		# Open temp dataframe
		dataframe_temp = pd.read_csv(fileOutputNameAux)

		dataframe_o = pd.concat([dataframe_o,dataframe_temp])

		# Remove the original and temp file
		os.remove(fileOutputName)
		os.remove(fileOutputNameAux)

		# Save the new file
		dataframe_o.to_csv(fileOutputName,index=False)
									
	else:

		with open(fileOutputName,mode='w',newline='') as output_file:

			output_writer = csv.writer(output_file,delimiter=',')
			output_writer.writerow(header_file)
			output_writer.writerow([path_experiment, loss, val_loss] + losses + accuracies_round + val_losses + val_accuracies_round)	

# TRAIN MODELS FUNCTIONS
# LSTM || CNN
def defineLSTM_p_CNN(input, nLayersSequence, nNeuronsSequence, nLayersConv1D, nNeuronsConv1D, kernelSize, percentageDropout, nLayers, nNeurons):

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

	  #if percentageDropout > 0.0:
	    #x = Dropout(percentageDropout)(x)            

	else:
	  print("Please, insert at least one recurrent layer.")
	  assert False


	#--------------
	# CONV1D block
	#--------------
	# Check if the user has entered at least one hidden layer conv1D
	if nLayersConv1D > 0:
	    x_2 = add_Conv1D_Layer(nNeuronsConv1D[0], kernelSize, input)
	    #x_2 = BatchNormalization()(x_2)

	    for i in range(1,nLayersConv1D):
	      x_2 = add_Conv1D_Layer(nNeuronsConv1D[i], kernelSize, x_2)
	      #x_2 = BatchNormalization()(x_2)

	      if i % 2 == 1:          	
	      	#if percentageDropout > 0.0:
	      		#x_2 = Dropout(percentageDropout)(x_2)
	      	x_2 = MaxPooling1D()(x_2)	      		      

	    #x_2 = GlobalAveragePooling1D()(x_2)
	    x_2 = Flatten()(x_2)

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
	  if percentageDropout > 0.0:
	    x = Dropout(percentageDropout)(x)

	return x

# GRU || CNN
def defineGRU_p_CNN(input, nLayersSequence, nNeuronsSequence, nLayersConv1D, nNeuronsConv1D, kernelSize, percentageDropout, nLayers, nNeurons):

	#--------------
	# GRU block
	#--------------
	# Check if the user has entered at least one hidden layer sequence
	if nLayersSequence > 0:
	  # Has two hidden LSTM layers
	  if nLayersSequence > 1:
	    x = add_CuDNNGRU_Layer(nNeuronsSequence[0], True, input)

	    for i in range(1,nLayersSequence-1):
	      x = add_CuDNNGRU_Layer(nNeuronsSequence[i], True, x)

	    x = add_CuDNNGRU_Layer(nNeuronsSequence[-1], False, x)

	  # Only one hidden LSTM layer
	  else:
	    x = add_CuDNNGRU_Layer(nNeuronsSequence[0], False, input)

	  #if percentageDropout > 0.0:
	    #x = Dropout(percentageDropout)(x)            

	else:
	  print("Please, insert at least one recurrent layer.")
	  assert False


	#--------------
	# CONV1D block
	#--------------
	# Check if the user has entered at least one hidden layer conv1D
	if nLayersConv1D > 0:
	    x_2 = add_Conv1D_Layer(nNeuronsConv1D[0], kernelSize, input)
	    #x_2 = BatchNormalization()(x_2)

	    for i in range(1,nLayersConv1D):
	      x_2 = add_Conv1D_Layer(nNeuronsConv1D[i], kernelSize, x_2)
	      #x_2 = BatchNormalization()(x_2)

	      if i % 2 == 1:          	
	      	#if percentageDropout > 0.0:
	      		#x_2 = Dropout(percentageDropout)(x_2)
	      	x_2 = MaxPooling1D()(x_2)	      		      

	    #x_2 = GlobalAveragePooling1D()(x_2)
	    x_2 = Flatten()(x_2)

	else:
	  print("Please, insert at least one conv1D layer.")
	  assert False

	#--------------
	# CONCATENATE GRU + Conv1D
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
	  if percentageDropout > 0.0:
	    x = Dropout(percentageDropout)(x)

	return x

# LSTM + CNN
def defineLSTM_CNN(input, nLayersSequence, nNeuronsSequence, nLayersConv1D, nNeuronsConv1D, kernelSize, percentageDropout, nLayers, nNeurons):

    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_CuDNNLSTM_Layer(nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_CuDNNLSTM_Layer(nNeuronsSequence[i], True, x)

      #if percentageDropout > 0.0:
            #x = Dropout(percentageDropout)(x)  
    else:
      print("Please, insert at least one recurrent layer.")
      assert False

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(nNeuronsConv1D[0], kernelSize, x)
        #x = BatchNormalization()(x)

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(nNeuronsConv1D[i], kernelSize, x)
          #x = BatchNormalization()(x)

          if i % 2 == 1:          	
          	#if percentageDropout > 0.0:
          		#x = Dropout(percentageDropout)(x)
          	x = MaxPooling1D()(x)          		

        # Apply global average pooling and make the output only one dimension
        #x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)   

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
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)

    return x

# GRU + CNN
def defineGRU_CNN(input, nLayersSequence, nNeuronsSequence, nLayersConv1D, nNeuronsConv1D, kernelSize, percentageDropout, nLayers, nNeurons):

    #--------------
    # GRU block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_CuDNNGRU_Layer(nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_CuDNNGRU_Layer(nNeuronsSequence[i], True, x)

      #if percentageDropout > 0.0:
            #x = Dropout(percentageDropout)(x)  
    else:
      print("Please, insert at least one recurrent layer.")
      assert False

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(nNeuronsConv1D[0], kernelSize, x)
        #x = BatchNormalization()(x)

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(nNeuronsConv1D[i], kernelSize, x)
          #x = BatchNormalization()(x)

          if i % 2 == 1:          	
          	#if percentageDropout > 0.0:
          		#x = Dropout(percentageDropout)(x)
          	x = MaxPooling1D()(x)          		

        # Apply global average pooling and make the output only one dimension
        #x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)   

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
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)

    return x

# CNN + LSTM
def defineCNN_LSTM(input, nLayersConv1D, nNeuronsConv1D, kernelSize, nLayersSequence, nNeuronsSequence, percentageDropout, nLayers, nNeurons):

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[0], kernelSize, input)

        for i in range(1,nLayersConv1D):
          x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[i], kernelSize, x)

          # Add a dropout and a Pooling each 2 conv1D layer
          if i % 2 == 1:
            #if percentageDropout > 0.0:
              #x = TimeDistributed(Dropout(percentageDropout))(x)
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
      #if percentageDropout > 0.0:
        #x = Dropout(percentageDropout)(x)
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
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)

    return x	

# CNN + GRU
def defineCNN_GRU(input, nLayersConv1D, nNeuronsConv1D, kernelSize, nLayersSequence, nNeuronsSequence, percentageDropout, nLayers, nNeurons):

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[0], kernelSize, input)

        for i in range(1,nLayersConv1D):
          x = add_Conv1DTimeDistributed_Layer(nNeuronsConv1D[i], kernelSize, x)

          # Add a dropout and a Pooling each 2 conv1D layer
          if i % 2 == 1:
            #if percentageDropout > 0.0:
              #x = TimeDistributed(Dropout(percentageDropout))(x)
            x = TimeDistributed(MaxPooling1D())(x)              

        # Apply flatten
        x = TimeDistributed(Flatten())(x)

    else:
      print("Please, insert at least one conv1D layer.")
      assert False
 
    #--------------
    # GRU block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      # The user has entered two hidden layers
      if nLayersSequence > 1:
        x = add_CuDNNGRU_Layer(nNeuronsSequence[0], True, x)

        for i in range(1,nLayersSequence-1):
          x = add_CuDNNGRU_Layer(nNeuronsSequence[i], True, x)

        x = add_CuDNNGRU_Layer(nNeuronsSequence[-1], False, x)
  
      # The user has entered only one hidden layer
      else:
        x = add_CuDNNGRU_Layer(nNeuronsSequence[0], False, x)

      # Add dropout layer 
      #if percentageDropout > 0.0:
        #x = Dropout(percentageDropout)(x)
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
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)

    return x	

# LSTM
def defineLSTM(input, nLayersSequence, nNeuronsSequence, percentageDropout, nLayers, nNeurons):

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
	      
	  #if percentageDropout > 0.0:
	    #x = Dropout(percentageDropout)(x)
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
	  if percentageDropout > 0.0:
	    x = Dropout(percentageDropout)(x)

	return x

# GRU
def defineGRU(input, nLayersSequence, nNeuronsSequence, percentageDropout, nLayers, nNeurons):

	#--------------
	# GRU block
	#--------------
	# ADD Recurrent layer
	# Check if the user has entered at least one hidden layer sequence
	if nLayersSequence > 0:
	  # The user has entered two hidden layers
	  if nLayersSequence > 1:
	    x = add_CuDNNGRU_Layer(nNeuronsSequence[0], True, input)

	    for i in range(1,nLayersSequence-1):
	      x = add_CuDNNGRU_Layer(nNeuronsSequence[i], True, x)

	    x = add_CuDNNGRU_Layer(nNeuronsSequence[-1], False, x)

	  # The user has entered only one hidden layer
	  else:
	    x = add_CuDNNGRU_Layer(nNeuronsSequence[0], False, input)
	      
	  #if percentageDropout > 0.0:
	    #x = Dropout(percentageDropout)(x)
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
	  if percentageDropout > 0.0:
	    x = Dropout(percentageDropout)(x)

	return x

# CNN
def defineCNN(input, nLayersConv1D, nNeuronsConv1D, kernelSize, percentageDropout, nLayers, nNeurons):

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(nNeuronsConv1D[0], kernelSize, input)

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(nNeuronsConv1D[i], kernelSize, x)

          # Add a dropout and a Pooling each 2 conv1D layer
          if i % 2 == 1:
            #if percentageDropout > 0.0:
              #x = Dropout(percentageDropout)(x)
            x = MaxPooling1D()(x)              

        # Apply flatten
        x = Flatten()(x)

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
      if percentageDropout > 0.0:
        x = Dropout(percentageDropout)(x)

    return x

# Train algorithm
def Train(lr=1e-03, batch_size=16, epochs=100, percentageDropout=0.0, nNeuronsConv1D=[128,256,128], nNeurons=[16,8], shuffle=False, min_delta= 1e-03, patience_stop = 30, 
	patience_reduce_lr = 8, loss_function = 'categorical_crossentropy',  metrics = ['categorical_accuracy'], *, x_train, y_train, x_test, y_test, num_classes, network, 
	nameExperimentsFolder, nameExperiment, experimentFolder, input, x, nameModel, num_modules_output):

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
	nLayersConv1D = len(nNeuronsConv1D)
	loss = 0.0
	accuracy = 0.0
	val_loss = 0.0
	val_accuracy = 0.0  

	# date
	date = dateTime.now().strftime("%d:%m:%y:%H:%M:%S")

	fileExtension = '{epoch:02d}-{val_loss:.4f}.hdf5'
	path_experiment = os.path.join(nameExperimentsFolder,nameExperiment,'models',experimentFolder,nameModel)

	experimentFolder = False
	experimentHasImage = False

	if os.path.exists(path_experiment):
		experimentFolder = True
		for i in os.listdir(path_experiment):
			if i.split(".")[-1] == "png":
				experimentHasImage=True
				break	

	# If the experiment folder already exists, we will ignore it.
	if experimentHasImage:
		print('Ignored the experiment %s. This experiment has been used before.' % (path_experiment))

	# The experiment folder doesn't exists
	else:
		
		if experimentFolder:
			shutil.rmtree(path_experiment)
		
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

		outputs = []
		loss_dict = {}
		y_dict = {}
		validation_data_dict = {}

		# Using only one module output
		if num_modules_output == 1:
			output = Dense(num_classes, activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=seed), name="output_0")(x)
			outputs.append(output)

			loss_dict["output_0"] = loss_function	
			y_dict["output_0"] = y_train
			validation_data_dict["output_0"] = y_test
		# Multiple module output
		else:
			for i in range(0,num_modules_output):
				output = Dense(num_classes, activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=seed), name="output_" + str(i))(x)
				outputs.append(output)

				loss_dict["output_" + str(i)] = loss_function	
				y_dict["output_" + str(i)] = y_train[:,i]
				validation_data_dict["output_" + str(i)] = y_test[:,i]		

		model = Model(input,outputs)

		# Show and write the neural net
		print(model.summary())
		write_model_structure(model, path_experiment)

		# Compiling the neural network
		model.compile(optimizer=adam(lr=lr), loss=loss_dict, metrics = metrics)

		# Training the model
		history = model.fit(
		    x=x_train,
		    validation_data=(x_test,validation_data_dict),
		    y=y_dict,
		    batch_size=batch_size, 
		    epochs=epochs, 
		    shuffle=shuffle,
		    callbacks=callbacks,
		    verbose=1,
		    workers=6,
		    use_multiprocessing=True)		

		plt.figure(figsize=(10,5))
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.yscale('log')
		  
		# Error de entrenamiento
		plt.plot(history.epoch,np.array(history.history['loss']),label='Loss (train)')
		# Error de validación
		plt.plot(history.epoch,np.array(history.history['val_loss']),label='Loss (val)')

		plt.legend()
		plt.grid()

		val_loss_index = np.array(history.history['val_loss']).argmin()
		loss = np.array(history.history['loss'])[val_loss_index]
		val_loss = np.array(history.history['val_loss'])[val_loss_index]
		losses = []
		accuracies = []
		val_losses = []
		val_accuracies = []
		# Using only one module output
		if num_modules_output == 1:
			losses.append(loss)
			val_losses.append(val_loss)
			accuracies.append(np.array(history.history['categorical_accuracy'])[val_loss_index])
			val_accuracies.append(np.array(history.history['val_categorical_accuracy'])[val_loss_index])

			print('|Precisión en Entrenamiento|')
			print("Mejor modelo: ", str(round(accuracies[0]*100,2)) + ' %')
			print("Mínimo: ", str(round(min(np.array(history.history['categorical_accuracy']))*100,2)) + ' %')

			print("")

			print('|Precisión en Validación|')
			print("Mejor modelo: ", str(round(val_accuracies[0]*100,2)) + ' %')
			print("Mínimo:", str(round(min(np.array(history.history['val_categorical_accuracy']))*100,2)) + ' %')

			print("")

		# Multiple module output			
		else:
			for i in range(0, num_modules_output):
				losses.append(np.array(history.history['output_'+str(i)+'_loss'])[val_loss_index])
				accuracies.append(np.array(history.history['output_'+str(i)+'_categorical_accuracy'])[val_loss_index])
				val_losses.append(np.array(history.history['val_output_'+str(i)+'_loss'])[val_loss_index])
				val_accuracies.append(np.array(history.history['val_output_'+str(i)+'_categorical_accuracy'])[val_loss_index])

				print('|Precisión en Entrenamiento|')
				print("Mejor modelo: ", str(round(accuracies[i]*100,2)) + ' %')
				print("Mínimo: ", str(round(min(np.array(history.history['output_'+str(i)+'_categorical_accuracy']))*100,2)) + ' %')

				print("")

				print('|Precisión en Validación|')
				print("Mejor modelo: ", str(round(val_accuracies[i]*100,2)) + ' %')
				print("Mínimo:", str(round(min(np.array(history.history['val_output_'+str(i)+'_categorical_accuracy']))*100,2)) + ' %')

				print("")				

		# Clean the folder where the models are saved
		best_model_name = cleanExperimentFolder(path_experiment)

		# Save figure
		plt.savefig(os.path.join(path_experiment, nameModel + ".png"))

		writeAccuracyResults(network,nameExperiment,path_experiment,loss,losses, accuracies, val_loss, val_losses, val_accuracies, num_modules_output)

		# Free memory
		del model
		sess = k.get_session()
		k.clear_session()
		sess.close()		

def main():

	# Modify percentageGPU for the experiment
	args = defineArgParsers()
	configureKerasForGPU(args.percentageGPU)

	tables_folder = 'tables'
	nameExperimentsFolder = 'experiments'
	radar_folder = 'radar'
	indexes_sentinel1_v2 = []

	if args.nameExperiment == '':
		print("Error -> 'nameExperiment' not specified")
		sys.exit()

	if args.sentinels == '':
		print("Error -> 'sentinels' not specified")
		sys.exit()
	sentinels = args.sentinels.split(",")

	if args.orbits == '':
		print("Error -> 'orbits' not specified")
		sys.exit()
	orbits = args.orbits.split(",")

	if args.indexes_sentinel1 == '':
		print("Error -> 'indexes_sentinel1' not specified")
		sys.exit()
	indexes_sentinel1 = args.indexes_sentinel1.split(",")

	if args.indexes_sentinel2 != '':
		indexes_sentinel2 = args.indexes_sentinel2.split(",")
	else:
		indexes_sentinel2 = []

	# Update indexes_sentinel1 in other var
	for i in indexes_sentinel1:
	  for o in orbits:
	    for s in sentinels:
	      indexes_sentinel1_v2.append(i+"_"+s+"_"+o)

	# Interpolate samples if we're going to use sentinel 1 AND 2, OR We are going to use sentinel-1 A AND B OR ASC AND DESC Separately
	interpolate = False

	# Count the number of ocurrences of indexes_sentinel1 (more than 2 means interpolate)
	for index_sentinel1 in indexes_sentinel1:
	  count = sum(index_sentinel1 in s for s in indexes_sentinel1_v2)
	  if count > 1:
	    interpolate = True
	    break

	# If the 'for' before doesn't detect that we have to interpolate, we search if we're going to use sentinel2.
	if not interpolate and len(indexes_sentinel2) > 0:
		interpolate = True

	# Combine the two indexes (sentinel1 and sentinel2)
	indexes = indexes_sentinel1_v2 + indexes_sentinel2

	if args.labels == '':
		print("Error -> 'labels' not specified")
		sys.exit()
	labels = args.labels.split(",")

	if args.colors_label == '':
		print("Error -> 'colors_label' not specified")
		sys.exit()
	colors_label = args.colors_label.split(",")	

	if args.campaings == '':
		print("Error -> 'campaings' not specified")
		sys.exit()
	campaings = args.campaings.split("|")

	# If we're using only one campaing, we have to split it to train and val samples
	if args.campaingsFull == False and len(campaings) == 1:
		args.campaingsFull = True

	experimentFolder = args.nameExperiment + "_" + ",".join(map(str,sentinels))  + "_" + ",".join(map(str,orbits)) + "-cF_" + str(args.campaingsFull)

	path_radar = os.path.join(tables_folder,radar_folder)		

	if args.tags_name == '':
		print("Error -> 'tags_name' not specified")
		sys.exit()

	if args.network in ["LSTM_p_CNN", "GRU_p_CNN", "LSTM+CNN", "GRU+CNN", "CNN+LSTM", "CNN+GRU", "LSTM", "GRU", "CNN"]:

		# --- LOAD DATA ---
		# Create experiments folder
		if not os.path.exists(nameExperimentsFolder):
			os.mkdir(nameExperimentsFolder)

		# Create nameexperiment
		if not os.path.exists(os.path.join(nameExperimentsFolder,args.nameExperiment)):
			os.mkdir(os.path.join(nameExperimentsFolder,args.nameExperiment))

		# Load data
		if args.campaingsFull:
			labels_header = splitTrainTestCampaings(test_size=0.3,campaings=campaings,path_radar=path_radar,tags_name=args.tags_name)
			x_train, y_train, x_test, y_test, time_step, num_features, num_classes = loadSamplesFull(labels,indexes,campaings,path_radar,interpolate)

		else:
			x_train, y_train, x_test, y_test, time_step, num_features, num_classes, labels_header = loadSamples(args.tags_name,labels,indexes,campaings,path_radar,interpolate)

		num_labels_header = len(labels_header)

		# Create 'scaler folder'
		path_folderScalers = os.path.join(nameExperimentsFolder,args.nameExperiment,"scalers")
		if not os.path.exists(path_folderScalers):
			os.mkdir(path_folderScalers)
		x_train, y_train, x_test, y_test = normalize_data(x_train, y_train, x_test, y_test, path_folderScalers, args.nameExperiment, experimentFolder)

		# Write options
		path_folderOptions = os.path.join(nameExperimentsFolder,args.nameExperiment,"options")
		path_optionsFile = os.path.join(path_folderOptions,experimentFolder+".csv")
		if not os.path.exists(path_folderOptions):
			os.mkdir(path_folderOptions)
		# Check if the options file is already saved
		if os.path.exists(path_optionsFile):
			print("Options file already saved")
		else:
			writeOptions(path_optionsFile, args.nameExperiment, indexes, interpolate, labels_header, labels, colors_label, args.campaingsFull, args.tags_name, time_step, campaings)

		# Convert string into int array
		nNeuronsSequence = [int(i) for i in args.nNeuronsSequence.split(",")]
		nNeuronsConv1D = [int(i) for i in args.nNeuronsConv1D.split(",")]

		# If the list is not empty, there are dense neurons in hidden layer
		if args.nNeurons != "":
			nNeurons = [int(i) for i in args.nNeurons.split(",")]
		else:
			nNeurons = []
		

		if args.network == "LSTM_p_CNN":

			# Create the model
			nameModel = 'LSTM_p_CNN__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineLSTM_p_CNN(input, len(nNeuronsSequence), nNeuronsSequence, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, args.percentageDropout, len(nNeurons), 
				nNeurons)

			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)

		elif args.network == "GRU_p_CNN":

			# Create the model
			nameModel = 'GRU_p_CNN__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineGRU_p_CNN(input, len(nNeuronsSequence), nNeuronsSequence, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, args.percentageDropout, len(nNeurons), 
				nNeurons)

			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)

		elif args.network == "LSTM+CNN":

			# Create the model
			nameModel = 'LSTM_CNN__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)				
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineLSTM_CNN(input, len(nNeuronsSequence), nNeuronsSequence, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, args.percentageDropout, len(nNeurons), nNeurons)
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)

		elif args.network == "GRU+CNN":

			# Create the model
			nameModel = 'GRU_CNN__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)				
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineGRU_CNN(input, len(nNeuronsSequence), nNeuronsSequence, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, args.percentageDropout, len(nNeurons), nNeurons)
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)				

		elif args.network == "CNN+LSTM":

			# Create the model
			nameModel = 'CNN_LSTM__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)
			substeps = 1				
			k.clear_session()
			input = Input(shape=(substeps,time_step,num_features,))
			x = defineCNN_LSTM(input, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, len(nNeuronsSequence), nNeuronsSequence, args.percentageDropout, len(nNeurons), nNeurons)
			x_train_2 = x_train.reshape((x_train.shape[0], substeps, time_step, num_features))
			x_test_2 = x_test.reshape((x_test.shape[0], substeps, time_step, num_features))
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train_2, y_train = y_train, x_test = x_test_2, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)

		elif args.network == "CNN+GRU":

			nameModel = 'CNN_GRU__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hnec%s-hne%s-epo%d-seqLen%d-KS%s-cF_%s' % (args.learning_rate, args.batch_size,
			args.percentageDropout, str(nNeuronsSequence), str(nNeuronsConv1D), str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)
			substeps = 1				
			k.clear_session()
			input = Input(shape=(substeps,time_step,num_features,))
			x = defineCNN_GRU(input, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, len(nNeuronsSequence), nNeuronsSequence, args.percentageDropout, len(nNeurons), nNeurons)
			x_train_2 = x_train.reshape((x_train.shape[0], substeps, time_step, num_features))
			x_test_2 = x_test.reshape((x_test.shape[0], substeps, time_step, num_features))
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train_2, y_train = y_train, x_test = x_test_2, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)				

		elif args.network == "LSTM":

			# Create the model
			nameModel = 'LSTM__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d-seqLen%d-cF_%s' % (args.learning_rate, args.batch_size, args.percentageDropout,
				str(nNeuronsSequence), str(nNeurons), args.epochs,time_step,args.campaingsFull)				
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineLSTM(input, len(nNeuronsSequence), nNeuronsSequence, args.percentageDropout, len(nNeurons), nNeurons)				
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)

		elif args.network == "GRU":

			# Create the model
			nameModel = 'GRU__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d-seqLen%d-cF_%s' % (args.learning_rate, args.batch_size, args.percentageDropout, str(nNeuronsSequence), 
				str(nNeurons), args.epochs, time_step, args.campaingsFull)
			k.clear_session()
			input = Input(shape=(time_step,num_features,))
			x = defineGRU(input, len(nNeuronsSequence), nNeuronsSequence, args.percentageDropout, len(nNeurons), nNeurons)
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)				

		elif args.network == "CNN":

			# Create the model
			nameModel = 'CNN__'+str(num_labels_header)+'out-lr%.1e-bs%d-drop%.2f-hnec%s-hne%s-epo%d-seqLen%d-KS%s,cF_%s' % (args.learning_rate, args.batch_size, args.percentageDropout, str(nNeuronsConv1D), 
				str(nNeurons), args.epochs, time_step, str(args.kernelSize), args.campaingsFull)				
			k.clear_session()
			input = Input(shape = (time_step,num_features,))
			x = defineCNN(input, len(nNeuronsConv1D), nNeuronsConv1D, args.kernelSize, args.percentageDropout, len(nNeurons), nNeurons)
			
			Train(lr = args.learning_rate, batch_size = args.batch_size, epochs = args.epochs, percentageDropout = args.percentageDropout, nNeuronsConv1D = nNeuronsConv1D,
				nNeurons = nNeurons, patience_stop = args.patience, patience_reduce_lr = args.patience_reduce_lr, loss_function = args.loss_function, shuffle = args.shuffle, 
				min_delta = args.min_delta, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, num_classes = num_classes, 
				nameExperimentsFolder = nameExperimentsFolder, network = args.network, nameExperiment = args.nameExperiment, experimentFolder = experimentFolder, 
				input = input, x = x, nameModel = nameModel, num_modules_output = num_labels_header)					
	else:
		print("Error. That model is not defined.")
		

if __name__ == '__main__':
	main()