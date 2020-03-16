import os
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re

import keras
import keras.backend as k
from keras.models import load_model

from sklearn.metrics import confusion_matrix

from train import configureKerasForGPU, argparse, atoi, natural_keys

from pickle import load

def defineArgParsers():

	parser = argparse.ArgumentParser(description='Test model.')
	parser.add_argument("--networkPath",type=str, default='', help="Path where the model is located")
	parser.add_argument("--campaingPath",type=str, default='', help="Path where the campaing data is located")
	parser.add_argument("--tags_name",type=str, default='', help="Tag filename of the regions")
	parser.add_argument("--percentageGPU",type=float, default=0.0, help="Amount of use the memory of the GPU")

	return parser.parse_args()

def loadOptions(pathOptions):

	dataframe = pd.read_csv(pathOptions)

	indexes = dataframe["indexes"].values[0].strip('][').replace("'","").replace(" ","").split(',')
	labels = dataframe["labels"].values[0].strip('][').replace("'","").replace(" ","").split(',')
	labels_header = dataframe["labels_header"].values[0]
	interpolate =  bool(dataframe["interpolate"].values[0])
	time_step =  int(dataframe["time_step"].values[0])
	num_features = len(indexes)
	num_classes = len(labels)

	return indexes, labels, labels_header, interpolate, time_step, num_features, num_classes;

def loadData(campaingPath, networkPath, indexes, labels, interpolate, time_step, num_features, scaler):

	pathData = os.path.join(campaingPath,"dataset")
	regions = os.listdir(pathData)
	num_regions = len(regions)

	# Sort by natural order
	regions.sort(key=natural_keys,reverse=False)

	sequences = []
	i = 1
	for region in regions:

		# open csv
		areadf = pd.read_csv(os.path.join(pathData,region))
		areadf = areadf[indexes].dropna(how='all')

		# interpolate data
		if(interpolate):
			areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()

		# Change time_step
		seq = areadf.values
		len_seq = len(seq)
		n = time_step - len_seq
		to_concat = np.repeat(seq[-1], n).reshape(num_features, n).transpose()
		seq = np.concatenate([seq, to_concat])

		# Normalize
		seq = seq.reshape((1,time_step*num_features),order='F')
		seq = scaler.transform(seq)
		seq = np.reshape(seq, (time_step, num_features),order='F')

		print("---Cargando datos---")
		print("Progreso %d/%d" %(i,num_regions))
		print("Recinto %s cargado." %(region))
		i=i+1

		# Save in memory
		sequences.append(seq)

	x_data = np.array(sequences)

	# Check if the model needs to reshape the input (CNN)
	modelName = networkPath.split("\\")[-2]
	if modelName[0:3] == "CNN":
		# Reshape due to CNN needs a vector of 4D
		x_data = x_data.reshape((x_data.shape[0], 1, time_step, num_features))

	return x_data, regions, num_regions

def loadDataTag(campaingPath, tags_name, labels_header, indexes, time_step, num_features, num_classes, scaler, interpolate):

	sequences = []
	targets = []
	path_tags = os.path.join(campaingPath, tags_name)

	# Read the test .csv
	tagDataFrame = pd.read_csv(path_tags)
	tagDataFrame = tagDataFrame[tagDataFrame[labels_header] != -1]
	tagDataFrameName = tagDataFrame["id"]
	print(" '%s' de test cargado correctamente" %(tags_name))

	total_test = tagDataFrame.shape[0]

	# Get the sequence for each area (TEST)
	i = 1
	for row in tagDataFrame.values:

		region_path = row[0]
		areadf = pd.read_csv(os.path.join(campaingPath,'dataset',region_path))
		areadf = areadf[indexes].dropna(how='all')

		if(interpolate):
			areadf = areadf.interpolate(method='linear', axis=0).ffill().bfill()

		# Change time_step
		seq = areadf.values
		len_seq = len(seq)
		n = time_step - len_seq
		to_concat = np.repeat(seq[-1], n).reshape(num_features, n).transpose()
		seq = np.concatenate([seq, to_concat])

		# Normalize
		seq = seq.reshape((1,time_step*num_features),order='F')
		seq = scaler.transform(seq)
		seq = np.reshape(seq, (time_step, num_features),order='F')

		print("---Test---")
		print(campaingPath)
		print("Progreso %d/%d" %(i,total_test))
		print("Recinto %s cargado." %(region_path))
		i=i+1

		# Save in memory
		sequences.append(seq)
		targets.append(row[1:])

	x_data = np.array(sequences)
	y_data = np.array(targets)

	# convert class vectors to binary class matrices
	y_data = keras.utils.to_categorical(targets, num_classes)

	unique_samples_test, num_testSamples = np.unique(y_data, axis=0, return_counts=True)
	print("")
	print("Test")
	print(unique_samples_test)
	print(num_testSamples)
	print("Total: %d" %(num_testSamples.sum()))

	return x_data, y_data, tagDataFrameName

def TestModel(model,modelPath,x_test,y_test, labels, steps, features, num_classes, tagDataFrameName, output_path):
  
	# For CNN+LSTM, we had changed the input shape (n_samples, substeps, steps, features)
	if "CNN" == modelPath.split("\\")[-2][0:3]:
		x_test = x_test.reshape((x_test.shape[0], 1, steps, features))
		print("x_test reshaped")

	# Get the predictions
	predictions = model.predict(x_test)

	# Evaluate test data
	score = model.evaluate(x_test, y_test, verbose=0)

	# Get the confussion matrix
	cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

	with open(output_path, mode='w', newline="") as output_file:

		output_writer = csv.writer(output_file, delimiter=',')
		output_writer.writerow(['Name', 'Loss', 'Accuracy'])
		output_writer.writerow([modelPath,score[0],str(round(score[1]*100,2)) + ' %'])
		output_writer.writerow([])

		name_label = ['Real/Predicted']
		name_label.extend(labels)
		output_writer.writerow(name_label)

		# Write confusion matrix in file
		for i in range(0,num_classes):
			row = [labels[i]]
			row.extend(cm[i,:])
			output_writer.writerow(row)

		# Write each area prediction
		num_samples = y_test.shape[0]
		output_writer.writerow([])
		output_writer.writerow(['Area', 'Real', 'Predicted'])
		for i in range(0,num_samples):
			output_writer.writerow([tagDataFrameName[i].split("/")[-1], labels[np.argmax(y_test[i])],labels[np.argmax(predictions[i])]])

		# Closing the file
		output_file.close()

		print('Results saved in %s' % (output_path))

def writePredictions(nameOutput,regions,num_regions,argPredictions):

	with open(nameOutput, mode='w', newline="") as output_file:

		output_writer = csv.writer(output_file, delimiter=',')
		output_writer.writerow(['Nombre', 'Cumple'])

		for i in range(0,num_regions):
			output_writer.writerow([regions[i], argPredictions[i]])

def main():

	# Modify percentageGPU for the experiment
	args = defineArgParsers()
	configureKerasForGPU(args.percentageGPU)

	# Load single model
	if args.networkPath != '':

		# Load model
		model = load_model(args.networkPath)
		print('Model: %s loaded' %(args.networkPath))

		# Load scaler (normalize data)
		tempPath = args.networkPath.split("\\")[0:5]
		scalerPath = os.path.join(tempPath[0],tempPath[1],"scalers",tempPath[3]+"-scaler.pkl")
		scaler = load(open(scalerPath, 'rb'))
		print("Scaler: %s loaded" %(scalerPath))

		# Load options
		indexes, labels, labels_header, interpolate, time_step, num_features, num_classes = loadOptions(os.path.join(tempPath[0],tempPath[1],"options",tempPath[3]+".csv"))

		# Load one campaing
		if args.campaingPath != '':

			# Load tag filename
			if args.tags_name != '':

				# Load data
				x_data, y_data, tagDataFrameName = loadDataTag(args.campaingPath, args.tags_name, labels_header, indexes, time_step, num_features, num_classes, scaler, interpolate)

				TestModel(model, args.networkPath, x_data, y_data, labels, time_step, num_features, num_classes, tagDataFrameName, os.path.join(tempPath[0],tempPath[1],"results",tempPath[3]+"-results_predictions.csv"))

			# Only predict without tags
			else:

				# Load data
				x_data, regions, num_regions = loadData(args.campaingPath, args.networkPath, indexes, labels, interpolate, time_step, num_features, scaler)

				# Get the predictions
				predictions = model.predict(x_data)
				argPredictions = predictions.argmax(axis=1)

				# Write the predictions
				writePredictions(os.path.join(tempPath[0],tempPath[1],"results",tempPath[4]+"-predictions.csv"),regions,num_regions,argPredictions)

		else:

			print("Error -> need a campaing path to load data")

	else:

		print("Loading multiple models")




if __name__ == '__main__':
	main()