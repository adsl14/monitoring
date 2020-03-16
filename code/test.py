import os
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re

import keras.backend as k
from keras.models import load_model

from train import configureKerasForGPU, argparse, atoi, natural_keys

from pickle import load

def defineArgParsers():

	parser = argparse.ArgumentParser(description='Generate a model.')
	parser.add_argument("--networkPath",type=str, default='', help="Path where the model is located")
	parser.add_argument("--campaingPath",type=str, default='', help="Path where the campaing data is located")
	parser.add_argument("--percentageGPU",type=float, default=0.0, help="Amount of use the memory of the GPU")

	return parser.parse_args()

def loadOptions(pathOptions):

	dataframe = pd.read_csv(pathOptions)

	indexes = dataframe["indexes"].values[0].strip('][').replace("'","").replace(" ","").split(',')
	labels = dataframe["labels"].values[0].strip('][').replace("'","").replace(" ","").split(',')
	interpolate =  bool(dataframe["interpolate"].values[0])
	time_step =  int(dataframe["time_step"].values[0])
	num_features = len(indexes)

	return indexes, labels, interpolate, time_step, num_features;

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
		indexes, labels, interpolate, time_step, num_features = loadOptions(os.path.join(tempPath[0],tempPath[1],"options",tempPath[3]+".csv"))

		if args.campaingPath != '':

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