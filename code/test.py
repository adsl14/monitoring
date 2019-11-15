from functions import defineArgParsersTest, addNDVI
from keras.models import load_model
import keras
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

def show_confussionMatrix(matrix,labels):

	row = matrix.shape[0]
	cols = matrix.shape[1]

	print('Real | Predicted | Amount')

	for i in range(0,row):
		for j in range(0,cols):
			print("%s | %s | %d" % (labels[i],labels[j],matrix[i,j]))

def testModel(args):

	labels = args.labels.split(',')
	num_classes = len(labels)

	# Get the path where the csv is located
	parentPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	testFilePath= os.path.join(parentPath,args.datasets_path,args.testDataName)

	# Load csv test
	testDataFrame = pd.read_csv(testFilePath)
	print('csv for testing loaded')

	# Get a numpy array of the dataframe
	test_n = testDataFrame.to_numpy()

	# Split the input and output values
	x_test = test_n[:,:-1]
	y_test = test_n[:,-1]

	# convert class vectors to binary class matrices
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# Add NDVI
	x_test = addNDVI(x_test)

	# Number of features
	num_features_input = x_test.shape[1]

	# Get a list of all saved models
	ExperimentPath = os.path.join(args.experiment_folder,args.experiment_name,args.model_parameters)
	models = os.listdir(ExperimentPath)
	models.sort(key=lambda x: os.path.getmtime(os.path.join(ExperimentPath,x)),reverse=True)

	# Load the best model for that experiment
	model = load_model(os.path.join(ExperimentPath,models[0]))

	# Get the predictions
	predictions = model.predict(x_test)

	# Confusion matrix
	cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

	# Evaluate test data
	score = model.evaluate(x_test, y_test, verbose=0)

	# Clean terminal
	print('\033c')

	print("RESULTS")
	print("------------------------")
	print("Confusion matrix")
	show_confussionMatrix(cm,labels)
	print("------------------------")
	print("Score")
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	print("------------------------")


def main():

	args = defineArgParsersTest()

	testModel(args)

if __name__ == "__main__":
	main()