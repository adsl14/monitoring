import numpy as np
import re
import os
import argparse

# Add NDVI to numpy array
def addNDVI(numpy_array):

  """Add NDVI to the dataset.
  Args: 
    features: a dictionary of input tensors keyed by feature name.
    label: the target label
  
  Returns:
    A numpy array with an NDVI added.
  """
  ndvi_array = []
  tam = numpy_array.shape[0]

  for i in range(0,tam):
    ndvi_array.append([normalizedDifference(numpy_array[i,3],numpy_array[i,2])])

  return np.append(numpy_array,ndvi_array, axis=1)

# Normalized the difference between B8 and B4
def normalizedDifference(a, b):
  """Compute normalized difference of two inputs.

  Compute (a - b) / (a + b).  If the denomenator is zero, add a small delta.  

  Args:
    a: B5 value
    b: B4 value

  Returns:
    The normalized difference.
  """
  nd = (a - b) / (a + b)
  nd_inf = (a - b) / (a + b + 0.000001)

  if np.isinf(nd):
    return nd
  else:
    return nd_inf

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

# When the network has finished the training, all the generated models will be erased, unless the three best models.
def cleanExperimentFolder(folderNameExperimentPath):

  models = os.listdir(folderNameExperimentPath)

  # Sort by natural order
  models.sort(key=natural_keys,reverse=True)

  # Remove from the list, the folder 'logs'
  models = models[1:]

  # We check if there is, at least, three min model saved
  if len(models) > 3:
    for i in range (3,len(models)):
      model_path = os.path.join(folderNameExperimentPath,models[i])
      os.remove(model_path)
      print("Experiment %s removed" %(model_path))
  else:
    print("Folder %s ignored" %(folderNameExperimentPath))

# Arguments for the training
def defineArgParsersTrain():

  # All the parameters that has the script
  parser = argparse.ArgumentParser(description='Generate a model.')
  parser.add_argument("--datasets_path", type=str, default='csv',
    help="Path where the dataset CSV files are stored.")
  parser.add_argument("--experiment_folder", type=str, default='experiments',
    help="Path where the experiments are stored.")
  parser.add_argument("--experiment_name", type=str, default='activity',
    help="Name of the experiment.")
  parser.add_argument("--trainDataName", type=str, default='Training_demo.csv',
    help="train dataset's name.")
  parser.add_argument("--testDataName", type=str, default='Testing_demo.csv',
    help="train dataset's name.")
  parser.add_argument("--typeNetwork",type=str, default='LSTM', help="NN, LSTM, LSTM_FCN, CuDNNLSTM,CuDNNLSTM_FCN")
  parser.add_argument("--percentageDropout",type=float, default=0.0, help="How many links of the network will be ommited in order to avoid overfitting")
  parser.add_argument("--batch_size",type=int, default=16, help="Size of batch (number of samples) to evaluate")
  parser.add_argument("--labels",type=str, default='no,yes', help="label for each class type")
  parser.add_argument("--nNeurons",type=str, default='16,8', help="Number of neurons that will be used in the dense layers.")
  parser.add_argument("--nNeuronsSequence",type=str, default='128,128', help="Number of neurons that will be used in the LSTM layers.")
  parser.add_argument("--nNeuronsConv1D",type=str, default='128,256,128', help="Number of neurons that will be used in the Conv layers.")  
  parser.add_argument("--shuffle",type=str2bool, default="n", help="Whether to shuffle the order of the batches at the beginning of each epoch.")
  parser.add_argument("--monitor_stop",type=str, default="val_loss")
  parser.add_argument("--monitor_reduce_lr",type=str, default="val_loss")
  parser.add_argument("--monitor_modelCheckPoint",type=str, default="val_loss")
  parser.add_argument("--patience_stop",type=int, default=50, help="Number of epochs with no improvement after which training will be stopped.")
  parser.add_argument("--patience_reduce_lr",type=int, default=8)
  parser.add_argument("--min_delta",type=float, default=1e-3, help="Minimum change in the monitored quantity to qualify as an improvement.")
  parser.add_argument("--epochs",type=int, default=100, help="Number of epochs.")
  parser.add_argument("--learning_rate",type=float, default=1e-3, help="Learning rate modifier.")
  parser.add_argument("--loss_function",type=str, default="categorical_crossentropy", help="Custom loss function (MSE, L1-smooth)")
  parser.add_argument("--variance_l1smooth",type=float, default=1.0, help="Variance for the L1-smooth loss function")

  return parser.parse_args()

# Arguments for the testing
def defineArgParsersTest():

	# All the parameters that has the script
	parser = argparse.ArgumentParser(description='Test a model.')
	parser.add_argument("--datasets_path", type=str, default='csv',
	help="Path where the dataset CSV files are stored.")
	parser.add_argument("--experiment_folder", type=str, default='experiments',
	help="Path where the experiments are stored.")
	parser.add_argument("--experiment_name", type=str, default='activity',
	help="Name of the experiment.")
	parser.add_argument("--model_parameters_path", type=str, default='',
	help="Parameters name of the model.")
	parser.add_argument("--testDataName", type=str, default='Testing_demo.csv',
	help="test dataset's name.")
	parser.add_argument("--output_name_loss", type=str, default='results_loss.csv',
	help="csv name where we will save the loss of the predictions.")
	parser.add_argument("--output_name_predictions", type=str, default='results_predictions.csv',
	help="csv name where we will save the results of the predictions.")
	parser.add_argument("--labels",type=str, default='no,yes', help="label for each class type")

	return parser.parse_args()

# Check if a string will return 'True' or 'False'
def str2bool(val):

  if val.lower() in ('yes', 'true', 't', 'y', '1'):
    return True

  elif val.lower() in ('no', 'false', 'f', 'n', '0'):
    return False

  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')