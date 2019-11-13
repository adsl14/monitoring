import numpy as np
import os
import argparse

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

# When the network has finished the training, all the generated models will be erased, unless the three best models.
def cleanExperimentFolder(folderNameExperimentPath):

  models = os.listdir(folderNameExperimentPath)
  models.sort(reverse=True)

  # We check if there is, at least, three min model saved
  if len(models) > 4:
    for i in range (4,len(models)):
      model_path = os.path.join(folderNameExperimentPath,models[i])
      os.remove(model_path)
      print("Experiment %s removed" %(model_path))
  else:
    print("Folder %s ignored" %(folderNameExperimentPath))

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

  # All the parameters that has the script
def defineArgParsers():

  parser = argparse.ArgumentParser(description='Generate a model.')
  parser.add_argument("--datasets_path", type=str, default=os.path.join('csv'),
    help="Path where the dataset CSV files are stored.")
  parser.add_argument("--experiment_folder", type=str, default=os.path.join('experiments'),
    help="Path where the experiments are stored.")
  parser.add_argument("--experiment_name", type=str, default=os.path.join('landsat'),
    help="Name of the experiment.")
  parser.add_argument("--percentageDropout",type=float, default=0.2, help="How many links of the network will be ommited in order to avoid overfitting")
  parser.add_argument("--batch_size",type=int, default=16, help="Size of batch (number of samples) to evaluate")
  parser.add_argument("--num_classes",type=int, default=2, help="Number of classes that the model will use")
  parser.add_argument("--nLayers",type=int, default=2, help="Number of dense layer")
  parser.add_argument("--nNeurons",type=str, default='16,8', help="Number of neurons that will be used in the dense layers. The number of neurons will be divided by two in each layer. ")
  parser.add_argument("--shuffle",type=str2bool, default="y", help="Whether to shuffle the order of the batches at the beginning of each epoch.")
  parser.add_argument("--monitor_stop",type=str, default="acc")
  parser.add_argument("--monitor_reduce_lr",type=str, default="loss")
  parser.add_argument("--patience_stop",type=int, default=50)
  parser.add_argument("--patience_reduce_lr",type=int, default=8)
  parser.add_argument("--min_delta",type=float, default=1e-3, help="Minimum change in the monitored quantity to qualify as an improvement.")
  parser.add_argument("--patience",type=int, default=15, help="Number of epochs with no improvement after which training will be stopped.")
  parser.add_argument("--epochs",type=int, default=100, help="Number of epochs.")
  parser.add_argument("--learning_rate",type=float, default=1e-3, help="Learning rate modifier.")
  parser.add_argument("--loss_function",type=str, default="categorical_crossentropy", help="Custom loss function (MSE, L1-smooth)")
  parser.add_argument("--variance_l1smooth",type=float, default=1.0, help="Variance for the L1-smooth loss function")

  return parser.parse_args()

# Check if a string will return 'True' or 'False'
def str2bool(val):

  if val.lower() in ('yes', 'true', 't', 'y', '1'):
    return True

  elif val.lower() in ('no', 'false', 'f', 'n', '0'):
    return False

  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')