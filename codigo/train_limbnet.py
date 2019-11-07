import csv
import cv2
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import random as rn
import threading
import tensorflow as tf
import sys
from shutil import move

# Fix the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)
import keras.backend as k
sess = tf.get_default_session()
k.set_session(sess)

import keras as kr
from keras import applications # modelo de red
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Input, Dropout, GaussianNoise, ZeroPadding2D, Activation, add, Add, Concatenate
from keras.models import Model, load_model
from keras.utils import get_file, convert_all_kernels_in_model, Sequence
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator, load_img, save_img
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.resnet import ResNet50, ResNet101, ResNet152
from keras_applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras_applications.resnext import ResNeXt50, ResNeXt101
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session

# class Datagenerator used by the network that has double output (pose and angle)
class DataGenerator(Sequence):

	"""Generates data for Keras.""" 
	def __init__(self, comunPathImages, comunPathCSV, nameCSV, dataGenerator, batch_size=16, shuffle=True, dim=(224, 224), n_channels=3):

		# Load DataFrames for positions 3D
		dataFrame = pd.read_csv(os.path.join(comunPathCSV,nameCSV), header=0)
		print("csv loaded.")

		# Initialization.
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dim = dim
		self.n_channels = n_channels
		self.dataFrame = dataFrame

		# Get the data. Check if the images first exists. Then, we will generate the batches
		self.input_images = []
		self.output_poses = []
		self.output_angles = []

		for i in range(0,dataFrame.shape[0]):

			name_image = os.path.join(comunPathImages, dataFrame.iloc[i,0])

			if os.path.exists(name_image):
				self.input_images.append([os.path.join(comunPathImages, dataFrame.iloc[i,0])])
				self.output_poses.append(dataFrame.iloc[i,1:19].values)
				self.output_angles.append(dataFrame.iloc[i,19])

		self.input_images = np.array(self.input_images)
		self.output_poses = np.array(self.output_poses)
		self.output_angles = np.array(self.output_angles)

		# This won't check if the images realy exists in the folder. It will give error if the image doesn't exists
		# self.input_images = dataFrame[["name"]]
		# self.input_images = self.input_images.applymap(lambda img_path: os.path.join(comunPathImages, img_path)).values
		# self.output_poses = dataFrame.iloc[:,1:19].values
		# self.output_angles = dataFrame.iloc[:,19].values

		# Initialize ImageDataGenerator for data augmentation
		self.img_generator = dataGenerator

		self.on_epoch_end()

	def __len__(self):

		# Denotes the number of batches per epoch.
		return (np.ceil(len(self.input_images) / self.batch_size)).astype(int)

	def __getitem__(self, index):

		# Generate one batch of data.
		# Generate indexes of the batch
		indexMax = (index+1)*self.batch_size
		if indexMax <= len(self.input_images):
			indexes = self.indexes[index*self.batch_size:indexMax]
		else:
			indexes = self.indexes[index*self.batch_size:]

		# Generate data
		batch_x, batch_y = self.__data_generation(indexes)

		return batch_x, batch_y

	def on_epoch_end(self):

		# Updates indexes after each epoch
		self.indexes = np.arange(len(self.input_images))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_indexes):

		"""Generates data containing batch_size samples"""
		input_images = []
		output_poses = []
		output_angles = []

		for i in list_indexes:

			# print("%d + %s" % (i+2,self.input_images[i]))

			# Load the  images of current sample
			img = cv2.imread(self.input_images[i][0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			transform_dictionary = self.img_generator.get_random_transform(img.shape, seed=42)
			img = self.img_generator.apply_transform(x=img, transform_parameters=transform_dictionary)
			img = img/255.

			# Append to batch list
			input_images.append(img)
			output_poses.append(self.output_poses[i])
			output_angles.append(self.output_angles[i])

		batch_x = [np.array(input_images)]
		batch_y = [np.array(output_poses), np.array(output_angles)]

		return batch_x, batch_y

# class DataGeneratorPose3D_ImageAnd2DKP used by the network that has double input (image and 2D values)
class DataGeneratorPose3D_ImageAnd2DKP(Sequence):

	"""Generates data for Keras.""" 
	def __init__(self, comunPathImages, comunPathCSV, nameCSV, dataGenerator, batch_size=16, shuffle=True, dim=(224, 224), n_channels=3):

		# Load DataFrames for positions 3D
		dataFrame = pd.read_csv(os.path.join(comunPathCSV,nameCSV), header=0)
		print("csv loaded.")

		# Initialization.
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dim = dim
		self.n_channels = n_channels
		self.dataFrame = dataFrame

		# Get the data. Check if the images first exists. Then, we will generate the batches
		self.input_images = []
		self.input_2dkeypoint = []
		self.output_poses = []

		for i in range(0,dataFrame.shape[0]):

			name_image = os.path.join(comunPathImages, dataFrame.iloc[i,0])

			if os.path.exists(name_image):
				self.input_images.append([os.path.join(comunPathImages, dataFrame.iloc[i,0])])
				self.input_2dkeypoint.append(dataFrame.iloc[i,1:13].values)
				self.output_poses.append(dataFrame.iloc[i,13:].values)

		self.input_images = np.array(self.input_images)
		self.input_2dkeypoint = np.array(self.input_2dkeypoint)
		self.output_poses = np.array(self.output_poses)

		# This won't check if the images realy exists in the folder. It will give error if the image doesn't exists
		# self.input_images = dataFrame[["name"]]
		# self.input_images = self.input_images.applymap(lambda img_path: os.path.join(comunPathImages, img_path)).values
		# self.input_2dkeypoint = dataFrame.iloc[:,1:13].values
		# self.output_poses = dataFrame.iloc[:,13:].values

		# Initialize ImageDataGenerator for data augmentation
		self.img_generator = dataGenerator

		self.on_epoch_end()

	def __len__(self):

		# Denotes the number of batches per epoch.
		return (np.ceil(len(self.input_images) / self.batch_size)).astype(int)

	def __getitem__(self, index):

		# Generate one batch of data.
		# Generate indexes of the batch
		indexMax = (index+1)*self.batch_size
		if indexMax <= len(self.input_images):
			indexes = self.indexes[index*self.batch_size:indexMax]
		else:
			indexes = self.indexes[index*self.batch_size:]

		# Generate data
		batch_x, batch_y = self.__data_generation(indexes)

		return batch_x, batch_y

	def on_epoch_end(self):

		# Updates indexes after each epoch
		self.indexes = np.arange(len(self.input_images))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_indexes):

		"""Generates data containing batch_size samples"""
		input_images = []
		input_2dkeypoint = []
		output_poses = []

		for i in list_indexes:

			# print("%d + %s" % (i+2,self.input_images[i]))

			# Load the  images of current sample
			img = cv2.imread(self.input_images[i][0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			transform_dictionary = self.img_generator.get_random_transform(img.shape, seed=42)
			img = self.img_generator.apply_transform(x=img, transform_parameters=transform_dictionary)
			img = img/255.

			# Append to batch list
			input_images.append(img)
			input_2dkeypoint.append(self.input_2dkeypoint[i])
			output_poses.append(self.output_poses[i])

		batch_x = [np.array(input_images), np.array(input_2dkeypoint)]
		batch_y = [np.array(output_poses)]

		return batch_x, batch_y

# class DataGeneratorPose3DAngle_ImageAnd2DKP used by the network that has double input (image and 2D values) and double output (pose and angle)
class DataGeneratorPose3DAngle_ImageAnd2DKP(Sequence):

	"""Generates data for Keras.""" 
	def __init__(self, comunPathImages, comunPathCSV, nameCSV, dataGenerator, batch_size=16, shuffle=True, dim=(224, 224), n_channels=3):

		# Load DataFrames for positions 3D
		dataFrame = pd.read_csv(os.path.join(comunPathCSV,nameCSV), header=0)
		print("csv loaded.")

		# Initialization.
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dim = dim
		self.n_channels = n_channels
		self.dataFrame = dataFrame

		# Get the data. Check if the images first exists. Then, we will generate the batches
		self.input_images = []
		self.input_2dkeypoint = []
		self.output_poses = []
		self.output_angles = []

		for i in range(0,dataFrame.shape[0]):

			name_image = os.path.join(comunPathImages, dataFrame.iloc[i,0])

			if os.path.exists(name_image):
				self.input_images.append([os.path.join(comunPathImages, dataFrame.iloc[i,0])])
				self.input_2dkeypoint.append(dataFrame.iloc[i,1:13].values)
				self.output_poses.append(dataFrame.iloc[i,13:-1].values)
				self.output_angles.append(dataFrame.iloc[i,-1])

		self.input_images = np.array(self.input_images)
		self.input_2dkeypoint = np.array(self.input_2dkeypoint)
		self.output_poses = np.array(self.output_poses)
		self.output_angles = np.array(self.output_angles)

		# This won't check if the images realy exists in the folder. It will give error if the image doesn't exists
		# self.input_images = dataFrame[["name"]]
		# self.input_images = self.input_images.applymap(lambda img_path: os.path.join(comunPathImages, img_path)).values
		# self.input_2dkeypoint = dataFrame.iloc[:,1:13].values
		# self.output_poses = dataFrame.iloc[:,13:-1].values
		# self.output_angles = dataFrame.iloc[:,-1].values

		# Initialize ImageDataGenerator for data augmentation
		self.img_generator = dataGenerator

		self.on_epoch_end()

	def __len__(self):

		# Denotes the number of batches per epoch.
		return (np.ceil(len(self.input_images) / self.batch_size)).astype(int)

	def __getitem__(self, index):

		# Generate one batch of data.
		# Generate indexes of the batch
		indexMax = (index+1)*self.batch_size
		if indexMax <= len(self.input_images):
			indexes = self.indexes[index*self.batch_size:indexMax]
		else:
			indexes = self.indexes[index*self.batch_size:]

		# Generate data
		batch_x, batch_y = self.__data_generation(indexes)

		return batch_x, batch_y

	def on_epoch_end(self):

		# Updates indexes after each epoch
		self.indexes = np.arange(len(self.input_images))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_indexes):

		"""Generates data containing batch_size samples"""
		input_images = []
		output_poses = []
		output_angles = []
		input_2dkeypoint = []

		for i in list_indexes:

			# print("%d + %s" % (i+2,self.input_images[i]))

			# Load the  images of current sample
			img = cv2.imread(self.input_images[i][0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			transform_dictionary = self.img_generator.get_random_transform(img.shape, seed=42)
			img = self.img_generator.apply_transform(x=img, transform_parameters=transform_dictionary)
			img = img/255.

			# Append to batch list
			input_images.append(img)
			input_2dkeypoint.append(self.input_2dkeypoint[i])
			output_poses.append(self.output_poses[i])
			output_angles.append(self.output_angles[i])

		batch_x = [np.array(input_images),np.array(input_2dkeypoint)]
		batch_y = [np.array(output_poses), np.array(output_angles)]

		return batch_x, batch_y

# Layer used by customResnet functions. Not used.
def block1(x, filters, kernel_size=3, stride=1,conv_shortcut=True, name=None):

	bn_axis = 3

	if conv_shortcut is True:
		shortcut = Conv2D(4 * filters, 1, strides=stride,
								 name=name + '_0_conv')(x)
		shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
											 name=name + '_0_bn')(shortcut)
	else:
		shortcut = x

	x = Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
								  name=name + '_1_bn')(x)
	x = Activation('relu', name=name + '_1_relu')(x)

	x = Conv2D(filters, kernel_size, padding='SAME',
					  name=name + '_2_conv')(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
								  name=name + '_2_bn')(x)
	x = Activation('relu', name=name + '_2_relu')(x)

	x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
								  name=name + '_3_bn')(x)

	x = Add(name=name + '_add')([shortcut, x])
	x = Activation('relu', name=name + '_out')(x)
	return x

# Layer used by customResnet functions. Not used.
def stack1(x, filters, blocks, stride1=2, name=None):

	x = block1(x, filters, stride=stride1, name=name + '_block1')

	for i in range(2, blocks + 1):
		x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

	return x

# VGG16 custom. It's exactly the same as the original VGG16, but adding, just after the input image layer, a gaussian layer.
def customVGG16(args):

	# Create the custom VGG16
	input_shape = _obtain_input_shape((224, 224,3),default_size=224,min_size=32,data_format=k.image_data_format(),require_flatten=False,
		weights='imagenet')
	img_input = Input(shape=input_shape)

	# Add gaussian layer
	x = GaussianNoise(args.stddev)(img_input)

	# Block 1
	x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1')(x)
	x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	# Block 2
	x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv1')(x)
	x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	# Block 3
	x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
	x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
	x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	# Block 4
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1')(x)
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	# Block 5
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
	x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	# Create model.
	inputs = img_input
	model = Model(inputs, x, name='vgg16')

	# Load weights
	weights_path = os.path.join("h5_files","vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
	model.load_weights(weights_path, by_name=True)

	return model

# RESNET. Not used.
def customResNet(args,stack_fn,preact,use_bias,model_name='resnet'):

	# Determine proper input shape
	input_shape = _obtain_input_shape((224, 224,3),default_size=224,min_size=32,data_format=k.image_data_format(),require_flatten=False,weights='imagenet')

	img_input = Input(shape=input_shape)

	bn_axis = 3

	# Add gaussian layer
	x = GaussianNoise(args.stddev)(img_input)

	x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
	x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

	if preact is False:
		x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='conv1_bn')(x)
		x = Activation('relu', name='conv1_relu')(x)

	x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
	x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

	x = stack_fn(x)

	if preact is True:
		x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='post_bn')(x)
		x = layers.Activation('relu', name='post_relu')(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	inputs = img_input

	# Create model.
	model = Model(inputs, x, name=model_name)

	# Load weights.
	weights_path = os.path.join("h5_files",model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5')
	model.load_weights(weights_path)

	return model

# RESNET50. Not used.
def customResNet50(args):
	def stack_fn(x):
		x = stack1(x, 64, 3, stride1=1, name='conv2')
		x = stack1(x, 128, 4, name='conv3')
		x = stack1(x, 256, 6, name='conv4')
		x = stack1(x, 512, 3, name='conv5')
		return x
	return customResNet(args,stack_fn, False, True, 'resnet50')

# RESNET50V2. Not used.
def customResNet50V2(args):
	def stack_fn(x):
		x = stack2(x, 64, 3, name='conv2')
		x = stack2(x, 128, 4, name='conv3')
		x = stack2(x, 256, 6, name='conv4')
		x = stack2(x, 512, 3, stride1=1, name='conv5')
		return x
	return customResNet(args,stack_fn, True, True, 'resnet50v2')

# RESNET101. Not used.
def customResNet101(args):
	def stack_fn(x):
		x = stack1(x, 64, 3, stride1=1, name='conv2')
		x = stack1(x, 128, 4, name='conv3')
		x = stack1(x, 256, 23, name='conv4')
		x = stack1(x, 512, 3, name='conv5')
		return x
	return customResNet(args,stack_fn, False, True, 'resnet101')

# RESNET101V2. Not used.
def customResNet101V2(args):
	def stack_fn(x):
		x = stack2(x, 64, 3, name='conv2')
		x = stack2(x, 128, 4, name='conv3')
		x = stack2(x, 256, 23, name='conv4')
		x = stack2(x, 512, 3, stride1=1, name='conv5')
		return x
	return customResNet(args,stack_fn, True, True, 'resnet101v2')

# RESNET152V2. Not used.
def customResNet152V2(args):
	def stack_fn(x):
		x = stack2(x, 64, 3, name='conv2')
		x = stack2(x, 128, 8, name='conv3')
		x = stack2(x, 256, 36, name='conv4')
		x = stack2(x, 512, 3, stride1=1, name='conv5')
		return x
	return customResNet(args,stack_fn, True, True, 'resnet152v2')

# RESNEXt50. Not used.
def customResNeXt50(args):
	def stack_fn(x):
		x = stack3(x, 128, 3, stride1=1, name='conv2')
		x = stack3(x, 256, 4, name='conv3')
		x = stack3(x, 512, 6, name='conv4')
		x = stack3(x, 1024, 3, name='conv5')
		return x
	return customResNet(args,stack_fn, False, False, 'resnext50')

# RESNEXt101. Not used.
def customResNeXt101(args):
	def stack_fn(x):
		x = stack3(x, 128, 3, stride1=1, name='conv2')
		x = stack3(x, 256, 4, name='conv3')
		x = stack3(x, 512, 23, name='conv4')
		x = stack3(x, 1024, 3, name='conv5')
		return x
	return customResNet(args,stack_fn, False, False, 'resnext101')

# MSE for the output pose estimation
def MSE_pose3D(clusters):

	def cluster_mse(y_true, y_pred):

		# Get the poses as outputs
		# y_pred = k.expand_dims(y_pred, axis=-1)
		# y_pred = k.sum(clusters*y_pred, axis=1)

		y_pred = k.dot(y_pred,clusters)
		
		# Wrong MSE, but we are using to compare the olders models (just forgot to divide in 18 (number of elements in 3D pose))
		return k.mean(k.sum(k.square(y_pred - y_true), axis=-1), axis=-1)
		
		# Good MSE
		# return k.mean(k.square(y_true - y_pred))

	return cluster_mse

# L1-smooth for the output pose estimation
def L1_smooth_pose3D(clusters,variance_l1smooth):

	def smooth_l1(y_true, y_pred):

		y_pred = k.dot(y_pred,clusters)

		loss = k.abs(y_pred - y_true)
		loss = k.switch(loss <= 1/variance_l1smooth, 0.5*variance_l1smooth*k.pow(loss, 2), loss - 0.5/variance_l1smooth)

		return k.sum(loss)

	return smooth_l1

# Check if a string will return 'True' or 'False'
def str2bool(val):

	if val.lower() in ('yes', 'true', 't', 'y', '1'):
		return True

	elif val.lower() in ('no', 'false', 'f', 'n', '0'):
		return False

	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

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
	set_session(sess)

# All the parameters that has the script
def defineArgParsers():

	parser = argparse.ArgumentParser(description='Generate a model.')
	parser.add_argument("--datasets_path", type=str, default=os.path.join('csv_files'),
		help="Path where the dataset CSV files are stored.")
	parser.add_argument("--experiment_folder", type=str, default=os.path.join('experiments'),
		help="Path where the experiments are stored.")
	parser.add_argument("--experiment_name", type=str, default=os.path.join('position'),
		help="Name of the experiment.")
	parser.add_argument("--network",type=str, default='vgg16', help="Select the network you want to use (vgg16, resnet50, resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2, resnext50, resnext101)")
	parser.add_argument("--percentageDropout",type=float, default=0.2, help="How many links of the network will be ommited in order to avoid overfitting")
	parser.add_argument("--batch_size",type=int, default=16, help="Size of batch (number of samples) to evaluate")
	parser.add_argument("--k",type=int, default=64, help="Number of clusters to load")
	parser.add_argument("--stddev",type=float, default=0.1, help="Standard deviation of the noise distribution. It is only active at training time")
	parser.add_argument("--nLayers",type=int, default=2, help="Number of dense layer")
	parser.add_argument("--nNeurons",type=int, default=1024, help="Number of neurons that will be used in the dense layers. The number of neurons will be divided by two in each layer. ")
	parser.add_argument("--percentageSample",type=float, default=1.0, help="Percentage of subsample that the model will use")
	parser.add_argument("--percentageGPU",type=float, default=0.0, help="Amount of use the memory of the GPU")
	parser.add_argument("--GPU",type=str, default="0", help="What GPU we will use")
	parser.add_argument("--horizontal_flip",type=str2bool, default="n", help="Flip the images in horizontal in the generator")
	parser.add_argument("--vertical_flip",type=str2bool, default="n", help="Flip the images in vertical in the generator")
	parser.add_argument("--shuffle",type=str2bool, default="y", help="Whether to shuffle the order of the batches at the beginning of each epoch.")
	parser.add_argument("--class_mode",type=str, default="other", help="One of categorical, binary, sparse, input, other or None. Default: other.")
	parser.add_argument("--monitor",type=str, default="val_loss", help="Quantity to be monitored.")
	parser.add_argument("--min_delta",type=float, default=1e-3, help="Minimum change in the monitored quantity to qualify as an improvement.")
	parser.add_argument("--patience",type=int, default=15, help="Number of epochs with no improvement after which training will be stopped.")
	parser.add_argument("--verbose",type=int, default=1, help="Verbose mode.")
	parser.add_argument("--epochs",type=int, default=40, help="Number of epochs.")
	parser.add_argument("--mode",type=str, default='min', help="One of {auto, min, max}. If the main target of the model is to maximize or minimize. For val_acc, this should be max, for val_loss this should be min.")
	parser.add_argument("--baseline",type=int, default=None, help="Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.")
	parser.add_argument("--restore_best_weights",type=str2bool, default="y", help="Whether to restore model weights from the epoch with the best value of the monitored quantity.")
	parser.add_argument("--save_best_only",type=str2bool, default="y", help="The latest best model according to the quantity monitored will not be overwritten.")
	parser.add_argument("--save_weights_only",type=str2bool, default="n", help="The model will save, or not, the weights of the trained model.")
	parser.add_argument("--period",type=int, default=1, help="Interval (number of epochs) between checkpoints.")
	parser.add_argument("--use_multiprocessing",type=str2bool, default="y", help="Use multiproccessing or not.")
	parser.add_argument("--workers",type=int, default=6, help="Maximum number of processes to spin up when using process-based threading.")
	parser.add_argument("--max_queue_size",type=int, default=10, help="Maximum size for the generator queue.")
	parser.add_argument("--learning_rate",type=float, default=1e-4, help="Learning rate modifier.")
	parser.add_argument("--freezeConvolutionalWeight",type=str2bool, default="n", help="Freeze the training of the convolutional weights ('y' are trainable, 'n' aren't trainable)")
	parser.add_argument("--loss_function",type=str, default="MSE", help="Custom loss function (MSE, L1-smooth)")
	parser.add_argument("--variance_l1smooth",type=float, default=1.0, help="Variance for the L1-smooth loss function")
	parser.add_argument("--priority_pose3D",type=float, default=1.0, help="Amount of priority for pose estimation (only in pose-angle)")
	parser.add_argument("--priority_angle",type=float, default=1.0, help="Amount of priorty for angle estimation (only in pose-angle)")
	parser.add_argument("--COCO",type=str2bool, default="n", help="Use samples of COCO")
	parser.add_argument("--COCO_fullTrain",type=str2bool, default="n", help="Use all the samples of COCO for training (y) or use only train+val for training (n)")
	parser.add_argument("--combine2Dkp",type=str2bool, default="n", help="Combine in the input the image and 2D keypoints")
	parser.add_argument("--modelOutputType",type=str, default="pose", help="Select the model type output (pose, pose-angle)")
	parser.add_argument("--Surreal",type=str2bool, default="n", help="Use samples of Surreal")


	return parser.parse_args()

# Loading the base model from Keras
def loadBaseModel(args):

	# Load the model
	if args.network == "resnet50":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet50(args)
		model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnet50v2":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet50V2(args)
		model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnet101":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet101(args)
		model = ResNet101(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnet101v2":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet101V2(args)
		model = ResNet101V2(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnet152":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet152(args)
		model = ResNet152(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnet152v2":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNet152V2(args)
		model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnext50":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNeXt50(args)
		model = ResNeXt50(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	elif args.network == "resnext101":

		if args.percentageGPU >= 0.6:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.6)

		# model = customResNeXt101(args)
		model = ResNeXt101(include_top=False, weights='imagenet', input_shape=(224,224,3), backend = kr.backend, layers = kr.layers, models = kr.models, utils = kr.utils)
	
	else:

		if args.percentageGPU >= 0.3:
			configureKerasForGPU(args.percentageGPU)
		else:
			configureKerasForGPU(0.3)

		model = customVGG16(args)

	return model

# Add custom layers for the network pose
def AddLayersForPositions3D(args,model,clustersDataFrame):

	# Make sure layers are set to trainable or not
	for i in range(0,len(model.layers)):
		model.layers[i].trainable = args.freezeConvolutionalWeight

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)

	# If the network is resnet, we won't add the dropout layer
	if not args.network in ("resnet50", "resnet50v2", "resnet101", "resnet101v2", "resnet152", "resnet152v2", "resnext50", "resnext101"):
		x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting

	x = Dense(clustersDataFrame.shape[0], name='pose3D')(x)

	model = Model(inputs=model.input, outputs=x)

	return model

# Add custom layers for the network pose+2D
def AddLayersForPositions3D_imageAnd2DarrayInput(args,model,clustersDataFrame):

	# Make sure layers are set to trainable or not
	for i in range(0,len(model.layers)):
		model.layers[i].trainable = args.freezeConvolutionalWeight

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	# Add double input for the network (image+array)
	vector_input = Input((12,))
	x = Concatenate()([vector_input, x])

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)

	# If the network is resnet, we won't add the dropout layer
	if not args.network in ("resnet50", "resnet50v2", "resnet101", "resnet101v2", "resnet152", "resnet152v2", "resnext50", "resnext101"):
		x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting

	x = Dense(clustersDataFrame.shape[0], name='pose3D')(x)

	model = Model(inputs=[model.input,vector_input], outputs=x)

	return model

# Train the model that estimate only pose 3D
def trainModelForPositions3D(args):

	# percentage from the original dataSet
	if args.percentageSample <= 0.0 or args.percentageSample > 1.0:
		args.percentageSample = 1.0

	# If the network we want to load is not vgg16, then, is resnet.
	if args.network != "vgg16":
		args.percentageDropout=0.0
	else:
		args.network = "vgg16"

	# Load DataFrames for positions 3D
	# Using COCO samples
	if args.COCO:

		# Using COCO train+val = train, test=val
		if not args.COCO_fullTrain:
			print("Loading csv_files...")
			trainDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.0034",'dataSetTrainPosition3D_n0.0034-cc(train+val)-rv-ws-path.csv'))
			print("csv_train loaded.")
			valDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.1", 'dataSetValPosition3D_n0.1-cc(test)-rv-path.csv'))
			print("csv_val loaded.")
			clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
			print("csv_clusters loaded.")

			folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (
				args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, args.max_queue_size, 
				args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, args.freezeConvolutionalWeight, 
				args.variance_l1smooth)

		# Using all the dataSet of COCO for training
		else:

			# Using surreal as training
			if args.Surreal:
				print("Loading csv_files...")
				trainDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.0034",'dataSetTrainPosition3D_n0.0034-sr(DB+OP)-cc(train+val+test)-rv-ws-path.csv'))
				print("csv_train loaded.")
				valDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.1", 'dataSetValPosition3D_n0.1-rv-path.csv'))
				print("csv_val loaded.")
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'sr_k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")

				folderNameExperiment = ("Network%s-SR_CC-sr_k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (args.network, 
					clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.variance_l1smooth)
			else:
				print("Loading csv_files...")
				trainDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.0034",'dataSetTrainPosition3D_n0.0034-cc(train+val+test)-rv-ws-path.csv'))
				print("csv_train loaded.")
				valDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.1", 'dataSetValPosition3D_n0.1-rv-path.csv'))
				print("csv_val loaded.")
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")

				folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (
					args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, args.max_queue_size, 
					args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, args.freezeConvolutionalWeight, 
					args.variance_l1smooth)
	# Using only synthetic and real samples				
	else:
		print("Loading csv_files...")
		trainDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.0034",'dataSetTrainPosition3D_n0.0034-rv-ws-path.csv'))
		print("csv_train loaded.")
		valDataFrame = pd.read_csv(os.path.join(args.datasets_path, "subSample0.1", 'dataSetValPosition3D_n0.1-rv-path.csv'))
		print("csv_val loaded.")
		clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
		print("csv_clusters loaded.")

		folderNameExperiment = ("Network%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (
		args.network,clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, args.max_queue_size, 
		args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, args.freezeConvolutionalWeight, 
		args.variance_l1smooth)

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, brightness_range=[0.8,1.3], rotation_range=20, shear_range=15.0, width_shift_range=10,
		height_shift_range=10)

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	print('Loading images for Train')
	train_generator = train_datagenerator.flow_from_dataframe(dataframe=trainDataFrame, directory=os.path.join('pre_processed_images'), 
		x_col="name", y_col=['x_left_waist','y_left_waist','z_left_waist','x_left_knee','y_left_knee','z_left_knee', 
			'x_left_ankle','y_left_ankle', 'z_left_ankle', 'x_right_waist','y_right_waist','z_right_waist',
			'x_right_knee','y_right_knee','z_right_knee', 'x_right_ankle','y_right_ankle','z_right_ankle'], has_ext=False, class_mode=args.class_mode, target_size=[224,224], 
		batch_size=args.batch_size, seed=42)
	print('Done')

	print('Loading images for Validation')
	val_generator = test_datagenerator.flow_from_dataframe(dataframe=valDataFrame,directory=os.path.join('pre_processed_images'), 
		x_col="name", y_col=['x_left_waist','y_left_waist','z_left_waist','x_left_knee','y_left_knee','z_left_knee', 
			'x_left_ankle','y_left_ankle', 'z_left_ankle', 'x_right_waist','y_right_waist','z_right_waist',
			'x_right_knee','y_right_knee','z_right_knee', 'x_right_ankle','y_right_ankle','z_right_ankle'], has_ext=False, class_mode=args.class_mode, target_size=[224,224], 
		batch_size=args.batch_size, seed=42)
	print('Done')

	# We will load or create the model
	print('Creating the model...')

	# Loading the clusters
	clusters_array = tf.cast(tf.constant(clustersDataFrame.iloc[:,1:]),dtype=tf.float32)

	# Creating the folder for the experiments. Check first is the new experiment has been created before
	args.experiment_name = "position3D_n"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))) and not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))):

		os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
		model = loadBaseModel(args)
		model = AddLayersForPositions3D(args,model,clustersDataFrame)
		print("network %s created" % (args.network))

	# Ignore the experiment
	else:
		print("network %s ignored" % (args.network))
		sys.exit()
	# # We will load or create the experiment
	# else:

	# 	print("Folder already saved. Checking if there is a network saved...")

	# 	models = []
	# 	models_old = []

	# 	try:
	# 		models = os.listdir(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
	# 		models.sort(reverse=True)
	# 		models_old = []
	# 	except Exception:
	# 		models_old = os.listdir(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))
	# 		models_old.sort(reverse=True)
	# 		pass

	# 	# We check if there is, at least, one min model saved in the old folders or the actual one
	# 	if len(models) > 1:

	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	elif len(models_old) > 1:

	# 		# We move the experiment from the 'old' to the actual one in order to see it evolution
	# 		move(os.path.join(args.experiment_folder,args.experiment_name+"(old)",folderNameExperiment),
	# 			os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment))
	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	else:
	# 		print("Didn't find any model. Creating a new one...")
	# 		model = loadBaseModel(args)
	# 		model = AddLayersForPositions3D(args,model,clustersDataFrame)
	# 		print("network %s created" % (args.network))

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_cluster_mse:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-08))

	# Load custom loss function
	# L1-smooth
	if args.loss_function == "L1-smooth":
		print("L1-smooth activated")
		model.compile(
			loss=L1_smooth_pose3D(clusters_array,args.variance_l1smooth),
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D' : MSE_pose3D(clusters_array)})
	# MSE
	else:
		print("MSE activated")
		model.compile(
			loss=MSE_pose3D(clusters_array),
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D' : MSE_pose3D(clusters_array)})

	# Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Clean the folder. This will leave only the three best models
	cleanExperimentFolder(args,folderNameExperiment)

# Train the model that estimate only pose 3D, but has, in the input, an image and 2D values
def trainModelForPositions3D_imageAnd2DarrayInput(args):

	# percentage from the original dataSet
	if args.percentageSample <= 0.0 or args.percentageSample > 1.0:
		args.percentageSample = 1.0

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, brightness_range=[0.8,1.3], rotation_range=20, shear_range=15.0, width_shift_range=10,
		height_shift_range=10)

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	# If the network we want to load is not vgg16, then, is resnet.
	if args.network != "vgg16":
		args.percentageDropout=0.0
	else:
		args.network = "vgg16"

	# Loading the images
	# Using COCO samples
	if args.COCO:
		# Using COCO train+val = train, test=val
		if not args.COCO_fullTrain:
			print('Loading images for Train')
			train_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n0.0034(openPose)-cc(train+val)-rv-ws-path.csv", 
				dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			print('Loading images for Validation')
			val_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n0.1(openPose)-cc(test)-rv-path.csv", 
				dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			# Loading cluster
			clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
			print("csv_clusters loaded.")			

			folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (
				args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
				args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
				args.freezeConvolutionalWeight, args.variance_l1smooth)
		# Using all the dataSet of COCO for training
		else:

			# Using Surreal
			if args.Surreal:
				print('Loading images for Train')
				train_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n0.0034-sr(DB+OP)-(openPose)-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n0.1(openPose)-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'sr_k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")								

				folderNameExperiment = ("Network%s-SR_CC-sr_k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (args.network, 
					clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.variance_l1smooth)
			else:
				print('Loading images for Train')
				train_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n0.0034(openPose)-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n0.1(openPose)-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")				

				folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-(\u03C3^2)%.2f") % (
					args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.variance_l1smooth)
	# Using only synthetic and real samples
	else:
		print('Loading images for Train')
		train_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n0.0034(openPose)-rv-ws-path.csv", 
			dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		print('Loading images for Validation')
		val_generator = DataGeneratorPose3D_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n0.1(openPose)-rv-path.csv", 
			dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		# Loading cluster
		clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
		print("csv_clusters loaded.")				

		folderNameExperiment = ("Network%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
			args.network,clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
			args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
			args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)		

	# We will load or create the model
	print('Creating the model...')

	# Loading the clusters
	clusters_array = tf.cast(tf.constant(clustersDataFrame.iloc[:,1:]),dtype=tf.float32)

	# Creating the folder for the experiments. Check first is the new experiment has been created before
	args.experiment_name = "position2D3D_n"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))) and not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))):
		
		os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
		model = loadBaseModel(args)
		model = AddLayersForPositions3D_imageAnd2DarrayInput(args,model,clustersDataFrame)
		print("network %s created" % (args.network))

	# Ignore the experiment
	else:
		print("network %s ignored" % (args.network))
		sys.exit()
	# # We will load or create the experiment
	# else:

	# 	print("Folder already saved. Checking if there is a network saved...")

	# 	models = []
	# 	models_old = []

	# 	try:
	# 		models = os.listdir(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
	# 		models.sort(reverse=True)
	# 	except Exception:
	# 		models_old = os.listdir(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))
	# 		models_old.sort(reverse=True)
	# 		pass

	# 	# We check if there is, at least, one min model saved in the old folders or the actual one
	# 	if len(models) > 1:

	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	elif len(models_old) > 1:

	# 		# We move the experiment from the 'old' to the actual one in order to see it evolution
	# 		move(os.path.join(args.experiment_folder,args.experiment_name+"(old)",folderNameExperiment),
	# 			os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment))
	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	else:
	# 		print("Didn't find any model. Creating a new one...")
	# 		model = loadBaseModel(args)
	# 		model = AddLayersForPositions3DAndAngle(args,model,clustersDataFrame)
	# 		print("network %s created" % (args.network))
		

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_cluster_mse:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-08))

	# Load custom loss function
	# L1-smooth
	if args.loss_function == "L1-smooth":
		print("L1-smooth activated")
		model.compile(
			loss=L1_smooth_pose3D(clusters_array,args.variance_l1smooth),
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D' : MSE_pose3D(clusters_array)})
	# MSE
	else:
		print("MSE activated")
		model.compile(
			loss=MSE_pose3D(clusters_array),
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D' : MSE_pose3D(clusters_array)})

	# Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Clean the folder. This will leave only the three best models
	cleanExperimentFolder(args,folderNameExperiment)

# Add custom layers for the network pose+angle
def AddLayersForPositions3DAndAngle(args,model,clustersDataFrame):

	# Make sure layers are set to trainable or not
	for i in range(0,len(model.layers)):
		model.layers[i].trainable = args.freezeConvolutionalWeight

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)

	# If the network is resnet, we won't add the dropout layer
	if not args.network in ("resnet50", "resnet50v2", "resnet101", "resnet101v2", "resnet152", "resnet152v2", "resnext50", "resnext101"):
		x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting

	# Outputs model
	output1 = Dense(clustersDataFrame.shape[0], name='pose3D')(x)

	x = Dense(64, activation='relu', name='fc'+str(args.nLayers+1))(x)
	output2 = Dense(1, name='angle')(x)

	# Generate model
	model = Model(inputs=model.input, outputs=[output1, output2])

	return model

# Add custom layers for the network pose+angle+2D
def AddLayersForPositions3DAndAngle_imageAnd2DarrayInput(args,model,clustersDataFrame):

	# Make sure layers are set to trainable or not
	for i in range(0,len(model.layers)):
		model.layers[i].trainable = args.freezeConvolutionalWeight

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	# Add double input for the network (image+array)
	vector_input = Input((12,))
	x = Concatenate()([vector_input, x])

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)

	# If the network is resnet, we won't add the dropout layer
	if not args.network in ("resnet50", "resnet50v2", "resnet101", "resnet101v2", "resnet152", "resnet152v2", "resnext50", "resnext101"):
		x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting

	# Outputs model
	output1 = Dense(clustersDataFrame.shape[0], name='pose3D')(x)

	x = Dense(64, activation='relu', name='fc'+str(args.nLayers+1))(x)
	output2 = Dense(1, name='angle')(x)

	# Generate model
	model = Model(inputs=[model.input,vector_input], outputs=[output1, output2])

	return model

# Train the model that estimate pose 3D and angle
def trainModelForPositions3DAndAngle(args):

	# percentage from the original dataSet
	if args.percentageSample <= 0.0 or args.percentageSample > 1.0:
		args.percentageSample = 1.0

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, brightness_range=[0.8,1.3], rotation_range=20, shear_range=15.0, width_shift_range=10,
		height_shift_range=10)

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	# If the network we want to load is not vgg16, then, is resnet.
	if args.network != "vgg16":
		args.percentageDropout=0.0
	else:
		args.network = "vgg16"

	# Loading the images
	# Using COCO samples
	if args.COCO:
		# Using COCO train+val = train, test=val
		if not args.COCO_fullTrain:
			print('Loading images for Train')
			train_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition3D_n-Angle0.0034-cc(train+val)-rv-ws-path.csv", 
				dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			print('Loading images for Validation')
			val_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition3D_n-Angle0.1-cc(test)-rv-path.csv", 
				dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			# Loading cluster
			clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
			print("csv_clusters loaded.")

			folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
				args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
				args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
				args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)
		# Using all the dataSet of COCO for training
		else:

			# Using surreal as training
			if args.Surreal:
				print('Loading images for Train')
				train_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition3D_n-Angle0.0034-sr(DB+OP)-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition3D_n-Angle0.1-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'sr_k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")

				folderNameExperiment = ("Network%s-SR_CC-sr_k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (args.network, 
					clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)
			else:
				print('Loading images for Train')
				train_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition3D_n-Angle0.0034-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition3D_n-Angle0.1-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")

				folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
					args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)
	# Using only synthetic and real samples	
	else:
		print('Loading images for Train')
		train_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition3D_n-Angle0.0034-rv-ws-path.csv", 
			dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		print('Loading images for Validation')
		val_generator = DataGenerator(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition3D_n-Angle0.1-rv-path.csv", 
			dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		# Loading cluster
		clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
		print("csv_clusters loaded.")

		folderNameExperiment = ("Network%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
			args.network,clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
			args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
			args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)		

	# We will load or create the model
	print('Creating the model...')

	# Loading the clusters
	clusters_array = tf.cast(tf.constant(clustersDataFrame.iloc[:,1:]),dtype=tf.float32)

	# Creating the folder for the experiments. Check first is the new experiment has been created before
	args.experiment_name = "position3D_n-angle"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))) and not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))):
		
		os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
		model = loadBaseModel(args)
		model = AddLayersForPositions3DAndAngle(args,model,clustersDataFrame)
		print("network %s created" % (args.network))

	# Ignore the experiment
	else:
		print("network %s ignored" % (args.network))
		sys.exit()
	# # We will load or create the experiment
	# else:

	# 	print("Folder already saved. Checking if there is a network saved...")

	# 	models = []
	# 	models_old = []

	# 	try:
	# 		models = os.listdir(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
	# 		models.sort(reverse=True)
	# 	except Exception:
	# 		models_old = os.listdir(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))
	# 		models_old.sort(reverse=True)
	# 		pass

	# 	# We check if there is, at least, one min model saved in the old folders or the actual one
	# 	if len(models) > 1:

	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	elif len(models_old) > 1:

	# 		# We move the experiment from the 'old' to the actual one in order to see it evolution
	# 		move(os.path.join(args.experiment_folder,args.experiment_name+"(old)",folderNameExperiment),
	# 			os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment))
	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	else:
	# 		print("Didn't find any model. Creating a new one...")
	# 		model = loadBaseModel(args)
	# 		model = AddLayersForPositions3DAndAngle(args,model,clustersDataFrame)
	# 		print("network %s created" % (args.network))
		

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_pose3D_cluster_mse:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-08))

	# Load custom loss function
	# L1-smooth
	if args.loss_function == "L1-smooth":
		print("L1-smooth activated")
		model.compile(
			loss={'pose3D': L1_smooth_pose3D(clusters_array,args.variance_l1smooth), 'angle': 'mse'}, 
			loss_weights={'pose3D' : args.priority_pose3D, 'angle' : args.priority_angle}, 
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'})
	# MSE
	else:
		print("MSE activated")
		model.compile(
			loss={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'}, 
			loss_weights={'pose3D' : args.priority_pose3D, 'angle' : args.priority_angle}, 
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'})

	#Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Clean the folder. This will leave only the three best models
	cleanExperimentFolder(args,folderNameExperiment)

# Train the model that estimate pose 3D and angle, but has, in the input, an image and 2D values
def trainModelForPositions3DAndAngle_imageAnd2DarrayInput(args):

	# percentage from the original dataSet
	if args.percentageSample <= 0.0 or args.percentageSample > 1.0:
		args.percentageSample = 1.0

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, brightness_range=[0.8,1.3], rotation_range=20, shear_range=15.0, width_shift_range=10,
		height_shift_range=10)

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	# If the network we want to load is not vgg16, then, is resnet.
	if args.network != "vgg16":
		args.percentageDropout=0.0
	else:
		args.network = "vgg16"

	# Loading the images
	# Using COCO samples
	if args.COCO:
		# Using COCO train+val = train, test=val
		if not args.COCO_fullTrain:
			print('Loading images for Train')
			train_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n-Angle0.0034(openPose)-cc(train+val)-rv-ws-path.csv", 
				dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			print('Loading images for Validation')
			val_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
				comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n-Angle0.1(openPose)-cc(test)-rv-path.csv", 
				dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
			print('Done')

			# Loading cluster
			clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
			print("csv_clusters loaded.")			

			folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
				args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
				args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
				args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)
		# Using all the dataSet of COCO for training
		else:

			# Using surreal:
			if args.Surreal:
				print('Loading images for Train')
				train_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n-Angle0.0034-sr(DB+OP)-(openPose)-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n-Angle0.1(openPose)-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'sr_k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")				

				folderNameExperiment = ("Network%s-SR_CC-sr_k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (args.network,
					clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, 
					args.percentageDropout, args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)				
			else:
				print('Loading images for Train')
				train_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n-Angle0.0034(openPose)-cc(train+val+test)-rv-ws-path.csv", 
					dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				print('Loading images for Validation')
				val_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
					comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n-Angle0.1(openPose)-rv-path.csv", 
					dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
				print('Done')

				# Loading cluster
				clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
				print("csv_clusters loaded.")				

				folderNameExperiment = ("Network%s-COCOFullTrain%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
					args.network,args.COCO_fullTrain, clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
					args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
					args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)
	# Using only synthetic and real samples
	else:
		print('Loading images for Train')
		train_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path, "subSample0.0034"), nameCSV="dataSetTrainPosition2D3D_n-Angle0.0034(openPose)-rv-ws-path.csv", 
			dataGenerator=train_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		print('Loading images for Validation')
		val_generator = DataGeneratorPose3DAngle_ImageAnd2DKP(comunPathImages=os.path.join('pre_processed_images'), 
			comunPathCSV=os.path.join(args.datasets_path,"subSample0.1"), nameCSV="dataSetValPosition2D3D_n-Angle0.1(openPose)-rv-path.csv", 
			dataGenerator=test_datagenerator, batch_size=args.batch_size, shuffle=args.shuffle, dim=(224, 224), n_channels=3)
		print('Done')

		# Loading cluster
		clustersDataFrame = pd.read_csv(os.path.join(args.datasets_path,'k'+str(args.k)+'-clusters_position3D_n.csv'))
		print("csv_clusters loaded.")		

		folderNameExperiment = ("Network%s-k%d-lf%s-stdGauss%.2f-bs%d-pat%d-epo%d-wor%d-mq%d-md%.1e-lay%d-i_neu%d-pss%.3f""-lr%.1e-dro%.3f-fW%s-prp%.2f-pra%.2f-(\u03C3^2)%.2f") % (
			args.network,clustersDataFrame.shape[0], args.loss_function, args.stddev, args.batch_size, args.patience, args.epochs, args.workers, 
			args.max_queue_size, args.min_delta, args.nLayers, args.nNeurons, args.percentageSample, args.learning_rate, args.percentageDropout, 
			args.freezeConvolutionalWeight, args.priority_pose3D, args.priority_angle, args.variance_l1smooth)		

	# We will load or create the model
	print('Creating the model...')

	# Loading the clusters
	clusters_array = tf.cast(tf.constant(clustersDataFrame.iloc[:,1:]),dtype=tf.float32)

	# Creating the folder for the experiments. Check first is the new experiment has been created before
	args.experiment_name = "position2D3D_n-angle"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))) and not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))):
		
		os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
		model = loadBaseModel(args)
		model = AddLayersForPositions3DAndAngle_imageAnd2DarrayInput(args,model,clustersDataFrame)
		print("network %s created" % (args.network))

	# Ignore the experiment
	else:
		print("network %s ignored" % (args.network))
		sys.exit()
	# # We will load or create the experiment
	# else:

	# 	print("Folder already saved. Checking if there is a network saved...")

	# 	models = []
	# 	models_old = []

	# 	try:
	# 		models = os.listdir(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
	# 		models.sort(reverse=True)
	# 	except Exception:
	# 		models_old = os.listdir(os.path.join(args.experiment_folder, args.experiment_name+"(old)", folderNameExperiment))
	# 		models_old.sort(reverse=True)
	# 		pass

	# 	# We check if there is, at least, one min model saved in the old folders or the actual one
	# 	if len(models) > 1:

	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	elif len(models_old) > 1:

	# 		# We move the experiment from the 'old' to the actual one in order to see it evolution
	# 		move(os.path.join(args.experiment_folder,args.experiment_name+"(old)",folderNameExperiment),
	# 			os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment))
	# 		model_location = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[1])
	# 		model = load_model(model_location,custom_objects={ 'cluster_mse': MSE_pose3D(clusters_array), 
	# 	'smooth_l1': L1_smooth_pose3D(clusters_array,args.variance_l1smooth)})
	# 		print("network %s loaded" %(model_location))

	# 	else:
	# 		print("Didn't find any model. Creating a new one...")
	# 		model = loadBaseModel(args)
	# 		model = AddLayersForPositions3DAndAngle(args,model,clustersDataFrame)
	# 		print("network %s created" % (args.network))
		

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_pose3D_cluster_mse:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-08))

	# Load custom loss function
	# L1-smooth
	if args.loss_function == "L1-smooth":
		print("L1-smooth activated")
		model.compile(
			loss={'pose3D': L1_smooth_pose3D(clusters_array,args.variance_l1smooth), 'angle': 'mse'}, 
			loss_weights={'pose3D' : args.priority_pose3D, 'angle' : args.priority_angle}, 
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'})
	# MSE
	else:
		print("MSE activated")
		model.compile(
			loss={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'}, 
			loss_weights={'pose3D' : args.priority_pose3D, 'angle' : args.priority_angle}, 
			optimizer=adam(lr=args.learning_rate), 
			metrics={'pose3D': MSE_pose3D(clusters_array), 'angle': 'mse'})

	#Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Clean the folder. This will leave only the three best models
	cleanExperimentFolder(args,folderNameExperiment)

# When the network has finished the training, all the generated models will be erased, unless the three best models.
def cleanExperimentFolder(args,folderNameExperiment):

	models = os.listdir(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))
	models.sort(reverse=True)

	# We check if there is, at least, three min model saved
	if len(models) > 4:
		for i in range (4,len(models)):
			model_path = os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment,models[i])
			os.remove(model_path)
			print("Experiment %s removed" %(model_path))
	else:
		print("Folder %s ignored" %(os.path.join(args.experiment_folder,args.experiment_name,folderNameExperiment)))

def main():

	args = defineArgParsers()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

	if args.modelOutputType == "pose-angle":
		print("Launching pose-angle")
		if args.combine2Dkp:
			print("Launching pose-angle with double input(image+2DKP)")
			trainModelForPositions3DAndAngle_imageAnd2DarrayInput(args)
		else:
			print("Launching pose-angle")
			trainModelForPositions3DAndAngle(args)
	else:
		if args.combine2Dkp:
			print("Launching pose with double input(image+2DKP)")
			trainModelForPositions3D_imageAnd2DarrayInput(args)
		else:
			print("Launching pose")
			trainModelForPositions3D(args)

if __name__ == "__main__":
	main()

# Train the model that estimate only the angle. Not used.
def trainModelForAngle():

	# Load DataFrames for Angle
	trainDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetTrainAngle.csv'))
	testDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetTestAngle.csv'))
	valDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetValAngle.csv'))

	# percentage from the original dataSet
	if args.percentageSample < 1.0 and args.percentageSample > 0.0:
		trainDataFrame = trainDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)
		valDataFrame = valDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)
		testDataFrame = testDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, horizontal_flip=args.horizontal_flip, vertical_flip=args.vertical_flip, 
		brightness_range=[0.8,1.1])

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	print('Loading images for Train')
	train_generator = train_datagenerator.flow_from_dataframe(dataframe=trainDataFrame, directory=os.path.join('pre_processed_images','trainImages'), 
		x_col='name', y_col='angle(rad)', has_ext=False, class_mode=args.class_mode, target_size=[224,224], batch_size=args.batch_size)
	print('Done')

	print('Loading images for Validation')
	val_generator = test_datagenerator.flow_from_dataframe(dataframe=valDataFrame,directory=os.path.join('pre_processed_images','valImages'), 
		x_col='name', y_col='angle(rad)', has_ext=False, class_mode=args.class_mode, target_size=[224,224], batch_size=args.batch_size)
	print('Done')

	print('Loading images for Test')
	test_generator = test_datagenerator.flow_from_dataframe(dataframe=testDataFrame, directory=os.path.join('pre_processed_images','testImages'), 
		x_col='name', y_col='angle(rad)', has_ext=False, class_mode=args.class_mode, target_size=[224,224], batch_size=args.batch_size)
	print('Done')

	folderNameExperiment = ("bs%d-patience%d-epochs%d-workers%d-max_queue%d-min_delta%.1e-layers%d-i_neurons%d-pertentageSubSample%.3f"
		"-learning_rate%.1e") % (args.batch_size, args.patience, args.epochs, args.workers, args.max_queue_size, args.min_delta, args.nLayers, 
		args.nNeurons, args.percentageSample, args.learning_rate)

	# Creating the folder for the experiments
	args.experiment_name = "angle"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))):
			os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_loss:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))

	# Loading the VGG16 model (include_top=False will remove the dense and flatten layers)
	# model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224,3))

	# With gaussian noise
	model = customVGG16()

	# Make sure layers are set to trainable
	for layer in model.layers:
		layer.trainable = True

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)
	x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting
	x = Dense(1, name='output')(x) # For angle estimation

	model = Model(input=model.input, output=x)

	# Compile the model
	print('Compiling the model')
	model.compile(loss='mean_squared_error',optimizer=adam(lr=args.learning_rate), metrics=['mse'])

	# Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Evaluate the model using the data from test
	# evaluations = model.evaluate_generator(test_generator, steps=len(test_generator), max_queue_size=args.max_queue_size, workers=args.workers, 
	# 	use_multiprocessing=args.use_multiprocessing, verbose=args.verbose)

# Train the model that estimate pose 2D. Not used.
def trainModelForPositions():

	# Load DataFrames for Positions
	trainDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetTrainPosition.csv'))
	testDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetTestPosition.csv'))
	valDataFrame = pd.read_csv(os.path.join(args.datasets_path,'dataSetValPosition.csv'))

	# percentage from the original dataSet
	if args.percentageSample < 1.0 and args.percentageSample > 0.0:
		trainDataFrame = trainDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)
		valDataFrame = valDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)
		testDataFrame = testDataFrame.sample(frac=args.percentageSample).reset_index(drop=True)

	# Generate syntetic samples for train
	train_datagenerator = ImageDataGenerator(rescale=1./255, horizontal_flip=args.horizontal_flip, vertical_flip=args.vertical_flip, 
		brightness_range=[0.8,1.1])

	# Generate syntetic samples for test/val
	test_datagenerator = ImageDataGenerator(rescale=1./255)

	print('Loading images for Train')
	train_generator = train_datagenerator.flow_from_dataframe(dataframe=trainDataFrame, directory=os.path.join('pre_processed_images','trainImages'), 
		x_col='name', y_col=['u_left_waist','v_left_waist', 'u_left_knee','v_left_knee', 'u_left_ankle','v_left_ankle', 'u_right_waist','v_right_waist', 
		'u_right_knee','v_right_knee', 'u_right_ankle','v_right_ankle'], has_ext=False, class_mode=args.class_mode, target_size=[224,224], 
		batch_size=args.batch_size)
	print('Done')

	print('Loading images for Validation')
	val_generator = test_datagenerator.flow_from_dataframe(dataframe=valDataFrame,directory=os.path.join('pre_processed_images','valImages'), 
		x_col='name', y_col=['u_left_waist','v_left_waist', 'u_left_knee','v_left_knee', 'u_left_ankle','v_left_ankle', 'u_right_waist','v_right_waist', 
		'u_right_knee','v_right_knee', 'u_right_ankle','v_right_ankle'], has_ext=False, class_mode=args.class_mode, target_size=[224,224], 
		batch_size=args.batch_size)
	print('Done')

	print('Loading images for Test')
	test_generator = test_datagenerator.flow_from_dataframe(dataframe=testDataFrame, directory=os.path.join('pre_processed_images','testImages'), 
		x_col='name', y_col=['u_left_waist','v_left_waist', 'u_left_knee','v_left_knee', 'u_left_ankle','v_left_ankle', 'u_right_waist','v_right_waist', 
		'u_right_knee','v_right_knee', 'u_right_ankle','v_right_ankle'], has_ext=False, class_mode=args.class_mode, target_size=[224,224], 
		batch_size=args.batch_size)
	print('Done')

	folderNameExperiment = ("bs%d-patience%d-epochs%d-workers%d-max_queue%d-min_delta%.1e-layers%d-i_neurons%d-pertentageSubSample%.3f"
		"-learning_rate%.1e") % (args.batch_size, args.patience, args.epochs, args.workers, args.max_queue_size, args.min_delta, args.nLayers, 
		args.nNeurons, args.percentageSample, args.learning_rate)

	# Creating the folder for the experiments
	args.experiment_name = "position"
	if not(os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))):
			os.makedirs(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment))

	callbacks = []
	callbacks.append(EarlyStopping(monitor=args.monitor, min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, 
		mode=args.mode, baseline=args.baseline, restore_best_weights=args.restore_best_weights))
	callbacks.append(ModelCheckpoint(os.path.join(args.experiment_folder, args.experiment_name, folderNameExperiment, '{epoch:02d}-{val_loss:.4f}.hdf5'), 
		monitor=args.monitor, verbose=args.verbose, save_best_only=args.save_best_only, save_weights_only=args.save_weights_only, mode=args.mode, 
		period=args.period))
	callbacks.append(TensorBoard(log_dir=os.path.join('.',args.experiment_folder, args.experiment_name, folderNameExperiment, 'logs'), write_graph=True))

	# Loading the VGG16 model (include_top=False will remove the dense and flatten layers)
	# model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224,3))

	# With gaussian noise
	model = customVGG16()

	# Make sure layers are set to trainable
	for layer in model.layers:
		layer.trainable = True

	# We add our last layers (flatten and dense)
	x = Flatten()(model.output)

	nNeurons = args.nNeurons
	for i in range(0,args.nLayers):
		x = Dense(nNeurons, activation='relu', name='fc'+str(i+1))(x)
		nNeurons = (int)(nNeurons / 2)
	x = Dropout(args.percentageDropout)(x) # This will prevent the overfitting
	x = Dense(12, name='output')(x) # For angle estimation

	model = Model(input=model.input, output=x)

	# Compile the model
	print('Compiling the model')
	model.compile(loss='mean_squared_error',optimizer=adam(lr=args.learning_rate), metrics=['mse'])

	# Training the model
	print('Training the model')
	model.fit_generator(train_generator,epochs=args.epochs,verbose=args.verbose,steps_per_epoch=len(train_generator), 
		validation_data=val_generator,validation_steps=len(val_generator), callbacks=callbacks, use_multiprocessing=args.use_multiprocessing, 
		workers=args.workers, shuffle=args.shuffle)

	# Evaluate the model using the data from test
	# evaluations = model.evaluate_generator(test_generator, steps=len(test_generator), max_queue_size=args.max_queue_size, workers=args.workers, 
	# 	use_multiprocessing=args.use_multiprocessing, verbose=args.verbose)


# # Test the loss function
# clusters = tf.constant([[1.0,2.0,3.0,4.0,5.0],[6.0,7.0,8.0,9.0,10.0]])
# y_pred = tf.constant([[1.0,2.0],[3.0,4.0]])
# y_true = [[15.0,16.0,19.0,22.0,25.0],[27.0,34.0,41.0,48.0,55.0]]
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run([loss(y_true, y_pred)]))

# # OLD RESNET50. Not used.
# def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2, 2)):

# 	filters1, filters2, filters3 = filters

# 	bn_axis = 3

# 	conv_name_base = 'res' + str(stage) + block + '_branch'
# 	bn_name_base = 'bn' + str(stage) + block + '_branch'

# 	x = Conv2D(filters1, (1, 1), strides=strides,
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2a')(input_tensor)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters2, kernel_size, padding='same',
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2b')(x)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters3, (1, 1),
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2c')(x)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

# 	shortcut = Conv2D(filters3, (1, 1), strides=strides,
# 							 kernel_initializer='he_normal',
# 							 name=conv_name_base + '1')(input_tensor)
# 	shortcut = BatchNormalization(
# 		axis=bn_axis, name=bn_name_base + '1')(shortcut)

# 	x = add([x, shortcut])
# 	x = Activation('relu')(x)
# 	return x

# def identity_block(input_tensor, kernel_size, filters, stage, block):

# 	filters1, filters2, filters3 = filters

# 	bn_axis = 3

# 	conv_name_base = 'res' + str(stage) + block + '_branch'
# 	bn_name_base = 'bn' + str(stage) + block + '_branch'

# 	x = Conv2D(filters1, (1, 1),
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2a')(input_tensor)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters2, kernel_size,
# 					  padding='same',
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2b')(x)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters3, (1, 1),
# 					  kernel_initializer='he_normal',
# 					  name=conv_name_base + '2c')(x)
# 	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

# 	x = add([x, input_tensor])
# 	x = Activation('relu')(x)
# 	return x

# def customResNet50(args):

# 	# Create the custom Resnet50
# 	input_shape = _obtain_input_shape((224, 224,3),default_size=224,min_size=32,data_format=k.image_data_format(),require_flatten=False,
# 		weights='imagenet')
# 	img_input = Input(shape=input_shape)

# 	Add gaussian layer
# 	x = GaussianNoise(args.stddev)(img_input)

# 	bn_axis = 3

# 	x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
# 	x = Conv2D(64, (7, 7),strides=(2, 2),padding='valid',kernel_initializer='he_normal',name='conv1')(x)
# 	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
# 	x = Activation('relu')(x)
# 	x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
# 	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# 	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
# 	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
# 	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

# 	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

# 	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

# 	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
# 	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
# 	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

# 	# Create model.
# 	inputs = img_input
# 	model = Model(inputs, x, name='resnet50')

# 	# Load weights.
# 	weights_path = os.path.join("h5_files","resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
# 	model.load_weights(weights_path, by_name=True)

# 	return model