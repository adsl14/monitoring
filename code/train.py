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
#tf.random.set_seed(seed)
import keras.backend as k
sess = tf.compat.v1.get_default_session()
k.set_session(sess)

import keras
from keras.optimizers import adam
from keras.models import Model, Sequential
from keras.layers import Dropout, Input, Dense, LSTM, Conv1D, CuDNNLSTM, BatchNormalization
from keras.layers import Concatenate, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing import sequence

def add_Dense_Layer(number, data):

    data = Dense(number, activation='relu',
                 kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(data)  
    return data
def add_LSTM_Layer(number,return_sequence,data):

    data = LSTM(number, activation='relu', return_sequences=return_sequence,
                kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=seed))(data) 
    return data
def add_CuDNNLSTM_Layer(number,return_sequence,data):

    data = CuDNNLSTM(number, return_sequences=return_sequence,
                kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=seed))(data) 
    return data
def add_Conv1D_Layer(number,data):

      data = Conv1D(filters=number, kernel_size=1,  data_format='channels_last', activation='relu', 
                 kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
                 bias_initializer=keras.initializers.glorot_uniform(seed=seed))(data)
      
      return data


def loadData(args):

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
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, 
    random_state=seed, stratify=y_train) # 80% train, 20% val 

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Number of features
    num_features_input = x_train.shape[1]

    return x_train, y_train, x_test, y_test, num_features_input, num_classes
def TrainNN(args, x_train, y_train, x_test, y_test, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    nLayers = len(args.nNeurons)

    # Experiment folder and name
    nameModel = 'NN-lr%.1e-bs%d-drop%.2f-hla%d-hne%s-epo%d' % (args.learning_rate,args.batch_size,
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
    input = Input(shape=(num_features_input,))

    x = Dense(args.nNeurons[0], activation=tf.nn.relu, input_dim=num_features_input,
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(input)

    # Check if the user has, in the input, more than one hidden layer
    if nLayers > 1:
        for i in range(1,nLayers):
            x=Dense(args.nNeurons[i], activation=tf.nn.relu,
                kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)

    # Check if the user wants to use a dropout layer
    if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)

    output=Dense(num_classes, activation='softmax', 
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)

    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)


def loadSequenceExample(args):

    # Example motion sensor
    # Labels name
    labels = args.labels.split(',')
    num_classes = len(labels)
    time_step = 60
    path = 'dataset/MovementAAL_RSS_'

    sequences = list()
    for i in range(1,315):
        file_path = path + str(i) + '.csv'
        #print(file_path)
        df = pd.read_csv(file_path, header=0)
        values = df.values
        sequences.append(values)

    targets = pd.read_csv('dataset/MovementAAL_target.csv')
    targets = targets.values[:,1]
    targets = (targets+1)/2

    groups = pd.read_csv('groups/MovementAAL_DatasetGroup.csv', header=0)
    groups = groups.values[:,1]

    #Padding the sequence with the values in last row to max length
    to_pad = 129
    new_seq = []

    for one_seq in sequences:
        len_one_seq = len(one_seq)
        last_val = one_seq[-1]
        n = to_pad - len_one_seq

        to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
        new_one_seq = np.concatenate([one_seq, to_concat])
        new_seq.append(new_one_seq)
    final_seq = np.stack(new_seq)

    #truncate the sequence to length 60
    final_seq=sequence.pad_sequences(final_seq, maxlen=time_step, padding='post', dtype='float',
                                     truncating='post')

    # Get the x_test and y_test. It will shuffle automatically
    x_train, x_test, y_train, y_test = train_test_split(final_seq, targets, test_size=0.3,random_state=seed, stratify=targets) # 70% train, 30% test

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    # Number of features
    num_features_input = x_train.shape[2]

    return x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes
def loadSequence(args):

    # Labels name
    labels = args.labels.split(',')
    num_classes = len(labels)
    num_features_input = 1
    time_step = 65 # Number of data for each time series. This is fixed

    # Names for output files.
    datasetNamePOL0 = 'POL-0_80.csv'
    datasetNamePOL1 = 'POL-81_161.csv'

    # Get the csv tables from Google drive
    datasetPathPOL0 = os.path.join('radar',datasetNamePOL0)
    datasetPathPOL1 = os.path.join('radar',datasetNamePOL1)

    # Get the dataframes
    datasetDataFramePOL0 = pd.read_csv(datasetPathPOL0)
    datasetDataFramePOL1 = pd.read_csv(datasetPathPOL1)

    # Clean data
    datasetDataFramePOL0 = datasetDataFramePOL0.drop('system:time_start', axis=1)
    datasetDataFramePOL1 = datasetDataFramePOL1.drop('system:time_start', axis=1)

    # Concat datasets (POL)
    datasetDataFramePOL = pd.concat([datasetDataFramePOL0,datasetDataFramePOL1], axis=1)

    # Get the sequence for each area
    num_areas = datasetDataFramePOL.shape[1]
    sequences = []
    for j in range(0,num_areas):
      values = pd.concat([datasetDataFramePOL.iloc[:,j]],axis=1, join='inner').values
      sequences.append(values)

    # Get the length of the time series for each area in order to get the max values of the time series
    len_sequences = []
    for one_seq in sequences:
        len_sequences.append(len(one_seq))
    seriesDescription = pd.Series(len_sequences).describe()
    max_valueSerie = int(seriesDescription[-1])
    min_valueSerie = int(seriesDescription[3])

    #Padding the sequence with the values in last row to max length.
    # If the sequence with most data values is greater than the time_step, we add to the last sequence of each time series the last row
    if max_valueSerie != min_valueSerie and max_valueSerie > time_step:
      new_seq = []
      for one_seq in sequences:
          len_one_seq = len(one_seq)
          n = max_valueSerie - len_one_seq

          to_concat = np.repeat(one_seq[-1], n).reshape(num_features_input, n).transpose()
          new_one_seq = np.concatenate([one_seq, to_concat])
          new_seq.append(new_one_seq)
      sequences = np.stack(new_seq)
    # If the time_step value is greater, then we add values from the last row of each time serie until we reach the number of time_step
    else:
      new_seq = []
      for one_seq in sequences:
          len_one_seq = len(one_seq)
          n = time_step - len_one_seq

          to_concat = np.repeat(one_seq[-1], n).reshape(num_features_input, n).transpose()
          new_one_seq = np.concatenate([one_seq, to_concat])
          new_seq.append(new_one_seq)
      sequences = np.stack(new_seq)


    #truncate the sequence to length time_step
    sequences=sequence.pad_sequences(sequences, maxlen=time_step, padding='post', 
                                     dtype='float', truncating='post')
    # Get the targets (now it's random)
    targets = np.random.randint(num_classes, size=num_areas)

    # Get the x_test and y_test. It will shuffle automatically
    x_train, x_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.3,random_state=seed, stratify=targets) # 70% train, 30% test                                 

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes
def TrainLSTM(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    args.nNeuronsSequence = list(map(int, args.nNeuronsSequence.split(',')))
    nLayers = len(args.nNeurons)
    nLayersSequence = len(args.nNeuronsSequence)

    # Experiment folder and name
    nameModel = 'LSTM-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d' % (args.learning_rate,args.batch_size,
        args.percentageDropout,str(args.nNeuronsSequence),str(args.nNeurons),args.epochs)
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
    input = Input(shape=(time_step,num_features_input,))

    # ADD Recurrent layer
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      # The user has entered two hidden layers
      if nLayersSequence > 1:
        x = add_LSTM_Layer(args.nNeuronsSequence[0], True, input)

        for i in range(1,nLayersSequence-1):
            x = add_LSTM_Layer(args.nNeuronsSequence[0], True, x)

        x = add_LSTM_Layer(args.nNeuronsSequence[-1], False, x)

      else:
        x = add_LSTM_Layer(args.nNeuronsSequence[0], False, input)

      if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)

    else:
        print("Please, insert at least one recurrent layer.")
        sys.exit()

    # ADD dense layer
    if nLayers > 0:
        for i in range(0,nLayers):
            x = add_Dense_Layer(args.nNeurons[i], x)

        # Add dropout before the output layer
        if args.percentageDropout > 0.0:
            x = Dropout(args.percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)
def TrainLSTM_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    args.nNeuronsSequence = list(map(int, args.nNeuronsSequence.split(',')))
    args.nNeuronsConv1D = list(map(int, args.nNeuronsConv1D.split(',')))
    nLayers = len(args.nNeurons)
    nLayersSequence = len(args.nNeuronsSequence)
    nLayersConv1D = len(args.nNeuronsConv1D)

    # Experiment folder and name
    nameModel = 'LSTM_FCN-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d' % (args.learning_rate,args.batch_size,
        args.percentageDropout,str(args.nNeuronsSequence),str(args.nNeurons),args.epochs)
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
    input = Input(shape=(time_step,num_features_input,))

    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_LSTM_Layer(args.nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_LSTM_Layer(args.nNeuronsSequence[i], True, x)

      if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)          
    else:
      print("Please, insert at least one recurrent layer.")
      sys.exit()

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(args.nNeuronsConv1D[0], x)
        x = BatchNormalization()(x)

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(args.nNeuronsConv1D[i], x)
          x = BatchNormalization()(x)

        # Apply global average pooling and make the output only one dimension
        x = GlobalAveragePooling1D()(x)          

        if args.percentageDropout > 0.0:
          x = Dropout(args.percentageDropout)(x)
  
    else:
      print("Please, insert at least one conv1D layer.")
      sys.exit()

    # ADD dense layer
    if nLayers > 0:
        for i in range(0,nLayers):
            x = add_Dense_Layer(args.nNeurons[i], x)

        # Add dropout before the output layer
        if args.percentageDropout > 0.0:
            x = Dropout(args.percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)
def TrainLSTM_parallel_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    args.nNeuronsSequence = list(map(int, args.nNeuronsSequence.split(',')))
    args.nNeuronsConv1D = list(map(int, args.nNeuronsConv1D.split(',')))
    nLayers = len(args.nNeurons)
    nLayersSequence = len(args.nNeuronsSequence)
    nLayersConv1D = len(args.nNeuronsConv1D)

    # Experiment folder and name
    nameModel = 'LSTM_parallel_FCN-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d' % (args.learning_rate,args.batch_size,
        args.percentageDropout,str(args.nNeuronsSequence),str(args.nNeurons),args.epochs)
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
    input = Input(shape=(time_step,num_features_input,))

    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_LSTM_Layer(args.nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_LSTM_Layer(args.nNeuronsSequence[i], True, x)

      if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)

    else:
      print("Please, insert at least one recurrent layer.")
      sys.exit() 

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x_2 = add_Conv1D_Layer(args.nNeuronsConv1D[0], input)
        x_2 = BatchNormalization()(x_2)

        for i in range(1,nLayersConv1D):
          x_2 = add_Conv1D_Layer(args.nNeuronsConv1D[i], x_2)
          x_2 = BatchNormalization()(x_2)

        x_2 = GlobalAveragePooling1D()(x_2)

        if args.percentageDropout > 0.0:
            x_2 = Dropout(args.percentageDropout)(x_2)

    else:
      print("Please, insert at least one conv1D layer.")
      sys.exit()

    #--------------
    # CONCATENATE LSTM + Conv1D
    #--------------    
    x = Concatenate()([x,x_2])

    # ADD dense layer
    if nLayers > 0:
        for i in range(0,nLayers):
            x = add_Dense_Layer(args.nNeurons[i], x)

        # Add dropout before the output layer
        if args.percentageDropout > 0.0:
            x = Dropout(args.percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)
def TrainCuDNNLSTM(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    args.nNeuronsSequence = list(map(int, args.nNeuronsSequence.split(',')))
    nLayers = len(args.nNeurons)
    nLayersSequence = len(args.nNeuronsSequence)

    # Experiment folder and name
    nameModel = 'CuDNNLSTM-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d' % (args.learning_rate,args.batch_size,
        args.percentageDropout,str(args.nNeuronsSequence),str(args.nNeurons),args.epochs)
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
    input = Input(shape=(time_step,num_features_input,))

    # ADD Recurrent layer
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      # The user has entered two hidden layers
      if nLayersSequence > 1:
        x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[0], True, input)

        for i in range(1,nLayersSequence-1):
            x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[0], True, x)

        x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[-1], False, x)

      else:
        x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[0], False, input)

      if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)

    else:
        print("Please, insert at least one recurrent layer.")
        sys.exit()

    # ADD dense layer
    if nLayers > 0:
        for i in range(0,nLayers):
            x = add_Dense_Layer(args.nNeurons[i], x)

        # Add dropout before the output layer
        if args.percentageDropout > 0.0:
            x = Dropout(args.percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)
def TrainCuDNNLSTM_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes):

    # Convert string list to int list
    args.nNeurons = list(map(int, args.nNeurons.split(',')))
    args.nNeuronsSequence = list(map(int, args.nNeuronsSequence.split(',')))
    args.nNeuronsConv1D = list(map(int, args.nNeuronsConv1D.split(',')))
    nLayers = len(args.nNeurons)
    nLayersSequence = len(args.nNeuronsSequence)
    nLayersConv1D = len(args.nNeuronsConv1D)

    # Experiment folder and name
    nameModel = 'CuDNNLSTM_FCN-lr%.1e-bs%d-drop%.2f-hnes%s-hne%s-epo%d' % (args.learning_rate,args.batch_size,
        args.percentageDropout,str(args.nNeuronsSequence),str(args.nNeurons),args.epochs)
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
    input = Input(shape=(time_step,num_features_input,))

    #--------------
    # LSTM block
    #--------------
    # Check if the user has entered at least one hidden layer sequence
    if nLayersSequence > 0:
      x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[0], True, input)

      for i in range(1,nLayersSequence):
        x = add_CuDNNLSTM_Layer(args.nNeuronsSequence[i], True, x)

      if args.percentageDropout > 0.0:
        x = Dropout(args.percentageDropout)(x)          
    else:
      print("Please, insert at least one recurrent layer.")
      sys.exit()

    #--------------
    # CONV1D block
    #--------------
    # Check if the user has entered at least one hidden layer conv1D
    if nLayersConv1D > 0:
        x = add_Conv1D_Layer(args.nNeuronsConv1D[0], x)
        x = BatchNormalization()(x)

        for i in range(1,nLayersConv1D):
          x = add_Conv1D_Layer(args.nNeuronsConv1D[i], x)
          x = BatchNormalization()(x)

        # Apply global average pooling and make the output only one dimension
        x = GlobalAveragePooling1D()(x) 

        if args.percentageDropout > 0.0:
          x = Dropout(args.percentageDropout)(x) 
    else:
      print("Please, insert at least one conv1D layer.")
      sys.exit()

    # ADD dense layer
    if nLayers > 0:
        for i in range(0,nLayers):
            x = add_Dense_Layer(args.nNeurons[i], x)

        # Add dropout before the output layer
        if args.percentageDropout > 0.0:
            x = Dropout(args.percentageDropout)(x)

    # Output
    output = Dense(num_classes, activation='softmax',
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(x)
    
    model = Model(input,output)

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
        validation_data=(x_test,y_test),
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        shuffle=args.shuffle,
        callbacks=callbacks,
        verbose=1)

    cleanExperimentFolder(experimentPath)

def main():

    args = defineArgParsersTrain()

    if args.typeNetwork == 'NN':

        # Load the dataset
        x_train, y_train, x_test, y_test, num_features_input, num_classes = loadData(args)

        # Train normal neural network
        TrainNN(args,x_train, y_train, x_test, y_test, num_features_input, num_classes)

    else:

        # Load the dataset (sequence)
        x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes = loadSequenceExample(args)
        #x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes = loadSequence(args)

        # Train RNN
        if args.typeNetwork == 'LSTM':
            TrainLSTM(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes)
        elif args.typeNetwork == 'LSTM_FCN':
            TrainLSTM_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes)
        elif args.typeNetwork == 'LSTM_parallel_FCN':
            TrainLSTM_parallel_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes)
        elif args.typeNetwork == 'CuDNNLSTM':
            TrainCuDNNLSTM(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes)
        elif args.typeNetwork == 'CuDNNLSTM_FCN':
            TrainCuDNNLSTM_FCN(args, x_train, y_train, x_test, y_test, time_step, num_features_input, num_classes)

if __name__ == "__main__":
    main()
