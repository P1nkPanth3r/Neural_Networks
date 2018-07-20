# ----------------------------------------------------------------------------------------#
# Program Purpose/Design
# ----------------------------------------------------------------------------------------#
# This program is designed to estimate a LSTM Recurrent Neural Network using Keras with a
# TensorFlow Backend. This program was developed using Python 3.5.2.

# ----------------------------------------------------------------------------------------#
# Recommendations for Program Improvement
# ----------------------------------------------------------------------------------------#
# Place recommendations for program improvement here.
# ----------------------------------------------------------------------------------------#
import os, time, warnings
import numpy as np
import tensorflow as tf
import keras
from numpy import newaxis
from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from tensorboard import main as tb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide Numpy warnings
# ----------------------------------------------------------------------------------------#
# FUNCTION DEFINITIONS
# ----------------------------------------------------------------------------------------#
def fnc_load_data(filepath, window_length):
    file = open(filepath, 'rb').read()
    decoded_file = file.decode().split('\n')[:-1] # Read in file except trailing blank at end of CSV file

    window_length += 1
    data = []
    for i in range(len(decoded_file) - window_length):
        data.append(decoded_file[i:(i+window_length)])

    normalized_data, list_y_all = fnc_normalise_windows(data)
    y_all = np.empty(shape=(len(list_y_all),1))

    for j in range(len(list_y_all)):
        y_all[j] = list_y_all[j]

    train_percent = 0.7

    rounded_train_percent = round(round(train_percent * normalized_data.shape[0])/window_length)*window_length

    np.random.seed(123654)
    train = normalized_data[:int(rounded_train_percent), :, :]
    np.random.shuffle(train)

    x_train = train[:,:,:-1]
    y_train = train[:,:,-1][:,-1]
    x_test = normalized_data[int(rounded_train_percent):,:,:-1]
    y_test = normalized_data[int(rounded_train_percent):,:,-1][:,-1]
    y_all = y_all[int(train_percent*len(y_all)):]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return [x_train, y_train, x_test, y_test, y_all, rounded_train_percent]

def fnc_normalise_windows(window_data):
    list_y_all = []
    length_of_window = len(window_data[0])
    count_of_windows = len(window_data)
    count_of_features_and_target = len(window_data[0][0].split(','))
    normalized_data = np.empty(shape=(count_of_windows,length_of_window,count_of_features_and_target))
    for window in range(count_of_windows):
        first_window_value = window_data[window][0].split(',')[-1]
        list_y_all.append(first_window_value)
        split_row = [""]*len(window_data[0][0].split(','))
        for row in range(length_of_window):
            for col in range(count_of_features_and_target):
                split_row[col] = float(window_data[window][row].split(',')[col])
                normalized_data[window][row][col] = split_row[col]
                # if col < count_of_features_and_target - 1: # Normalize features
                #     normalized_data[window][row][col] = (float(normalized_data[window][row][col]) / float(first_window_value)) - 1
                # if col == count_of_features_and_target - 1: # Normalize target
                #     normalized_data[window][row][-1] = (float(normalized_data[window][row][-1]) / float(first_window_value)) - 1
            if classification_model:
                if normalized_data[window][row][-1] >= 0:
                    normalized_data[window][row][-1] = 1
                else:
                    normalized_data[window][row][-1] = 0
    return [normalized_data, list_y_all]

def fnc_build_model(layers,dropout_percent):
    model = Sequential()
    if classification_model:
        model.add(LSTM(input_shape=layers[0],output_dim=layers[1],return_sequences=True))
        model.add(Activation(K.relu))

        model.add(LSTM(output_dim=layers[2],return_sequences=False))
        model.add(Activation(K.relu))

        model.add(Dense(output_dim=layers[3]))
        model.add(Activation('softmax'))

        start = time.time()
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else: # Regression Model
        model.add(LSTM(input_shape=layers[0], output_dim=layers[1], return_sequences=True))
        model.add(Dropout(dropout_percent))
        model.add(Activation(K.relu))

        model.add(LSTM(output_dim=layers[2], return_sequences=False))
        model.add(Dropout(dropout_percent))
        model.add(Activation(K.relu))

        model.add(Dense(output_dim=layers[3]))
        model.add(Activation('linear'))

        start = time.time()
        model.compile(loss='mse', optimizer='rmsprop')
    print("> Compilation Time : ", time.time() - start)
    return model

def generate_predictions(model, x_test):
    predictions = model.predict(x=x_test)
    predictions = np.reshape(predictions, (predictions.size,))
    return predictions
# ----------------------------------------------------------------------------------------#
# MAIN PROGRAM
if __name__=='__main__':
    start_time = time.time()
    epochs = 50
    window_length = 60
    layer_1 = 100
    layer_2 = 100
    validation_percent = 0.3
    batch_size = 3000
    classification_model = False

    print('> Loading data... ')

    filepath = r'..\project_data\data.csv'
    x_train, y_train, x_test, y_test, y_all, rounded_train_percent = fnc_load_data(filepath, window_length)

    print('> Data Loaded. Compiling...')
    # (len(x_train[0]), 1)
    # model = fnc_build_model([(len(x_train[0][0]), 1), 50, 100, 1])
    model = fnc_build_model([(len(x_train[0]),len(x_train[0][0])), layer_1, layer_2, 1],0.2)

    # Add TensorBoard callback
    os.chdir('..\project_data\\')
    tbCallback = keras.callbacks.TensorBoard(log_dir='./LSTM_TF_log', histogram_freq=0, batch_size=32, write_graph=True,
                                             write_grads=False, write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None)

    model_history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        nb_epoch=epochs,
        verbose=2,
        validation_split=validation_percent,
        callbacks=[tbCallback])

    predictions = generate_predictions(model, x_test)

    prediction_losses = []
    for i in range(len(predictions)):
        if classification_model:
            predictions[i] = round(predictions[i])
        loss = abs((predictions[i]-y_test[i])/y_test[i])
        prediction_losses.append(loss)
    predictions_vs_actuals = [predictions, y_test, prediction_losses]

    print('Training took {} seconds. \n Train set size: {} \n Test set size: {}'.format(time.time() - start_time,len(x_train),len(x_test)))

    outfile_path = r'..\project_data\predictions.csv'
    np.savetxt(outfile_path,np.transpose(predictions_vs_actuals),delimiter=',')
    print('Output saved to {}'.format(outfile_path))

# Explicitly reap session to avoid an AttributeError sometimes thrown by TensorFlow on shutdown. See:
# https://github.com/tensorflow/tensorflow/issues/3388
import gc; gc.collect()

print('Program Complete')

# Start TensorBoard session
tf.flags.FLAGS.logdir = './LSTM_TF_log'
tb.main()
