#!/usr/bin/env python
# Script to train and test a neural network with TF's Keras API for face detection

import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev

def get_smaller_image(image_array):

    new = []
    for i in range(image_array.shape[0]):
        #print(m[i].shape)
        large_image = image_array[i]
        input_size = 480
        output_size = 48
        bin_size = input_size // output_size
        small_image = large_image.reshape((output_size, bin_size, 
                                           output_size, bin_size, 3)).max(3).max(1)
        new.append(small_image)
        
    new = np.asarray(new)
    return new

def load_data_from_path(file_path):
    """
    Load data from npz file
    :param file_path: path to npz file with training data
    :return: input features and target data as numpy arrays
    """
    os.chdir(file_path)
    images = np.load("images.npy")
    target= np.load("labels.npy")
    landmarks = np.load("landmarks.npy")

    train = get_smaller_image(images)

    return train, target, landmarks


def normalize_data_per_row(data):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :return: normalized data with pixel values in [0,1]
    """

    # sanity checks!
    assert len(data.shape) == 4, "Expected the input data to be a 4D matrix"

    mean, stdev = compute_normalization_parameters(data)

    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    #normalized_data = data.astype(int) / int(255)
    normalized_data = data / 255.0

    return normalized_data



# def compute_average_L2_error(test_target, predicted_targets):
#     """
#     Compute the average L2 error for the predictions
#     :param test_target: matrix with ground truth targets [N x 1]
#     :param predicted_targets: matrix with predicted targets [N x 1]
#     :return: average L2 error
#     """
#     diff = predicted_targets - test_target
#     l2_err = np.sqrt(np.sum(np.power(diff, 2), axis=1))
#     assert l2_err.shape[0] == predicted_targets.shape[0], \
#         "Invalid dim {} vs {}".format(l2_err.shape, predicted_targets.shape)
#     average_l2_err = np.mean(l2_err)

#     return average_l2_err

def main(batch_size, epochs, lr, logs_dir):
    """
    Main function that performs training and test on a validation set

    :param npz_data_file: npz input file with training data
    :param batch_size: batch size to use at training time
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param val: percentage of the training data to use as validation
    :param logs_dir: directory where to save logs and trained parameters/weights
    """
    #TRAIN_PATH = "/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/ears_features/train"
    #TEST_PATH = "/Users/Kaitlynn/Desktop/CPSC_459/bim2020_ears/ears_features/test"

    TRAIN_PATH = "/home/ubuntu/ears/bim2020_ears/aws_scripts/ears_features/train"
    TEST_PATH = "/home/ubuntu/ears/bim2020_ears/aws_scripts/ears_features/test"

    input_train, target_train, input_landmarks = load_data_from_path(TRAIN_PATH)
    input_test, target_test, target_landmarks = load_data_from_path(TEST_PATH)

    #input features and target data as numpy arrays
    N = input.shape[0]
    assert N == target.shape[0], \
        "The input and target arrays had different amounts of data ({} vs {})".format(N, target.shape[0]) # sanity check!
    print("Loaded {} training examples.".format(N))

   
    normalized_data = normalize_data_per_row(input_train)
    mean, stdev = compute_normalization_parameters(input_train)
    # build the model
    model = build_fn(input_train.shape[1:])

    # train the model
    print("\n\nTRAINING...")
    train_model(model, input_train, target_train, input_test, target_test, mean, stdev,
                epochs=epochs, learning_rate=lr, batch_size=batch_size)

    print("saving model...")
    model_name = 'ears_model_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    model_name = model_name + ".hdf5"
    print(model_name)
    model.save(model_name)

    # test the model
    print("\n\nTESTING...")
    predicted_targets = test_model(model, input_test, mean, stdev)

    # Report average L2 error
    l2_err = compute_average_L2_error(target_test, predicted_targets)
    print("L2 Error on Testing Set: {}".format(l2_err))

    # visualize the result
    # plt.plot(test_input, test_target, predicted_targets, title="Predictions")


def build_fn(num_inputs, landmarks):
    

    #print(num_inputs)
    # number of images, x, y, 
    input1 = tf.keras.layers.Input(shape=(num_inputs[0],num_inputs[1], num_inputs[2],), name="inputs1")
    input2 = tf.keras.layers.Input(shape=(landmarks[0],landmarks[1],), name="inputs2")
    # pick the first few layers, and last softmax  5 layers and be okay

    # TRAIN THE PIXELS SEPARATELY
    hidden0 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3))(input1)
    hidden1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(hidden0) 
    hidden2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(hidden1)
    hidden3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(hidden2)
    hidden4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3))(hidden3)
    hidden5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(hidden4)
    hidden6 = tf.keras.layers.Flatten()(hidden5)
    

    #output = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(hidden6)

    # TRAIN THE LANDMARKS SEPARATELY
    hidden10 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3))(input2)
    hidden11 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(hidden10) 
    hidden12 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3))(hidden11)
    hidden13 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(hidden12)
    hidden14 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3))(hidden13)
    hidden15 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(hidden14)
    hidden16 = tf.keras.layers.Flatten()(hidden5)
    #output = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(hidden16)

    merged = Concatenate([hidden6, hidden16])
    output = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(merged)

    #print(output.shape)
    #output2 = tf.keras.layers.Reshape((2207, 1), input_shape=(None,1164))(output)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="kait_model")
    return model

def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # normalize
    norm_train_input = normalize_data_per_row(train_input)
    norm_val_input = normalize_data_per_row(val_input)
    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])

    # TODO - Create callbacks for saving checkpoints and visualizing loss on TensorBoard


    # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_ears_weights.h5'),
                                                            monitor='binary_accuracy',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)

    # do training for the specified number of epochs and with the given batch size
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
              validation_data=(norm_val_input, val_target),
              callbacks=[tbCallBack, checkpointCallBack]) # add this extra parameter to the fit function


def test_model(model, test_input, input_mean, input_stdev, batch_size=60):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # normalize
    #norm_test_input = normalize_data_per_row(test_input)

    # evaluate
    predicted_targets = model.predict(test_input, batch_size=batch_size)

    return predicted_targets


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=50)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=100)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=1e-3)
    # parser.add_argument("--val", help="percent of training data to use for validation",
    #                     type=float, default=0.8)
    # parser.add_argument("--input", help="input file (npz format)",
    #                     type=str, required=True)
    parser.add_argument("--logs_dir", help="logs directory",
                        type=str, default="")
    args = parser.parse_args()

    if len(args.logs_dir) == 0: # parameter was not specified
        args.logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    
    #logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    # if not os.path.isdir(logs_dir):
    #     os.makedirs(logs_dir)

    # main(256, 100, 1e-4, 0.8, logs_dir)

    main(args.batch_size, args.epochs, args.lr, args.logs_dir)

    sys.exit(0)
