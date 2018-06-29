import numpy as np
import numpy.matlib as mat
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM

import path_generator_keras


class ProcessData:
    def __init__(self, filename):
        self.load_path = filename
        self.trainData = np.load(self.load_path, encoding='latin1').tolist()
        self.num_dim = self.trainData[0].shape[1]
        self.TrainX = None
        self.TrainY = None

    def formatData(self, print_shapes):
        """remove all NoneTypes from training set"""
        self.trainData = [x for x in self.trainData if x is not None]
        p = 0
        for _ in range(0,  len(self.trainData)): p += len(self.trainData[_])
        reformated_trainData = np.zeros((p, self.num_dim))
        goals_trainData = np.zeros((p, self.num_dim))
        count = 0
        for i in tqdm(range(1, len(self.trainData))):
            target_length = len(self.trainData[i])
            reformated_trainData[count:count + target_length] = self.trainData[i]
            goals_trainData[count:count + target_length] = \
                mat.repmat(self.trainData[i][-1], target_length, 1)
            count += target_length
        TrainX = np.concatenate((reformated_trainData, goals_trainData), axis=1)
        TrainY = np.roll(reformated_trainData, -1, axis=0)
        self.TrainY = np.expand_dims(TrainY, axis=1)
        self.TrainX = np.expand_dims(TrainX, axis=1)
        if print_shapes:
            print('TrainX shape: ', self.TrainX.shape)
            print('TrainY shape: ', self.TrainY.shape)
        return self.TrainX, self.TrainY

    def sampleBatches(self, batch_size):
        offset = random.randint(0, self.TrainY.shape[1]-batch_size)
        idx = np.arange(batch_size)
        return self.TrainX[:, idx + offset, :], self.TrainY[:, idx + offset, :]


class SimpleRNN:
    def __new__(self, hidden, inp_dim, op_dim, stacked_lstm_layers):
        self.model = Sequential()
        self.model.add(LSTM(output_dim=hidden,
                            init='glorot_uniform', inner_init='orthogonal', activation='relu',
                            W_regularizer=None, U_regularizer=None, b_regularizer=None,
                            dropout_W=0, dropout_U=0, return_sequences=True,
                            input_shape=(None, inp_dim)))
        for j in range(stacked_lstm_layers - 1):
            self.model.add(LSTM(hid, return_sequences=True))

        self.model.add(Dense(output_dim=op_dim))
        self.model.add(Activation('relu'))

        self.model.compile(loss='mse',
                           optimizer='adadelta',
                           metrics=['accuracy'])
        return self.model



if __name__ == '__main__':

    obstacle_path = 'random_squares_1.npy'
    data_filename = 'training_data_5k_r_sq_1.npy'
    cx = cy = 100
    trainingData = ProcessData(filename=data_filename)
    train_X, train_Y = trainingData.formatData(print_shapes=True)

    # ------------- HYPER-PARMAMETER DEFINITIONS! ---------------------------------------

    inp_dim = train_X.shape[-1]
    hid = 200
    op_dim = train_Y.shape[-1]
    n_iters = 10000  # iterations per epoch
    batch_size = 100
    stacked_hidden_layers = 2

    # ------------- MODEL CONSTRUCTION! ---------------------------------------

    model = SimpleRNN(hidden=hid, inp_dim=inp_dim, op_dim=op_dim, stacked_lstm_layers=stacked_hidden_layers)
    model.summary()

    ## ------------- TRAINING TIME! ----------------------------------------
    n_iters = 10000
    success_tracker = []
    model_name = 'model_keras_oraclenet_test.h5'
    t1 = time.time()
    for k in range(n_iters):
        model.fit(train_X, train_Y, batch_size=100, verbose=False,
                  epochs=10, validation_split=0.1, shuffle=True)
        print(model_name + ' Epoch set:', k, ' for ', model_name)
        if k%100==0:
            model.save(model_name)
            _, validity = path_generator_keras.main(cx, cy, obstacle_path, model,
                                 num_evals=100, eval_mode=True, plotopt=False)
            success_tracker.append([k, validity])
    t2 = time.time()
    print('time taken to train: ', t2 - t1)
    print('Training complete!')
    
    ##---------------------------------------------------------------------

    plt.figure()
    success_tracker = np.asarray(success_tracker)
    plt.plot(success_tracker[:, 0], success_tracker[:, 1])
    plt.title('Success rate trends')
    plt.xlabel('Training Iterations')
    plt.ylabel('Success Rate over 100 trials')