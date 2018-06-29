import numpy as np
import numpy.matlib as mat
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from path_generator import main as model_eval

# make variable types for automatic setting to GPU or CPU, depending on GPU availability
def set_torch_types():
    use_cuda = torch.cuda.is_available()
    type_dict = {
        'FloatTensor': torch.cuda.FloatTensor if use_cuda else torch.FloatTensor,
        'LongTensor' : torch.cuda.LongTensor if use_cuda else torch.LongTensor,
        'ByteTensor' : torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    }
    return type_dict


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
        self.TrainY = np.expand_dims(TrainY, axis=0)
        self.TrainX = np.expand_dims(TrainX, axis=0)
        if print_shapes:
            print('TrainX shape: ', self.TrainX.shape)
            print('TrainY shape: ', self.TrainY.shape)
        return self.TrainX, self.TrainY

    def sampleBatches(self, batch_size):
        offset = random.randint(0, self.TrainY.shape[1]-batch_size)
        idx = np.arange(batch_size)
        return self.TrainX[:, idx + offset, :], self.TrainY[:, idx + offset, :]


class SimpleRNN(nn.Module):
    def __init__(self, inp_dim, hidden_size, op_dim, stacked_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.op_dim = op_dim
        self.inp_dim = inp_dim

        self.inp = nn.Linear(inp_dim, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, stacked_layers, dropout=0.0)
        self.out = nn.Linear(hidden_size, op_dim)

    def step(self, input, hidden=None):
        input = self.inp(input.view(-1, self.inp_dim)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if 'FloatTensor' not in locals(): FloatTensor = set_torch_types()['FloatTensor']
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, list(inputs.size())[1], self.op_dim)).type(FloatTensor)
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden

if __name__ == '__main__':
    types = set_torch_types()
    FloatTensor = types['FloatTensor']

    obstacle_path = 'random_squares_1.npy'
    data_filename = 'training_data_5k_r_sq_1.npy'
    cx = cy = 100
    trainingData = ProcessData(filename=data_filename)
    train_X, train_Y = trainingData.formatData(print_shapes=True)


    # ## HYPER-PARAMETER DEFINITIONS ###

    hid = 100
    n_epochs = 2000
    n_iters = 100  # iterations per epoch
    batch_size = 1000
    learning_rate = 0.02
    stacked_hidden_layers = 2
    # #################################

    model = SimpleRNN(inp_dim=train_X.shape[-1],
                      hidden_size=hid,
                      op_dim=train_Y.shape[-1],
                      stacked_layers=stacked_hidden_layers)

    criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters())
    model.cuda()

    losses = np.zeros(n_epochs) # For plotting
    success_tracker = np.zeros(n_epochs)
    for epoch in range(n_epochs):

        for iter in range(n_iters):
            _inputs, _targets = trainingData.sampleBatches(batch_size=batch_size)
            inputs = Variable(torch.from_numpy(_inputs)).type(FloatTensor)
            targets = Variable(torch.from_numpy(_targets)).type(FloatTensor)

            # Use teacher forcing 50% of the time
            force = random.random() < 0.5
            outputs, hidden = model(inputs, None, force)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses[epoch] += loss.data[0]

        if epoch > 0:
            print(epoch, loss.data[0])
        torch.save(model.state_dict(), './Models/test_oralenet.pkl')

        ## test online performance of network
        _, validity = model_eval(cx, cy, obstacle_path, model,
                                 num_evals=100, eval_mode=True, plotopt=False)
        success_tracker[epoch] = validity
        print('Success Rate trends: ', validity)

    plt.figure()
    plt.plot(validity)