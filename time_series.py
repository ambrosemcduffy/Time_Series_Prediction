import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

# How many time-steps/data pts are in one batch of data
seq_length = 20

'''
first example of test data.
'''
# generate evenly spaced data pts over specific interval
start = 0
stop = np.pi
num = seq_length + 1
time_steps = np.linspace(start, stop, num)
data = np.sin((time_steps))
# resizing to 21, 1
data.resize((seq_length + 1, 1))
# x all but the last piece of data
x = data[:-1]
# all but the first piece
y = data[1:]
# plt.plot(time_steps[1:], x, 'r.', label='input, x')
# plt.plot(time_steps[1:], y, 'b.', label='input, y')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        # defining the RNN parameters
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # setting up fully connected layers params
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # hidden = (nlayers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return output, hidden


test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))
test__input = torch.Tensor(data).unsqueeze(0)
print('Input size: {}'.format(test__input.size()))
test_out, test_h = test_rnn(test__input, None)
print('Output size:', test_out.size())
print('Hidden state size:', test_h.size())

input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 2
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
# because it's coordinate values use Mean squared Error
criterion = nn.MSELoss()
# For RNN the optimizer is Adam by default.
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


def train(rnn, n_steps, print_every):
    # initializing the hidden state
    hidden = None
    for batch_i, step, in enumerate(range(n_steps)):
        # creating the training data
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length+1)
        data = np.sin((time_steps))
        data.resize(seq_length+1, 1)
        x = data[:-1]
        y = data[1:]
        # convert to torch tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)
        # getting out predicted output
        prediction, hidden = rnn(x_tensor, hidden)
        hidden = hidden.data
        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_i % print_every == 0:
            print('loss:', loss.item())
    plt.plot(time_steps[1:], x, 'r.')
    plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')
    plt.show()
    return rnn

# initialize the number of steps to be taken.
n_steps = 300
print_every = 100
train_rnn = train(rnn, n_steps, print_every)
