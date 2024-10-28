import numpy as np
import struct
import os
import time


def show_matrix(mat, name):
    # print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass


def show_time(time, name):
    # print(name + str(time))
    pass


# ----------------------------------------------------------------------------------------
# --------                   BASIC UNIT MODULE                                       -----
# ----------------------------------------------------------------------------------------
class FullyConnectedLayer(object):
    # Layer initialize with input number and output number
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))

    # parameter initialize
    def init_param(self, std=0.01):
        # random weight initialize
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        # all zero initialize for bias
        self.bias = np.zeros([1, self.num_output])
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')

    # forward computing
    def forward(self, input_value):
        self.input = input_value
        # Y= W * X + b
        self.output = self.input.dot(self.weight) + self.bias
        return self.output

    # backward computing
    def backward(self, top_diff):
        self.d_weight = np.matmul(self.input.T, top_diff)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        bottom_diff = np.matmul(top_diff, self.weight.T)
        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias

    # parameter update
    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    # load parameter
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')

    # save parameter
    def save_param(self):
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
        return self.weight, self.bias


class ReLULayer(object):
    def __init__(self):
        print('\t Relu layer')

    # ReLU forward computing
    def forward(self, input):
        self.input = input
        # ReLU= max(0,x)
        output = np.maximum(0, self.input)
        return output

    # ReLU backward computing, deliver the gradient which larger than zero to next layer
    def backward(self, top_diff):
        bottom_diff = top_diff * (self.input >= 0.)
        return bottom_diff


class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')

    # Softmax forward computing
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        exp_sum = np.sum(input_exp, axis=1, keepdims=True)
        self.prob = input_exp / exp_sum
        return self.prob

    # loss function
    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-10) * self.label_onehot) / self.batch_size  # Add epsilon to avoid log(0)
        return loss

    # Softmax backward computing
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # convolutional layer initialization
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (
        self.kernel_size, self.channel_in, self.channel_out))

    # initialize parameter
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std,
                                       size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):  # forward computing
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        # Padding
        height = self.input.shape[2] + 2 * self.padding
        width = self.input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding + self.input.shape[2],
        self.padding:self.padding + self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size + self.stride) // self.stride
        width_out = (height - self.kernel_size + self.stride) // self.stride
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.weight[:, :, :, idxc] * self.input_pad[idxn, :,
                                                         idxh * self.stride:idxh * self.stride + self.kernel_size,
                                                         idxw * self.stride:idxw * self.stride + self.kernel_size]) + \
                                                              self.bias[idxc]
        return self.output

    def load_param(self, weight, bias):  # parameter load
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # max pooling layer initialization
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))

    def forward(self, input):
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size + self.stride) // self.stride
        width_out = (self.input.shape[3] - self.kernel_size + self.stride) // self.stride
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = self.input[idxn, idxc,
                                                              idxh * self.stride:idxh * self.stride + self.kernel_size,
                                                              idxw * self.stride:idxw * self.stride + self.kernel_size].max()
        return self.output


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
