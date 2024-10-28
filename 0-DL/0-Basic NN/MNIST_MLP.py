import os
import struct
import time
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

def show_matrix(mat, name):
    pass

def show_time(time, name):
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

class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=2,
                 print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.lowest_loss = float("inf")

    # ----------------------------------------------------------------------------------------
    # --------                   DATA LOAD MODULE                                        -----
    # ----------------------------------------------------------------------------------------

    def load_mnist(self, file_dir, is_images=True):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:  # read image data
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:  # read label data
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1  # Set num_rows and num_cols for label data
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        # True is image data, False is label data
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    # ----------------------------------------------------------------------------------------
    # --------                   NETWORK ARCHITECTURE MODULE                             -----
    # ----------------------------------------------------------------------------------------
    # build network architecture
    def build_model(self):
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    # ----------------------------------------------------------------------------------------
    # --------                   NETWORK TRAINING MODULE                                 -----
    # ----------------------------------------------------------------------------------------
    # forward training of neural network
    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax.forward(h3)  # output probability
        return prob

    # backward training of neural network
    def backward(self):
        dloss = self.softmax.backward()  # Cross entropy loss function
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    # update parameter
    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    # save parameter
    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    # Shuffle training data to ensure better training performance
    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size  # python3

        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_images = batch_images.reshape(self.batch_size, -1)  # Ensure the input shape is (batch_size, 784)
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1].astype(int)
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
                    if loss < self.lowest_loss:
                        self.lowest_loss = loss
                        print('find lowest loss, saving model')
                        self.save_model('mlp-%d-%d-%depoch.npy' % (self.hidden1, self.hidden2, self.max_epoch))

    # ----------------------------------------------------------------------------------------
    # --------                   NETWORK INFERENCE MODULE                                 -----
    # ----------------------------------------------------------------------------------------
    # load network parameter
    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir, allow_pickle=True).item()
        # weight parameter
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    # main function of inference
    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            batch_images = batch_images.reshape(self.batch_size, -1)  # Ensure the input shape is (batch_size, 784)
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set:%f' % accuracy)

def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 100, 100, 20
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    mlp.load_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
