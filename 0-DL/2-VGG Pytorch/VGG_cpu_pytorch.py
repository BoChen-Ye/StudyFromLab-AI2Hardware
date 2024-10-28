import os
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms

os.putenv('MLU_VISIBLE_DEVICES', '')
cfgs = [64, 'R', 64, 'R', 'M', 128, 'R', 128, 'R', 'M',
        256, 'R', 256, 'R', 256, 'R', 256, 'R', 'M',
        512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M',
        512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M']

IMAGE_PATH = './cat.jpg'
VGG_PATH = './vgg19.pth'


# =========================================================================================
#                        NETWORK ARCHITECTURE MODULE                                 ======
# =========================================================================================
def vgg19():
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
    ]
    layer_container = nn.Sequential()
    in_channels = 3
    num_classes = 1000
    for i, layer_name in enumerate(layers):
        if layer_name.startswith('conv'):
            # input convolution layer in container
            while cfgs and isinstance(cfgs[0], str):
                cfgs.pop(0)
            out_channels = cfgs.pop(0) if cfgs else in_channels
            layer_container.add_module(layer_name, nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        elif layer_name.startswith('relu'):
            # input ReLU layer in container
            layer_container.add_module(layer_name, nn.ReLU(inplace=True))
        elif layer_name.startswith('pool'):
            # input Max Pooling layer in container
            layer_container.add_module(layer_name, nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer_name == 'flatten':
            # input flatten layer in container
            layer_container.add_module(layer_name, nn.Flatten())
        elif layer_name == 'fc6':
            #  input FC layer in container
            out_features = 4096
            layer_container.add_module(layer_name, nn.Linear(512 * 7 * 7, out_features))
            in_channels = out_features
        elif layer_name == 'fc7':
            #  input FC layer in container
            out_features = 4096
            layer_container.add_module(layer_name, nn.Linear(in_channels, out_features))
            in_channels = out_features
        elif layer_name == 'fc8':
            #  input FC layer in container
            out_features = num_classes
            layer_container.add_module(layer_name, nn.Linear(in_channels, out_features))
            in_channels = out_features
        elif layer_name == 'softmax':
            # input Softmax layer in container
            layer_container.add_module(layer_name, nn.Softmax(dim=1))
    return layer_container


# =========================================================================================
#                        NETWORK ARCHITECTURE MODULE                                 ======
# =========================================================================================
def load_image(path):
    # Use the Image.open module to read in the input image
    # return the array image with the shape of (1,244,244,3).
    image = Image.open(path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    # Call transform function to preprocess the image.
    image = transform(image)
    # Extend the 0 th dimension of tensor.
    image = image.unsqueeze(0)
    return image


# =========================================================================================
#                        CPU INFERENCE MODULE                                        ======
# =========================================================================================
if __name__ == '__main__':
    input_image = load_image(IMAGE_PATH)
    # generate VGG19 model and save to net
    net = vgg19()
    # load parameter to net
    net.load_state_dict(torch.load(VGG_PATH))
    # The model enters reasoning mode.
    net.eval()
    st = time.time()
    # calculate net and get prob
    prob = net(input_image)
    print("cpu infer time:{:.3f} s".format(time.time() - st))
    with open('./imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        _, indices = torch.sort(prob, descending=True)
    print("Classification result: id = %s, prob = %f " % (classes[indices[0][0]], prob[0][indices[0][0]].item()))

    '''    
    if classes[indices[0][0]] == 'strawberry':
        print('TEST RESULT PASS.')
    else:
        print('TEST RESULT FAILED.')
        exit()
    '''

