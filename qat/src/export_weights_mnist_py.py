import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


def square_act(x):
    return torch.mul(x, x)


class SquareAct(nn.Module):
    def __init__(self):
        super(SquareAct, self).__init__()

    def forward(self, x):
        return torch.mul(x, x)


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 5, 5, stride=(2, 2),
        #                        padding=0, bias=False)
        # # self.act1 = SquareAct()
        # # self.conv2 = nn.Conv2d(83, 163, 6, stride=(2, 2),
        # #                        padding=0, bias=False)
        # self.fc1 = nn.Linear(500, 100, bias=False)
        # self.fc2 = nn.Linear(100, 10, bias=False)
        # self.quant_inp = qnn.QuantIdentity(bit_width=2, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(1, 5, 5, stride=(2, 2),
                                     padding=0, bias=True, weight_bit_width=16, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(5, 50, 5, stride=(2, 2),
                                     padding=0, bias=True, weight_bit_width=16)

        # self.fc1 = qnn.QuantLinear(500, 32, bias=True, weight_bit_width=3, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(800, 10, bias=True, weight_bit_width=16, return_quant_tensor=True)

        self.act1 = SquareAct()
        self.act2 = SquareAct()
        # self.act3 = nn.Sigmoid()

    def forward(self, xb):
        # out = self.quant_inp(xb)
        # out = (((xb*4).int())/4).float() # Scaling to 2bit
        # print(np.min(out.numpy()))
        # print(np.max(out.numpy()))
        # print(out.numpy()[out.numpy() != 0.0])
        out = self.conv1(xb)
        out = out * out
        # out = F.avg_pool2d(out, 3, stride=1, padding=0, divisor_override=1)
        out = self.conv2(out)
        # out = F.avg_pool2d(out, 3, stride=1, padding=1, divisor_override=1)
        out = out.reshape(out.shape[0], -1)
        out = out * out
        out = self.fc1(out)
        # out = out * out
        # out = self.fc2(out)
        # out = self.act3(out)
        return out


class LowPrecisionLeNet(nn.Module):
    def __init__(self):
        super(LowPrecisionLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            1, 6, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(
            6, 16, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(
            256, 120, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(
            120, 84, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(
            84, 10, bias=False, weight_bit_width=3)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 83, 8, stride=(2, 2),
        #                        padding=0, bias=False)
        # self.conv2 = nn.Conv2d(83, 163, 6, stride=(2, 2),
        #                        padding=0, bias=False)
        # self.fc1 = nn.Linear(2608, 10, bias=False)

        self.conv1 = nn.Conv2d(3, 32, 3, stride=(1, 1), padding=0, bias=True,)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=(1, 1), padding=0, bias=True,)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=(1, 1), padding=0, bias=True,)

        self.fc1 = nn.Linear(512, 256, bias=True,)
        self.fc2 = nn.Linear(256, 10, bias=True,)

        # self.conv1 = qnn.QuantConv2d(3, 32, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=16)
        # self.conv2 = qnn.QuantConv2d(32, 64, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=16)
        # self.conv3 = qnn.QuantConv2d(64, 128, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=16)
        #
        # self.fc1 = qnn.QuantLinear(512, 256, bias=True, weight_bit_width=16)
        # self.fc2 = qnn.QuantLinear(256, 10, bias=True, weight_bit_width=16)

        # Small version for debugging
        # self.conv1 = qnn.QuantConv2d(3, 8, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=2)
        # self.conv2 = qnn.QuantConv2d(8, 16, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=2)
        # self.conv3 = qnn.QuantConv2d(16, 32, 3, stride=(1, 1), padding=0, bias=True, weight_bit_width=2)
        #
        # self.fc1 = qnn.QuantLinear(128, 256, bias=True, weight_bit_width=2)
        # self.fc2 = qnn.QuantLinear(256, 10, bias=True, weight_bit_width=2)

        self.act1 = SquareAct()
        self.act2 = SquareAct()
        self.act3 = SquareAct()
        self.act4 = nn.Sigmoid()

    def forward(self, xb):
        # out = self.quant_inp(xb)
        out = self.conv1(xb)
        out = self.act1(out)
        out = F.avg_pool2d(out, 2, stride=2, padding=0, divisor_override=1)
        out = self.conv2(out)
        out = self.act2(out)
        out = F.avg_pool2d(out, 2, stride=2, padding=0, divisor_override=1)
        out = self.conv3(out)
        out = self.act3(out)
        out = F.avg_pool2d(out, 2, stride=2, padding=0, divisor_override=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.act4(out)
        return out


def generate_size(name, layer):
    string = ""

    if "Conv2d" in str(layer.type):
        string += name + "_kernel_size  = " + str(layer.kernel_size) + "\n"
        string += name + "_in_channels  = " + str(layer.in_channels) + "\n"
        string += name + "_out_channels = " + str(layer.out_channels) + "\n"
        string += name + "_stride       = " + str(layer.stride) + "\n"
        string += name + "_dilation     = " + str(layer.dilation) + "\n"
    elif "Linear" in str(layer.type):
        string += name + "_input  = " + str(layer.in_features) + "\n"
        string += name + "_output = " + str(layer.out_features) + "\n"

    string += "\n"
    return string


def generate_string(name, array):
    splitted = name.split("/")
    flat_array = array.flatten()
    variable_name = splitted[0]

    # In case of bias: add to name
    if "bias" in name:
        variable_name = variable_name + "_bias"

    string = "inline double " + str(variable_name) + " [" + str(flat_array.shape[0]) + "] = {"

    # In case of short array, start direct writing. Else add line escape
    if len(flat_array) > 15:
        string += "\n"

    for i in range(len(flat_array)):
        string += str(flat_array[i])
        if (i != len(flat_array) - 1):
            string += ','
        if i % 7 == 0 and i != 0:
            string += '\n'

    # Remove the line escape if it is there
    if string[-1] == '\n':
        string = string[:-1]
    string += "};\n\n"

    return string


def main():
    parser = argparse.ArgumentParser(description='Extract parameters of TF model')
    parser.add_argument('data', metavar='DIR',
                        help='path to Torch model')
    parser.add_argument('export_filepath', metavar='EXPORT',
                        help='path to export file')

    args = parser.parse_args()

    # model = CIFAR10Model()
    model = MNISTModel()
    # model = LowPrecisionLeNet()
    model.load_state_dict(torch.load(args.data, map_location=torch.device('cpu')))
    

    if ".py" not in args.export_filepath:
        f = open(args.export_filepath + ".py", "w")
    else:
        f = open(args.export_filepath, "w")

    f.write("import numpy as np \n")
    f.write("# " + str(args.data) + "\n\n")

    f.write(generate_size("conv2d", model.conv1))
    f.write(generate_size("conv2d_1", model.conv2))
    # f.write(generate_size("conv2d_2", model.conv3))
    f.write(generate_size("dense", model.fc1))
    # f.write(generate_size("dense_1", model.fc2))

    if ".pth" in args.data:
        f.write("conv2d = np.array(" + str(model.conv1.weight.data.tolist()) + ')\n\n')
        f.write("conv2d_1 = np.array(" + str(model.conv2.weight.data.tolist()) + ')\n\n')
        # f.write("conv2d_2 = np.array(" + str(model.conv3.weight.data.tolist()) + ')\n\n')

        f.write("dense = np.array(" + str(model.fc1.weight.data.tolist()) + ')\n\n')
        # f.write("dense_1 = np.array(" + str(model.fc2.weight.data.tolist()) + ')\n\n')
    else:
        f.write("conv2d_scale = " + str(model.conv1.quant_weight().scale.data.tolist()) + '\n\n')
        f.write("conv2d = np.array(" + str(model.conv1.quant_weight().int().tolist()) + ')\n\n')
        f.write("conv2d_1_scale = " + str(model.conv2.quant_weight().scale.data.tolist()) + '\n\n')
        f.write("conv2d_1 = np.array(" + str(model.conv2.quant_weight().int().tolist()) + ')\n\n')
        f.write("conv2d_2_scale = " + str(model.conv3.quant_weight().scale.data.tolist()) + '\n\n')
        f.write("conv2d_2 = np.array(" + str(model.conv3.quant_weight().int().tolist()) + ')\n\n')

        f.write("dense_scale = " + str(model.fc1.quant_weight().scale.data.tolist()) + '\n\n')
        f.write("dense = np.array(" + str(model.fc1.quant_weight().int().tolist()) + ')\n\n')
        f.write("dense_1_scale = " + str(model.fc2.quant_weight().scale.data.tolist()) + '\n\n')
        f.write("dense_1 = np.array(" + str(model.fc2.quant_weight().int().tolist()) + ')\n\n')

    f.write("conv2d_bias = np.array(" + str(model.conv1.bias.data.tolist()) + ')\n\n')
    f.write("conv2d_1_bias = np.array(" + str(model.conv2.bias.data.tolist()) + ')\n\n')
    # f.write("conv2d_2_bias = np.array(" + str(model.conv3.bias.data.tolist()) + ')\n\n')
    f.write("dense_bias = np.array(" + str(model.fc1.bias.data.tolist()) + ')\n\n')
    # f.write("dense_1_bias = np.array(" + str(model.fc2.bias.data.tolist()) + ')\n\n')


if __name__ == '__main__':
    main()
