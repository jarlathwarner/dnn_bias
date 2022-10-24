from brevitas.nn import QuantLinear, QuantReLU
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, Conv2d, MaxPool2d

input_size = 593
hidden1 = 64
hidden2 = 64
hidden3 = 64
num_classes = 1

# quantization bit widths
weight_bit_width = 2
act_bit_width = 2


def get_quantized_network_with_bits(weight_bits, act_bits, num_outputs):
    model = Sequential(
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bits),
        BatchNorm1d(hidden1),
        Dropout(0.5),
        QuantReLU(bit_width=act_bits),
        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bits),
        BatchNorm1d(hidden2),
        Dropout(0.5),
        QuantReLU(bit_width=act_bits),
        QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bits),
        BatchNorm1d(hidden3),
        Dropout(0.5),
        QuantReLU(bit_width=act_bits),
        QuantLinear(hidden3, num_outputs, bias=True, weight_bit_width=weight_bits)
    )
    return model


def get_quantized_network(num_outputs=num_classes):
    return get_quantized_network_with_bits(weight_bit_width, act_bit_width, num_outputs)


def get_non_quantized_network():
    model = Sequential(
        Linear(input_size, hidden1, bias=True),
        BatchNorm1d(hidden1),
        Dropout(0.5),
        ReLU(),
        Linear(hidden1, hidden2, bias=True),
        BatchNorm1d(hidden2),
        Dropout(0.5),
        ReLU(),
        Linear(hidden2, hidden3, bias=True),
        BatchNorm1d(hidden3),
        Dropout(0.5),
        ReLU(),
        Linear(hidden3, num_classes, bias=True)
    )
    return model


def get_non_quantized_conv2d_network(num_outputs):
    model = Sequential(
        Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Linear(32 * 7 * 7, num_outputs)
    )
    return model
