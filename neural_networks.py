from brevitas.nn import QuantLinear, QuantReLU
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

input_size = 593
hidden1 = 64
hidden2 = 64
hidden3 = 64
num_classes = 1

# quantization bit widths
weight_bit_width = 2
act_bit_width = 2


def get_quantized_network_with_bits(weight_bits, act_bits):
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
        QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bits)
    )
    return model


def get_quantized_network():
    return get_quantized_network_with_bits(weight_bit_width, act_bit_width)


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
