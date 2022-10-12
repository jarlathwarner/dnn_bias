from brevitas.nn import QuantLinear, QuantReLU
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

input_size = 593
hidden1 = 64
hidden2 = 64
hidden3 = 64
weight_bit_width = 2
act_bit_width = 2
num_classes = 1


def get_quantized_network():
    model = Sequential(
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
        BatchNorm1d(hidden1),
        Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
        BatchNorm1d(hidden2),
        Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bit_width),
        BatchNorm1d(hidden3),
        Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bit_width)
    )
    return model


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
