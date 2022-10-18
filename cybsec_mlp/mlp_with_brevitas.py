from copy import deepcopy

import brevitas.onnx as bo
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear, QuantReLU
from brevitas.quant_tensor import QuantTensor
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from neural_networks import get_quantized_network


def get_preqnt_dataset(data_dir: str, train: bool):
    unsw_nb15_data = np.load(data_dir + "/unsw_nb15_binarized.npz")
    if train:
        partition = "train"
    else:
        partition = "test"

    part_data = unsw_nb15_data[partition].astype(np.float32)
    part_data = torch.from_numpy(part_data)
    part_data_in = part_data[:, :-1]
    part_data_out = part_data[:, -1]
    return TensorDataset(part_data_in, part_data_out)


# Define Train and Test methods
def train(model, train_loader, optimizer, criterion):
    losses = []
    # ensure model is in training mode
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(inputs.float())
        loss = criterion(output, target.unsqueeze(1))

        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.data.cpu().numpy())

    return losses


def test(model, test_loader):
    # ensure model is in eval mode
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            output_orig = model(inputs.float())
            # run the output through sigmoid
            output = torch.sigmoid(output_orig)
            # compare against a threshold of 0.5 to generate 0/1
            pred = (output.detach().cpu().numpy() > 0.5) * 1
            target = target.cpu().float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    return accuracy_score(y_true, y_pred)


def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


class CybSecMLPForExport(nn.Module):
    def __init__(self, my_pretrained_model):
        super(CybSecMLPForExport, self).__init__()
        self.pretrained = my_pretrained_model
        self.qnt_output = QuantIdentity(
            quant_type='binary',
            scaling_impl_type='const',
            bit_width=1, min_val=-1.0, max_val=1.0)

    def forward(self, x):
        # assume x contains bipolar {-1,1} elems
        # shift from {-1,1} -> {0,1} since that is the
        # input range for the trained network
        x = (x + torch.tensor([1.0]).to(x.device)) / 2.0
        out_original = self.pretrained(x)
        out_final = self.qnt_output(out_original)  # output as {-1,1}
        return out_final


def test_padded_bipolar(model, test_loader):
    # ensure model is in eval mode
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            # pad inputs to 600 elements
            input_padded = torch.nn.functional.pad(inputs, (0, 7, 0, 0))
            # convert inputs to {-1,+1}
            input_scaled = 2 * input_padded - 1
            # run the model
            output = model(input_scaled.float())
            y_pred.extend(list(output.flatten().cpu().numpy()))
            # make targets bipolar {-1,+1}
            expected = 2 * target.float() - 1
            expected = expected.cpu().numpy()
            y_true.extend(list(expected.flatten()))

    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    train_quantized_dataset = get_preqnt_dataset("../data", True)
    test_quantized_dataset = get_preqnt_dataset("../data", False)

    print("Samples in each set: train = %d, test = %s" % (len(train_quantized_dataset), len(test_quantized_dataset)))
    print("Shape of one input sample:" + str(train_quantized_dataset[0][0].shape))

    batch_size = 1000

    # dataset loaders
    train_quantized_loader = DataLoader(train_quantized_dataset, batch_size=batch_size, shuffle=True)
    test_quantized_loader = DataLoader(test_quantized_dataset, batch_size=batch_size, shuffle=False)

    count = 0
    for x, y in train_quantized_loader:
        print("Input shape for 1 batch: " + str(x.shape))
        print("Label shape for 1 batch: " + str(y.shape))
        count += 1
        if count == 1:
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    input_size = 593
    hidden1 = 64
    hidden2 = 64
    hidden3 = 64
    weight_bit_width = 2
    act_bit_width = 2
    num_classes = 1

    # set seed
    torch.manual_seed(0)

    quantized_model = get_quantized_network()
    quantized_model.to(device)

    num_epochs = 10
    lr = 0.001

    # loss criterion and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=lr, betas=(0.9, 0.999))

    # setting seeds
    torch.manual_seed(0)
    np.random.seed(0)

    running_loss = []
    running_test_acc = []
    t = trange(num_epochs, desc="Training loss", leave=True)

    for epoch in t:
        loss_epoch = train(quantized_model, train_quantized_loader, optimizer, criterion)
        test_acc = test(quantized_model, test_quantized_loader)
        t.set_description("Training loss = %f test accuracy = %f" % (np.mean(loss_epoch), test_acc))
        t.refresh()  # to show immediately the update
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)

    loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
    display_loss_plot(loss_per_epoch)

    acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
    display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")

    test(quantized_model, test_quantized_loader)

    # save the brevitas model to disk
    torch.save(quantized_model.state_dict(), "state_dict_self-trained.pth")

    # move the model to CPU before surgery
    quantized_model = quantized_model.cpu()

    modified_model = deepcopy(quantized_model)

    W_orig = modified_model[0].weight.data.detach().numpy()
    print("W_orig_shape: " + str(W_orig.shape))

    # pad the second (593-sized) dimensions with 7 zeros at the end
    W_new = np.pad(W_orig, [(0, 0), (0, 7)])
    print("W_new_shape: " + str(W_new.shape))

    modified_model[0].weight.data = torch.from_numpy(W_new)
    print("modified_model[0].weight.shape: " + str(modified_model[0].weight.shape))

    model_for_export = CybSecMLPForExport(modified_model)
    model_for_export.to(device)

    test_padded_bipolar(model_for_export, test_quantized_loader)

    ready_model_filename = "cybsec-mlp-ready.onnx"
    input_shape = (1, 600)

    # create a QuantTensor instance to mark input as bipolar during export
    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)
    input_qt = QuantTensor(
        input_t, scale=torch.tensor(scale), bit_width=torch.tensor(1.0), signed=True
    )

    # Move to CPU before export
    model_for_export.cpu()

    # Export to ONNX
    bo.export_finn_onnx(
        model_for_export, export_path=ready_model_filename, input_t=input_qt
    )

    print("Model saved to %s" % ready_model_filename)
