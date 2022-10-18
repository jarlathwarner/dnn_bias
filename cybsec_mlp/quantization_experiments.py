import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from neural_networks import get_quantized_network_with_bits

# local constants
batch_size = 1000
num_epochs = 25
lr = 0.001


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


def display_confusion_matrix(actual, predicted, title):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    print(f"--> Confusion matrix: {confusion_matrix}")

    class_bad_acc = confusion_matrix[0][0] / np.sum(confusion_matrix[0])
    class_normal_acc = confusion_matrix[1][1] / np.sum(confusion_matrix[1])
    print(f"--> Class: Bad accuracy: {class_bad_acc}")
    print(f"--> Class: Normal accuracy: {class_normal_acc}")

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title(title)
    plt.show()


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

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def run_model(model_to_run, train_data_loader, test_data_loader, criterion):
    optimizer = torch.optim.Adam(model_to_run.parameters(), lr=lr, betas=(0.9, 0.999))

    running_loss = []
    running_test_acc = []
    t = trange(num_epochs, desc="Training loss", leave=True)

    for epoch in t:
        loss_epoch = train(model_to_run, train_data_loader, optimizer, criterion)
        test_acc, y_true, y_pred = test(model_to_run, test_data_loader)
        t.set_description("Training loss = %f test accuracy = %f" % (np.mean(loss_epoch), test_acc))
        t.refresh()  # to show immediately the update
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)

    loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
    display_loss_plot(loss_per_epoch)

    acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
    display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")


def run_latency_test(model_to_run, data_loader):
    model_to_run.eval()
    elapsed_times = []

    print("--> Running inference time test <---")
    n_iterations = 100
    for i in range(n_iterations):
        inputs, target = next(iter(data_loader))

        # start time here
        start_time = time.time()
        model_to_run(inputs.float())
        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    print(f"Average inference time: {avg_elapsed_time}\n")


def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    # get quantized dataset
    train_quantized_dataset = get_preqnt_dataset("data", True)
    test_quantized_dataset = get_preqnt_dataset("data", False)
    print("Samples in each set: train = %d, test = %s" % (len(train_quantized_dataset), len(test_quantized_dataset)))
    print("Shape of one input sample:" + str(train_quantized_dataset[0][0].shape))

    # dataset loaders
    train_quantized_loader = DataLoader(train_quantized_dataset, batch_size=batch_size, shuffle=True)
    test_quantized_loader = DataLoader(test_quantized_dataset, batch_size=batch_size, shuffle=False)

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    ##################
    # Quantized model
    ##################
    # get model for training
    # quantized_model = get_quantized_network()
    weight_bits = [16, 32]
    act_bits = [2, 4, 6, 8]
    for weight_bit in weight_bits:
        for act_bit in act_bits:
            print(f"[Quantized network] [weight: {weight_bit}][act: {act_bit}]")
            quantized_model = get_quantized_network_with_bits(weight_bit, act_bit)
            quantized_model.to(device)

            # specify loss criterion and optimizer
            criterion = nn.BCEWithLogitsLoss().to(device)
            run_model(quantized_model, train_quantized_loader, test_quantized_loader, criterion)

            test_acc, y_true, y_pred = test(quantized_model, test_quantized_loader)
            display_confusion_matrix(y_true, y_pred, f"Quantized network [weight: {weight_bit}][act: {act_bit}]")
            print(f"--> Test accuracy: {test_acc * 100}%")

            run_latency_test(quantized_model, test_quantized_loader)

            # save the brevitas model to disk
            torch.save(quantized_model.state_dict(), "state_dict_quantized-model.pth")

    ######################
    # Non-Quantized model
    ######################
    # get model for training
    # non_quantized_model = get_non_quantized_network()
    # non_quantized_model.to(device)
    #
    # # specify loss criterion and optimizer
    # run_model(non_quantized_model, train_quantized_loader, test_quantized_loader, criterion)
    #
    # test_acc, y_true, y_pred = test(non_quantized_model, test_quantized_loader)
    # display_confusion_matrix(y_true, y_pred, "Non-Quantized network")
    # print("[Non-Quantized network]")
    # print(f"--> Test accuracy: {test_acc * 100}%")
    #
    # run_latency_test(non_quantized_model, test_quantized_loader)
    #
    # # save the brevitas model to disk
    # torch.save(non_quantized_model.state_dict(), "state_dict_non-quantized-model.pth")
