import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import pandas as pd
from tqdm.notebook import tqdm

from neural_networks import get_quantized_network_with_bits
import pathlib

# local constants
batch_size = 1000
num_epochs = 25
lr = 0.001
num_classes = 10

dataset_dir = pathlib.Path("data")
binarized_dataset_dir = pathlib.Path(dataset_dir, "unsw_nb15_binarized.npz")
non_binarized_train_dataset_dir = pathlib.Path(dataset_dir, "UNSW_NB15_training-set.csv")
non_binarized_test_dataset_dir = pathlib.Path(dataset_dir, "UNSW_NB15_testing-set.csv")


def get_binarized_data(data_dir: str, train: bool):
    unsw_nb15_data = np.load(data_dir + "/unsw_nb15_binarized.npz")
    if train:
        partition = "train"
    else:
        partition = "test"

    part_data = unsw_nb15_data[partition].astype(np.float32)
    part_data = torch.from_numpy(part_data)
    part_data_in = part_data[:, :-1]
    part_data_out = part_data[:, -1]
    return part_data_in, part_data_out


def get_binarized_dataset(data_dir: pathlib.Path, train: bool):
    features, labels = get_binarized_data(data_dir, train)
    return TensorDataset(features, labels)


def get_non_binarized_dataset(data_dir: pathlib.Path, train_partition: bool):
    dataset_df = pd.read_csv(data_dir)

    # load features from binarized dataset
    features, _ = get_binarized_data("data", train_partition)

    # convert categorical labels to numeric labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(dataset_df["attack_cat"])
    numeric_labels_tensor = torch.from_numpy(numeric_labels)
    label_mappings = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mappings: ", label_mappings)
    return TensorDataset(features, numeric_labels_tensor)


def train(model, train_loader, optimizer, criterion):
    losses = []
    acc = []

    # ensure model is in training mode
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(X_train_batch).squeeze()
        loss = criterion(output, y_train_batch)
        train_acc = multi_acc(output, y_train_batch)

        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.data.cpu().numpy())
        acc.append(train_acc.cpu().numpy())

    return losses, acc


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

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    correct = 0
    total = 0
    max_accuracy = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)

            # run the output through sigmoid
            output = model(data)

            # compare against a threshold of 0.5 to generate 0/1
            _, predicted = torch.max(output, dim=1)

            correct_matrix = (predicted == labels)

            c = correct_matrix.squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total += labels.size(0)
            correct += correct_matrix.sum().item()
        max_accuracy = max(max_accuracy, int(100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s [%d/%d]: %2f %%' % (i, class_correct[i], class_total[i],
                                                   100 * class_correct[i] / class_total[i]))
    return max_accuracy, y_true, y_pred


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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


def get_class_distribution(obj):
    count_dict = {
        'Analysis': 0,
        'Backdoor': 0,
        'DoS': 0,
        'Exploits': 0,
        'Fuzzers': 0,
        'Generic': 0,
        'Normal': 0,
        'Reconnaissance': 0,
        'Shellcode': 0,
        'Worms': 0
    }

    for i in obj:
        if i == 0:
            count_dict['Analysis'] += 1
        elif i == 1:
            count_dict['Backdoor'] += 1
        elif i == 2:
            count_dict['DoS'] += 1
        elif i == 3:
            count_dict['Exploits'] += 1
        elif i == 4:
            count_dict['Fuzzers'] += 1
        elif i == 5:
            count_dict['Generic'] += 1
        elif i == 6:
            count_dict['Normal'] += 1
        elif i == 7:
            count_dict['Reconnaissance'] += 1
        elif i == 8:
            count_dict['Shellcode'] += 1
        elif i == 9:
            count_dict['Worms'] += 1
        else:
            print("Check classes.")

    return count_dict

if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    # get non-quantized dataset
    train_dataset = get_non_binarized_dataset(non_binarized_train_dataset_dir, True)
    test_dataset = get_non_binarized_dataset(non_binarized_test_dataset_dir, False)

    # dataset loaders
    train_quantized_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_quantized_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ########################################################
    ########################################################
    weight_bit = 2
    act_bit = 2

    # get weight of classes
    class_count = [i for i in get_class_distribution(train_dataset.tensors[1]).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    # get QNN
    print(f"[Quantized network] [weight: {weight_bit}][act: {act_bit}]")
    quantized_model = get_quantized_network_with_bits(weight_bit, act_bit, num_classes)
    quantized_model.to(device)

    # specify loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=lr, betas=(0.9, 0.999))

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("--- Begin training ---")
    for e in tqdm(range(1, 10 + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        quantized_model.train()
        for X_train_batch, y_train_batch in train_quantized_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = quantized_model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            quantized_model.eval()
            loss_stats['train'].append(train_epoch_loss / len(train_quantized_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_quantized_loader))
            print(f'Epoch {e + 0:03}: | '
                  f'Train Loss: {train_epoch_loss / len(train_quantized_loader):.5f} | '
                  f'Train Acc: {train_epoch_acc / len(train_quantized_loader):.3f}| ')

    # t = trange(10, desc="Training loss", leave=True)
    # for epoch in t:
    #     loss_epoch, acc_epoch = train(quantized_model, train_quantized_loader, optimizer, criterion)
    #     test_acc, y_true, y_pred = test(quantized_model, test_quantized_loader)
    #     t.set_description(
    #         "Training loss = %f | train accuracy = %f | test accuracy" % (np.mean(loss_epoch), np.mean(acc_epoch)), test_acc)
    #     t.refresh()
    #
    # test_acc, y_true, y_pred = test(quantized_model, test_quantized_loader)
    # display_confusion_matrix(y_true, y_pred, f"Quantized network [weight: {weight_bit}][act: {act_bit}]")
    # print(f"--> Test accuracy: {test_acc * 100}%")

    ########################################################
    ########################################################
    sys.exit(1)

    ##################
    # Quantized model
    ##################
    # get model for training
    # quantized_model = get_quantized_network()
    weight_bits = [2]
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
            torch.save(quantized_model.state_dict(), "model_files/state_dict_quantized-model.pth")

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
