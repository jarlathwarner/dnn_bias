import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from tqdm.notebook import tqdm

from neural_networks import get_quantized_network_with_bits, get_non_quantized_network

# local constants
batch_size = 256
num_epochs = 50
lr = 0.001
num_classes = 10

dataset_dir = pathlib.Path("data")
binarized_dataset_dir = pathlib.Path(dataset_dir, "unsw_nb15_binarized.npz")
non_binarized_train_dataset_dir = pathlib.Path(dataset_dir, "UNSW_NB15_training-set.csv")
non_binarized_test_dataset_dir = pathlib.Path(dataset_dir, "UNSW_NB15_testing-set.csv")
results_dir = pathlib.Path("cybsec_mlp/results")

classes = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode',
           'Worms']


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


def create_confusion_matrix(actual, predicted):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    print(f"--> Confusion matrix: \n{confusion_matrix_df}")

    # calculate per class accuracies
    cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    class_accuracies = {k: v for (k, v) in zip(classes, cm.diagonal())}

    # add per class accuracies to dictionary
    for val in classes:
        experiment_stats[val] = class_accuracies[val]

    print(f"<-- Class accuracies -->\n{class_accuracies}")
    return confusion_matrix_df


def display_confusion_matrix(actual, predicted, title):
    confusion_matrix = create_confusion_matrix(actual, predicted)

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')
    ax.set(title=title)
    plt.tight_layout()

    cm_filename = f"cm_qnn_weight{weight_bit}_act{act_bit}.png"
    cm_filepath = pathlib.Path(results_dir / cm_filename)
    plt.savefig(cm_filepath, dpi=1000)
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


def begin_training_process(model, train_data_loader):
    print("--- Begin training ---")

    for e in tqdm(range(1, num_epochs + 1)):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_data_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            model.eval()
            loss_stats['train'].append(train_epoch_loss / len(train_quantized_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_quantized_loader))
            print(f'Epoch {e + 0:03}: | '
                  f'Train Loss: {train_epoch_loss / len(train_quantized_loader):.5f} | '
                  f'Train Acc: {train_epoch_acc / len(train_quantized_loader):.3f}| ')
    display_loss_plot(loss_stats['train'])


def begin_testing_process(model, test_data_loader):
    print("--- Begin testing ---")
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_data_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_arr = np.concatenate(y_pred_list)
    y_true = test_data_loader.dataset.tensors[1]

    test_acc = accuracy_score(y_true, np.concatenate(y_pred_list))
    experiment_stats['overall_acc'] = test_acc

    print(f"Accuracy: {test_acc * 100}%")
    print(classification_report(y_true, y_pred_arr))
    display_confusion_matrix(y_true, y_pred_arr, f"Test Accuracy = {test_acc * 100:.3f}%")


if __name__ == "__main__":
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    # get non-quantized dataset
    train_dataset = get_non_binarized_dataset(non_binarized_train_dataset_dir, True)
    test_dataset = get_non_binarized_dataset(non_binarized_test_dataset_dir, False)

    # dataset loaders
    train_quantized_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_quantized_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # get weight of classes
    class_count = [i for i in get_class_distribution(train_dataset.tensors[1]).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))

    ########################################################
    # EXPERIMENTS
    ########################################################
    weight_bits = [6, 8, 16, 32]
    act_bits = [2, 4, 6, 8, 16, 32]
    for weight_bit in weight_bits:
        for act_bit in act_bits:
            # get QNN
            print(f"[Quantized network] [weight: {weight_bit}][act: {act_bit}]")
            model = get_quantized_network_with_bits(weight_bit, act_bit, num_classes)

            # print(f"[Non-Quantized network]")
            # model = get_non_quantized_network(num_classes)

            model.to(device)

            # specify loss criterion and optimizer
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

            accuracy_stats = {
                'train': [],
                "val": []
            }
            loss_stats = {
                'train': [],
                "val": []
            }

            experiment_stats = {k: v for (k, v) in zip(['overall_acc'] + classes, list(np.zeros(num_classes,)))}

            begin_training_process(model, train_quantized_loader)
            begin_testing_process(model, test_quantized_loader)

            # export experiment results to csv
            experiment_stats_df = pd.DataFrame(experiment_stats.items())
            filename = f"cybsec_mlp/results/qnn_weight{weight_bit}_act{act_bit}.csv"
            filepath = pathlib.Path(results_dir / filename)
            experiment_stats_df.to_csv(filename, sep='\t', encoding='utf-8')
