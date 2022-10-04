import onnx
import torch
import numpy as np
from torch.utils.data import TensorDataset


def get_preqnt_dataset(data_dir: str, train: bool):
    unsw_nb15_data = np.load(data_dir + "/unsw_nb15_binarized.npz")
    if train:
        partition = "train"
    else:
        partition = "test"

    part_data = unsw_nb15_data[partition].astype(np.float32)
    part_data = torch.from_numpy(part_data)
    part_data_in = part_data[:, :-1]
    part_data_out = part_data[:, :-1]
    return TensorDataset(part_data_in, part_data_out)


train_quantized_dataset = get_preqnt_dataset(".", True)
test_quantized_dataset = get_preqnt_dataset(".", False)

print("Samples ine ach set: train = %d, test = %s" % (len(train_quantized_dataset), len(test_quantized_dataset)))
print("Shape of one input sample:" + str(train_quantized_dataset[0][0].shape))
