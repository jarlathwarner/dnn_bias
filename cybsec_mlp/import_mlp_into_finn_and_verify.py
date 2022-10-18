import finn.core.onnx_exec as oxe
import numpy as np
import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from torch.utils.data import TensorDataset
from tqdm import trange


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


def inference_with_brevitas(current_inp):
    brevitas_output = brevitas_model.forward(current_inp)

    # apply sigmoid + threshold
    brevitas_output = torch.sigmoid(brevitas_output)
    brevitas_output = (brevitas_output.detach().numpy() > 0.5) * 1

    # convert output to bipolar
    brevitas_output = 2 * brevitas_output - 1
    return brevitas_output


def inference_with_finn_onnx(current_inp):
    finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
    finnonnx_model_in_shape = model_for_sim.get_tensor_shape(finnonnx_in_tensor_name)
    finnonnx_out_tensor_name = model_for_sim.graph.output[0].name

    # convert input to numpy for FINN
    current_inp = current_inp.detach().numpy()

    # add padding and re-scale to bipolar
    current_inp = np.pad(current_inp, [(0, 0), (0, 7)])
    current_inp = 2 * current_inp - 1

    # reshape to expected input (add 1 for batch dimension)
    current_inp = current_inp.reshape(finnonnx_model_in_shape)

    # create the input dictionary
    input_dict = {finnonnx_in_tensor_name: current_inp}

    # run with FINN's execute_onnx
    output_dict = oxe.execute_onnx(model_for_sim, input_dict)

    # get the output tensor
    finn_output = output_dict[finnonnx_out_tensor_name]
    return finn_output


if __name__ == "__main__":
    ready_model_filename = "cybsec-mlp-ready.onnx"
    model_for_sim = ModelWrapper(ready_model_filename)

    dir(model_for_sim)

    finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
    finnonnx_out_tensor_name = model_for_sim.graph.output[0].name
    print("Input tensor name: %s" % finnonnx_in_tensor_name)
    print("Output tensor name: %s" % finnonnx_out_tensor_name)

    finnonnx_model_in_shape = model_for_sim.get_tensor_shape(finnonnx_in_tensor_name)
    finnonnx_model_out_shape = model_for_sim.get_tensor_shape(finnonnx_out_tensor_name)
    print("Input tensor shape: %s" % finnonnx_model_in_shape)
    print("Output tensor shape: %s" % finnonnx_model_out_shape)

    finnonnx_model_in_dt = model_for_sim.get_tensor_datatype(finnonnx_in_tensor_name)
    finnonnx_model_out_dt = model_for_sim.get_tensor_datatype(finnonnx_out_tensor_name)
    print("Input tensor datatype: %s" % finnonnx_model_in_dt)
    print("Output tensor datatype: %s" % finnonnx_model_out_dt)
    print("List of node operator types in the graph: ")
    print([x.op_type for x in model_for_sim.graph.node])

    model_for_sim = model_for_sim.transform(InferShapes())
    model_for_sim = model_for_sim.transform(FoldConstants())
    model_for_sim = model_for_sim.transform(GiveUniqueNodeNames())
    model_for_sim = model_for_sim.transform(GiveReadableTensorNames())
    model_for_sim = model_for_sim.transform(InferDataTypes())
    model_for_sim = model_for_sim.transform(RemoveStaticGraphInputs())

    verif_model_filename = "cybsec-mlp-verification.onnx"
    model_for_sim.save(verif_model_filename)

    n_verification_inputs = 100
    test_quantized_dataset = get_preqnt_dataset("../data", False)
    input_tensor = test_quantized_dataset.tensors[0][:n_verification_inputs]
    print("input_tensor.shape: " + str(input_tensor.shape))

    input_size = 593
    hidden1 = 64
    hidden2 = 64
    hidden3 = 64
    weight_bit_width = 2
    act_bit_width = 2
    num_classes = 1

    brevitas_model = nn.Sequential(
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden1),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden2),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden3),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bit_width)
    )

    trained_state_dict = torch.load("state_dict_self-trained.pth")

    brevitas_model.load_state_dict(trained_state_dict, strict=False)

    verify_range = trange(n_verification_inputs, desc="FINN execution", position=0, leave=True)
    brevitas_model.eval()

    ok = 0
    nok = 0

    for i in verify_range:
        # run in Brevitas with pytorch tensor
        current_inp = input_tensor[i].reshape((1, 593))
        brevitas_output = inference_with_finn_onnx(current_inp)
        finn_output = inference_with_finn_onnx(current_inp)

        # compare the outputs
        ok += 1 if finn_output == brevitas_output else 0
        nok += 1 if finn_output != brevitas_output else 0

        verify_range.set_description("ok %d nok %d" % (ok, nok))
        verify_range.refresh()

    if ok == n_verification_inputs:
        print("Verification succeeded. Brevitas and FINN-ONNX execution outputs are identical")
    else:
        print("Verification failed. Brevitas and FINN-ONNX execution outputs are NOT identical")
