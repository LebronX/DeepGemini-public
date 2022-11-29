from lrp_toolbox.model_io import write, read
import numpy as np
from german_fairness_training import FairHalfNet
import torch
import torch.nn as nn

# import onnx
# import keras
# import tensorflow as tf

import sys

sys.path.append("/Users/xiexuan/Downloads/Research/DeepProperty/DeepProperty")


def find_relevant_neurons(
    model_path, lrpmodel, inps, subject_layer, num_rel, lrpmethod=None
):
    lrpmodel = read(model_path + ".txt", "txt")
    # final_relevants = np.zeros([1, kerasmodel.layers[subject_layer].output_shape[-1]])

    totalR = None
    cnt = 0
    for inp in inps:
        cnt += 1
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        # prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
        mask = np.zeros_like(ypred)
        mask[:, np.argmax(ypred)] = 1
        Rinit = ypred * mask

        if not lrpmethod:
            R_inp, R_all = lrpmodel.lrp(
                Rinit
            )  # as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == "epsilon":
            R_inp, R_all = lrpmodel.lrp(
                Rinit, "epsilon", 0.01
            )  # as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == "alphabeta":
            R_inp, R_all = lrpmodel.lrp(
                Rinit, "alphabeta", 3
            )  # as Eq(60) from DOI: 10.1371/journal.pone.0130140
        else:
            print("Unknown LRP method!")
            raise Exception

        if totalR:
            for idx, elem in enumerate(totalR):
                totalR[idx] = elem + R_all[idx]
        else:
            totalR = R_all

    #      THE MOST RELEVANT                               THE LEAST RELEVANT
    return (
        np.argsort(totalR[subject_layer])[0][::-1][:num_rel],
        np.argsort(totalR[subject_layer])[0][:num_rel],
        totalR,
    )


network_file_pt = "../../networks/german/german_bias_saliencymap_3_8_fc.pt"
pt_model = FairHalfNet()

pt_model.load_state_dict(torch.load(network_file_pt, map_location=torch.device("cpu")))

# for layer,param in pt_model.state_dict().items(): # param is weight or bias(Tensor)
#     if "bias" in layer:
#         print(layer)
#         print(param)

# np.set_printoptions(formatter={'float': '{: 0ies('\n')


# def onnx_to_h5(output_path ):
#     '''
#     将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
#     '''
#     onnx_model = onnx.load(output_path)
#     k_model = onnx_to_keras(onnx_model, ['input'])
#     keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)    #第二个参数是新的.h5模型的保存地址及文件名
#     # 下面内容是加载该模型，然后将该模型的结构打印出来
#     model = tf.keras.models.load_model('kerasModel.h5')
#     model.summary()
#     print(model)

model_path = "../../lrp_toolbox/models/MNIST/trail"

lrpmodel = read(model_path + ".txt", "txt")
inp = [[0, 0, 1, 0, 1, 0.72, 0, 0, 1, 0, 1, 0, 0, 1, 0.33, 1, 0]]
subject_layer = -1
num_rel = 1
# relevant_neurons, least_relevant_neurons, total_R = find_relevant_neurons(lrpmodel, inp, subject_layer, num_rel)

# print(relevant_neurons)
# print(least_relevant_neurons)
# print(total_R)

criterion = nn.CrossEntropyLoss()
inp_tensor = torch.tensor(inp)
outputs = pt_model(inp_tensor)
target_list = [1]
targets = torch.Tensor(target_list).long()
print(targets)
print(outputs)
loss = criterion(outputs, targets)
loss.backward()

# middle_gradient = 0
# all_tensor = torch.zeros(17)
# for name, params in pt_model.named_parameters():
#     # print(name)
#     if "weight" in name and "output" in name:
#         print("output weight: ")
#         print(params)
#         middle_gradient = float(params[target_list[0]][relevant_neurons[0]])
#     if "input" in name and "weight" in name:
#         # print(name)
#         # print(params)
#         print(params.grad)
#         for grad in params.grad:
#             all_tensor += grad
#             print(grad)
#             print(all_tensor)

# print(all_tensor)
# print(middle_gradient)
# all_tensor /= middle_gradient
# print(all_tensor)
# print(int(all_tensor.argmax(dim=0)))

# print(pt_model.fc3.weight.retain_grad())
