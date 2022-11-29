from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_num = 50
input_dimension = 17
output_dimension = 2


class FairNet(nn.Module):
    def __init__(self):
        super(FairNet, self).__init__()
        self.fc_input = nn.Linear(input_dimension, node_num)
        self.fc1 = nn.Linear(node_num, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc_output = nn.Linear(node_num, output_dimension)

        self.fc_input_prop = nn.Linear(input_dimension, node_num)
        self.fc1_prop = nn.Linear(node_num, node_num)
        self.fc2_prop = nn.Linear(node_num, node_num)
        self.fc3_prop = nn.Linear(node_num, node_num)
        self.fc_output_prop = nn.Linear(node_num, output_dimension)

    def forward(self, x, x_prop):

        # For network under verified
        x = self.fc_input(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc_output(x)
        output = x

        # for property network
        x_prop = self.fc_input_prop(x_prop)
        x_prop = F.relu(x_prop)
        x_prop = self.fc1_prop(x_prop)
        x_prop = F.relu(x_prop)
        x_prop = self.fc2_prop(x_prop)
        x_prop = F.relu(x_prop)
        x_prop = self.fc3_prop(x_prop)
        x_prop = F.relu(x_prop)
        x_prop = self.fc_output_prop(x_prop)
        output_prop = x_prop

        return output, output_prop


class FairHalfNet(nn.Module):
    def __init__(self):
        # For Computing Saliency Map
        super(FairHalfNet, self).__init__()
        self.fc_input = nn.Linear(input_dimension, node_num)
        self.fc1 = nn.Linear(node_num, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc_output = nn.Linear(node_num, output_dimension)

    def forward(self, x):

        # For network under verified
        x = self.fc_input(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc_output(x)
        output = x

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device)
        optimizer.zero_grad()

        output, output_prop = model(data, data)
        target_list = [1 if item == 1 else 0 for item in target]
        target_tensor = torch.Tensor(target_list).long()
        loss = criterion(output, target_tensor)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.long().view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            print(
                "Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    loss,
                    correct,
                    len(train_loader.dataset),
                    100.0 * correct / len(train_loader.dataset),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output, output_prop = model(data, data)

            test_loss += criterion(output, target.long()).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            # print(data)
            # print(target.long())
            # print(output)
            # print(pred)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def parse_arguments():
    """
    Parse command line argument
    :return: args
    """

    # define the program description
    text = "PyTorch Training DNN"

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        metavar="M",
        help="Learning rate step gamma (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--datasettype",
        type=str,
        default="fair",
        metavar="fair",
        help="For which type of dataset: fair, bias, original (default: fair)",
    )
    parser.add_argument(
        "--serialnum",
        type=str,
        default="nonexp",
        metavar="nonexp",
        help="serial number for runing experiment",
    )
    parser.add_argument(
        "--rqfolder",
        type=str,
        default="nonrq",
        metavar="nonrq",
        help="research question asking",
    )

    # parse command-line arguments
    args = parser.parse_args()

    return args


def main():
    # Training settings
    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # read from dataset
    data = pd.read_csv(
        "../../TrainingData/German/german-" + str(args.datasettype) + ".csv",
        header=None,
        sep=";",
    )

    # Normalization for int data
    int_columns = data.dtypes == "int64"
    int_columns = list(int_columns[int_columns].index)
    for col_name in int_columns:
        # The label data
        if col_name == 8:
            break
        Scaler = MinMaxScaler(feature_range=(0, 1))
        col_value = np.array(data[col_name]).reshape(-1, 1)
        new_col = Scaler.fit_transform(col_value)
        data[col_name] = new_col

    # One hot encoding
    row = data
    d = {}
    for index in row:
        temp = row[index]
        if temp.dtype != float and temp.dtype != int:
            keys = list(set(data[index]))
            values = range(len(keys))
            d.update(
                dict(
                    zip(
                        keys,
                        torch.nn.functional.one_hot(
                            torch.tensor(list(values)), len(keys)
                        ).tolist(),
                    )
                )
            )

    data = data.applymap(lambda x: d[x] if type(x) != float and type(x) != int else x)

    data = np.array(data)

    final_data = data[:, :8]
    final_label = data[:, 8]

    final_label[final_label > 1] = 0

    final_list_data = []
    for row in final_data:
        flatList = []
        for i in row:
            if type(i) == int or type(i) == float:
                flatList.append(i)
            else:
                flatList.extend(i)
        final_list_data.append(flatList)

    dataset = Data.TensorDataset(
        torch.tensor(final_list_data), torch.tensor(final_label.astype(int))
    )

    # split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = FairNet().to(device)
    model_half = FairHalfNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    data, labels = next(iter(train_loader))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # print([x.grad for x in optimizer.param_groups[0]["params"]])

    if args.save_model:

        # Copy for the second network
        dict_params_copy = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if name == "fc_input_prop.weight":
                arr = dict_params_copy["fc_input.weight"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc_input_prop.bias":
                arr = dict_params_copy["fc_input.bias"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc1_prop.weight":
                arr = dict_params_copy["fc1.weight"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc1_prop.bias":
                arr = dict_params_copy["fc1.bias"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc2_prop.weight":
                arr = dict_params_copy["fc2.weight"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc2_prop.bias":
                arr = dict_params_copy["fc2.bias"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc3_prop.weight":
                arr = dict_params_copy["fc3.weight"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc3_prop.bias":
                arr = dict_params_copy["fc3.bias"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc_output_prop.weight":
                arr = dict_params_copy["fc_output.weight"].data
                dict_params_copy[name].data.copy_(arr)
            elif name == "fc_output_prop.bias":
                arr = dict_params_copy["fc_output.bias"].data
                dict_params_copy[name].data.copy_(arr)

        model_copy = FairNet()
        model_copy.load_state_dict(dict_params_copy, strict=False)

        prefix_folder = "../../networks/german/"
        if args.rqfolder != "nonrq":
            prefix_folder += args.rqfolder + "/" + args.datasettype + "/"
        torch.save(
            model_copy.state_dict(),
            prefix_folder
            + "german_"
            + args.datasettype
            + "_"
            + args.serialnum
            + "_3_"
            + str(node_num)
            + "_fc.pt",
        )

        # Save model for marabou
        dummy_input = torch.randn(input_dimension)
        dummy_input_prop = torch.randn(input_dimension)
        torch.onnx.export(
            model_copy,
            (dummy_input, dummy_input_prop),
            prefix_folder
            + "german_"
            + str(args.datasettype)
            + "_"
            + args.serialnum
            + "_3_"
            + str(node_num)
            + "_fc.onnx",
            input_names=["input_nuv", "input_prop"],
            output_names=["output_nuv", "output_prop"],
            export_params=True,
        )

        ########################################
        # copy weight for half-model for saliency map
        ########################################

        dict_params_ori = dict(model.named_parameters())
        dict_params_half = dict(model_half.named_parameters())
        for name, param in model_half.named_parameters():
            if name == "fc_input.weight":
                arr = dict_params_ori["fc_input.weight"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc_input.bias":
                arr = dict_params_ori["fc_input.bias"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc1.weight":
                arr = dict_params_ori["fc1.weight"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc1.bias":
                arr = dict_params_ori["fc1.bias"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc2.weight":
                arr = dict_params_ori["fc2.weight"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc2.bias":
                arr = dict_params_ori["fc2.bias"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc3.weight":
                arr = dict_params_ori["fc3.weight"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc3.bias":
                arr = dict_params_ori["fc3.bias"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc_output.weight":
                arr = dict_params_ori["fc_output.weight"].data
                dict_params_half[name].data.copy_(arr)
            elif name == "fc_output.bias":
                arr = dict_params_ori["fc_output.bias"].data
                dict_params_half[name].data.copy_(arr)

        model_half_copy = FairHalfNet()
        model_half_copy.load_state_dict(dict_params_half, strict=False)

        torch.save(
            model_half_copy.state_dict(),
            prefix_folder
            + "german_"
            + str(args.datasettype)
            + "_"
            + args.serialnum
            + "_saliencymap_3_"
            + str(node_num)
            + "_fc.pt",
        )

        #######################################
        # Save LRP txt file for deepimportance analysis
        ########################################

        lrp_txtfile = (
            prefix_folder
            + "german_"
            + str(args.datasettype)
            + "_"
            + args.serialnum
            + "_lrp_3_"
            + str(node_num)
            + "_fc.txt"
        )

        with open(lrp_txtfile, mode="w") as file:
            for layer, param in model_half_copy.state_dict().items():
                if "weight" in layer:
                    if "input" in layer:
                        file.writelines(
                            "Linear "
                            + str(input_dimension)
                            + " "
                            + str(node_num)
                            + "\n"
                        )
                    elif "output" in layer:
                        file.writelines(
                            "Rect\nLinear "
                            + str(node_num)
                            + " "
                            + str(output_dimension)
                            + "\n"
                        )
                    else:
                        file.writelines(
                            "Rect\nLinear " + str(node_num) + " " + str(node_num) + "\n"
                        )
                    tmp_list = []
                    for line in param.numpy():
                        tmp_list.extend(line)
                    for weight in tmp_list:
                        file.writelines(str(weight) + " ")
                    file.writelines("\n")
                elif "bias" in layer:
                    tmp_list = []
                    for line in param.numpy():
                        tmp_list.append(line)
                    for bias in tmp_list:
                        file.writelines(str(bias) + " ")
                    file.writelines("\n")


if __name__ == "__main__":
    main()
