"""
Converter pt2libra
======================
"""
import argparse
import torch

from japanese_fairness_training import FairHalfNet


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("pt_model", help="pytorch model to convert")
    parser.add_argument("--serialnum", type=str, default="exp1", help="serialnum")
    parser.add_argument(
        "--node",
        type=int,
        default=5,
        metavar="N",
        help="Hidden layer size (default: 5)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=4,
        metavar="N",
        help="Hidden layer num (default: 4)",
    )
    parser.add_argument(
        "--rqfolder",
        type=str,
        default="rq3",
        metavar="rq3",
        help="research question asking",
    )

    args = parser.parse_args()


    #######################################
    # Save LRP txt file for deepimportance analysis
    ########################################

    prefix_folder = "../../networks/japanese/" + args.rqfolder + "/"
    network_file = (
        prefix_folder
        + "japanese_"
        + args.serialnum
        + "_"
        + "saliencymap_"
        + str(args.layer)
        + "_"
        + str(args.node)
        + "_fc.pt"
    )
    libra_file = (
        prefix_folder
        + "japanese_"
        + args.serialnum
        + "_"
        + str(args.layer)
        + "_"
        + str(args.node)
        + "_fc_libra.py"
    )

    # load_state_dict(torch.load("save.pt"))

    print(libra_file)

    model_half = FairHalfNet()
    model_half.load_state_dict(torch.load(network_file))

    # with open(network_file, mode="r") as file:
    with open(libra_file, mode="w") as libra_file:
        print("", file=libra_file)

        # Assumption for japanese rq3

        print("#", file=libra_file)
        # print("assume(0 <= x03 <= 1)", file=libra_file)
        print("assume(0 <= x04 <= 0)", file=libra_file)
        print("assume(1 <= x05 <= 1)", file=libra_file)
        print("assume(0 <= x06 <= 0)", file=libra_file)
        print("assume(1 <= x07 <= 1)", file=libra_file)
        print("assume(0 <= x08 <= 0)", file=libra_file)
        print("assume(0 <= x09 <= 0)", file=libra_file)
        # print("assume(x10 <= 0.5)", file=libra_file)
        # print("assume(1 <= x11 <= 1)", file=libra_file)
        # print("assume(0 <= x12 <= 0)", file=libra_file)
        # print("assume(1 <= x13 <= 1)", file=libra_file)
        # print("assume(0 <= x14 <= 0)", file=libra_file)
        print("assume(x15 <= 0.5)", file=libra_file)
        print("assume(1 <= x18 <= 1)", file=libra_file)
        print("assume(0 <= x19 <= 0)", file=libra_file)
        print("assume(0 <= x20 <= 0)", file=libra_file)
        print("assume(x21 <= 0.5)", file=libra_file)
        print("assume(x22 <= 0.5)", file=libra_file)
        print("", file=libra_file)
        print("", file=libra_file)


        l = 1
        cur_weight = []
        cur_bias = []
        for layer, param in model_half.state_dict().items():
            print(layer)
            # if not "output" in layer:
            if "weight" in layer:
                print(param)
                for i in range(len(param)):
                    tmp_list = [float('{:.6f}'.format(i)) for i in param[i].tolist()]
                    cur_weight.append(tmp_list)
                # cur_weight = copy.deepcopy(param[0].tolist())
            # print(param[0].tolist()[0])
            elif "bias" in layer:
                # print(len(param))
                tmp_list = [float('{:.6f}'.format(i)) for i in param.tolist()]
                cur_bias = tmp_list
                # print(cur_weight)
                # print(cur_bias)
                for i in range(0, len(cur_bias)):
                    print("x%d%d =" % (l, i), end=" ", file=libra_file)
                    for j in range(0, len(cur_weight[0])):
                        # print(j)
                        # print(i)
                        print("(%f)*x%d%d +" % (cur_weight[i][j], l - 1, j), end=" ", file=libra_file)
                    print("(%f)" % cur_bias[i], file=libra_file)
                # activation
                print("#", file=libra_file)

                if not "output" in layer:
                    for i in range(0, len(cur_bias)):
                        print("ReLU(x%d%d)" % (l, i), file=libra_file)
                    print("", file=libra_file)

                    cur_weight = []
                    cur_bias = []
                    l += 1        


if __name__ == '__main__':
    main()