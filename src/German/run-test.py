from marabou_encoding import marabouEncoding
import time
import csv
import argparse
import random
import collections
import copy
import numpy as np
import itertools
import torch
import torch.nn as nn
from advertorch.attacks import JacobianSaliencyMapAttack
from german_fairness_training import FairHalfNet
from lrp_toolbox.model_io import write, read


# Information for noting the results in csv format
veri_info = [
    "Specification",
    "NetworkFile",
    "Epsilon",
    "Delta",
    "TimeTaken",
    "Counterexample",
    "FairRegion",
    "FairRegionPercent",
    "DebugCurRegion",
    "JSMADimension",
    "DeepimportanceDimension",
]
veri_dict = {i: None for i in veri_info}


def write_csv_header(csvfile):
    with open(csvfile, mode="w") as file:
        writer = csv.DictWriter(file, fieldnames=veri_info)
        writer.writeheader()


def write_line_csv(csvfile):
    with open(csvfile, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=veri_info)
        writer.writerow(veri_dict)


def checking_fairness(prop, mara, network_file, query):
    log_time = time.time()
    counterExample = mara.checkProperties(
        prop=prop,
        networkFile=network_file,
        query=query,
    )
    mara_time = time.time() - log_time
    return mara_time, counterExample


# def filter_region(cur_region, sensitive_feature, total_fair_region, input_dimension):
####################
# For filter repetitive, contained, containing region
# Intersection will not appear
####################

# repetitive region
# if cur_region in total_fair_region:
#     return total_fair_region
# else:
#     total_fair_region.append(cur_region)
#     return total_fair_region

# Compare with every region
# total_flag = False
# for fair_region in total_fair_region:
#     region_flag = False
#     being_contained_flag = False
#     contain_flag = False
#     for i in range(input_dimension):
#         if cur_region[i] == fair_region[i]:
#             continue

#         if i in [0, 3, 4]:
#             fair_region_closed_space_cnt = fair_region[i].count(-1)
#             cur_region_closed_space_cnt = cur_region[i].count(-1)
#             if cur_region_closed_space_cnt == 0 and fair_region_closed_space_cnt == 2:
#                 # contain fair region
#                 contain_flag = True
#             elif cur_region_closed_space_cnt == 2 and fair_region_closed_space_cnt == 0:
#                 # being contained
#                 being_contained_flag = True
#             else:
#                 # different open dimension, go to next fair region
#                 break

#             if contain_flag and being_contained_flag

#         elif i in [1, 7]:
#         elif i in [2, 6]:

#         if cur_region[i] == fair_region[i]:

# filter_fair_region = []
# for i in range(len(cur_region)):
#     # if i == sensitive_feature:
#     #     continue
#     for j in cur_region[i]:
#         filter_fair_region.append(j)

# if not filter_fair_region in total_fair_region:
#     total_fair_region.append(filter_fair_region)

# return total_fair_region


def random_split_region(cur_region, input_dimension):
    split_dimension = random.randint(0, 7)
    while (
        (split_dimension in [0, 3, 4] and cur_region[split_dimension].count(-1) == 2)
        or (split_dimension in [1, 7] and cur_region[split_dimension].count(-1) == 1)
        or split_dimension == 5
    ):  # unsplittable and re-choose
        split_dimension = random.randint(0, 7)

    new_region, other_region = split_region(
        input_dimension, split_dimension, copy.deepcopy(cur_region)
    )

    return new_region, other_region


def saliency_split_dimension(adversary, counterExample, total_input_dimension):

    # post-process raw marabou counterexample
    # Two counterexamples
    # raw counterExample is a set
    tmp_ce_list1 = []
    tmp_ce_list2 = []
    for i in range(total_input_dimension):
        tmp_ce_list1.append(counterExample[i])
    for i in range(total_input_dimension, 2 * total_input_dimension):
        tmp_ce_list2.append(counterExample[i])

    # Don't know which label the ce have, so have to try
    ce1_0_adv_untargeted = adversary.perturb(
        torch.tensor([tmp_ce_list1]), torch.tensor([[0]])
    )
    ce1_1_adv_untargeted = adversary.perturb(
        torch.tensor([tmp_ce_list1]), torch.tensor([[1]])
    )
    ce2_0_adv_untargeted = adversary.perturb(
        torch.tensor([tmp_ce_list2]), torch.tensor([[0]])
    )
    ce2_1_adv_untargeted = adversary.perturb(
        torch.tensor([tmp_ce_list2]), torch.tensor([[1]])
    )

    ce1_0_adv_untargeted_list = ce1_0_adv_untargeted.tolist()[0]
    ce1_1_adv_untargeted_list = ce1_1_adv_untargeted.tolist()[0]
    ce2_0_adv_untargeted_list = ce2_0_adv_untargeted.tolist()[0]
    ce2_1_adv_untargeted_list = ce2_1_adv_untargeted.tolist()[0]

    # float to int for comparison
    tmp_ce_list1_int = [int(x) for x in tmp_ce_list1]
    tmp_ce_list2_int = [int(x) for x in tmp_ce_list2]
    ce1_0_adv_untargeted_list_int = [int(x) for x in ce1_0_adv_untargeted_list]
    ce1_1_adv_untargeted_list_int = [int(x) for x in ce1_1_adv_untargeted_list]
    ce2_0_adv_untargeted_list_int = [int(x) for x in ce2_0_adv_untargeted_list]
    ce2_1_adv_untargeted_list_int = [int(x) for x in ce2_1_adv_untargeted_list]

    split_dimension_list = []
    if tmp_ce_list1_int != ce1_0_adv_untargeted_list_int:
        for i in range(len(tmp_ce_list1_int)):
            if tmp_ce_list1_int[i] != ce1_0_adv_untargeted_list_int[i]:
                split_dimension_list.append(i)
    else:
        for i in range(len(tmp_ce_list1_int)):
            if tmp_ce_list1_int[i] != ce1_1_adv_untargeted_list_int[i]:
                split_dimension_list.append(i)

    if tmp_ce_list2_int != ce2_0_adv_untargeted_list_int:
        for i in range(len(tmp_ce_list2_int)):
            if tmp_ce_list2_int[i] != ce2_0_adv_untargeted_list_int[i]:
                split_dimension_list.append(i)
    else:
        for i in range(len(tmp_ce_list2_int)):
            if tmp_ce_list2_int[i] != ce2_1_adv_untargeted_list_int[i]:
                split_dimension_list.append(i)

    return split_dimension_list


def compute_region_percentage(cur_region, total_fair_region_set):
    """Compute Region Percentage

    Will filter contain, contained, intersection region

    """
    total_space_size = 10800000  # 3*3*3*2*2*1000*100
    dimension_2_helper_list = [
        i for i in range(int(cur_region[2][0] * 1000), int(cur_region[2][1] * 1000))
    ]
    dimension_6_helper_list = [
        i for i in range(int(cur_region[6][0] * 100), int(cur_region[6][1] * 100))
    ]

    cur_region_cover_space = {
        d
        for d in itertools.product(
            cur_region[0],
            cur_region[1],
            dimension_2_helper_list,
            cur_region[3],
            cur_region[4],
            dimension_6_helper_list,
            cur_region[7],
        )
        if -1 not in d
    }

    new_fair_space = total_fair_region_set.union(cur_region_cover_space)
    fair_region_percent = len(new_fair_space) / total_space_size

    return fair_region_percent, new_fair_space


def split_region(input_dimension, split_dimension, cur_region):

    """Split Region into two new regions, can be bisect or trisect"""

    new_region = []
    other_dimension = []
    for i in range(input_dimension):  # split and produce two regions
        if i == split_dimension:
            if i in [0, 3, 4]:  # trisect
                tmp_sub_region = [0, 1, 2]  # cur_region[i]
                # rand_open = random.randint(0, 2)
                # tmp_sub_region[rand_open] = rand_open  # Prevent intersection of regions
                if cur_region[i].count(-1) == 0:
                    rand_closed = random.randint(0, 2)
                    tmp_sub_region[rand_closed] = -1
                    other_dimension = [-1, -1, -1]
                    other_dimension[rand_closed] = rand_closed
                elif cur_region[i].count(-1) == 1:
                    rand_closed = random.randint(0, 2)
                    while tmp_sub_region[rand_closed] == -1:
                        rand_closed = random.randint(0, 2)
                    tmp_sub_region[rand_closed] = -1
                    other_dimension = [-1, -1, -1]
                    other_dimension[rand_closed] = rand_closed
                new_region.append(tmp_sub_region)
            elif i in [1, 7]:  # bisect
                tmp_sub_region = cur_region[i]
                rand_closed = random.randint(0, 1)
                tmp_sub_region[rand_closed] = -1
                new_region.append(tmp_sub_region)
                other_dimension = [0, 1]
                other_dimension[1 - rand_closed] = -1
            else:  # bisect
                middle = (cur_region[i][0] + cur_region[i][1]) / 2
                new_region.append([cur_region[i][0], middle])

                other_dimension = [middle, cur_region[i][1]]
        else:  # unchanged dimension
            new_region.append(cur_region[i])
    # append two new-produced regions
    other_region = copy.deepcopy(new_region)
    other_region[split_dimension] = other_dimension

    return new_region, other_region


def find_relevant_neurons(lrpmodel, inps, subject_layer, num_rel, lrpmethod=None):

    totalR = None
    cnt = 0
    for inp in inps:
        cnt += 1
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        # prepare initial relevance to reflect the model's dominant prediction
        # (ie depopulate non-dominant output neurons)
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


def deepimportance_split_dimension(
    pt_model,
    lrp_model,
    num_rel,
    counterexample,
    total_input_dimension,
    subject_layer,
):
    """
    Select splitting dimension based on deepimportance
    """

    # post-process raw marabou counterexample
    # Two counterexamples
    # raw counterExample is a set
    tmp_ce_list = [[], []]
    for i in range(total_input_dimension):
        tmp_ce_list[0].append(counterexample[i])
    for i in range(total_input_dimension, 2 * total_input_dimension):
        tmp_ce_list[1].append(counterexample[i])

    # subject layer typically select the penultimate layer according to deepimportance
    relevant_neurons, least_relevant_neurons, total_R = find_relevant_neurons(
        lrp_model, tmp_ce_list, subject_layer, num_rel
    )

    # Randomly pick a dimension to compute important neuron
    criterion = nn.CrossEntropyLoss()
    inp_tensor_1 = torch.tensor(tmp_ce_list[0])
    inp_tensor_2 = torch.tensor(tmp_ce_list[1])
    dice = random.randint(0, 1)
    if dice == 0:
        output_1 = pt_model(inp_tensor_1)
        target_1 = torch.Tensor([0]).long()
        loss_1 = criterion(output_1.unsqueeze(dim=0), target_1)
        loss_1.backward()
    elif dice == 1:
        output_2 = pt_model(inp_tensor_2)
        target_2 = torch.Tensor([1]).long()
        loss_2 = criterion(output_2.unsqueeze(dim=0), target_2)
        loss_2.backward()

    middle_gradient = 0
    all_tensor = torch.zeros(total_input_dimension)
    for name, params in pt_model.named_parameters():
        if "weight" in name and "output" in name:
            middle_gradient = float(params[dice][relevant_neurons[0]])
        if "input" in name and "weight" in name:
            for grad in params.grad:
                all_tensor += grad

    all_tensor /= middle_gradient
    split_dimension = int(all_tensor.argmax(dim=0))
    return split_dimension


def parse_arguments():
    """
    Parse command line argument
    :return: args
    """

    # define the program description
    text = "DeepGemini"

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument(
        "--node",
        type=int,
        default=10,
        metavar="N",
        help="Hidden layer size (default: 10)",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=600,
        metavar="N",
        help="Experiment time (default: 600)",
    )
    parser.add_argument(
        "--prop",
        type=str,
        default="cegfa",
        help="Verifying property (default: cegfa): fairness, cegfa",
    )
    parser.add_argument(
        "--cegfa",
        type=str,
        default="random",
        help="Refinement policy (default: random): random, saliencymap, deepimportance",
    )
    parser.add_argument(
        "--smepsilon",
        type=float,
        default=0.1,
        metavar="N",
        help="Epsilon-greedy for saliency map ratio (default: 0.1)",
    )
    parser.add_argument(
        "--diepsilon",
        type=float,
        default=0.1,
        metavar="N",
        help="Epsilon-greedy for deepimportance ratio (default: 0.1)",
    )
    parser.add_argument(
        "--datasettype",
        type=str,
        default="fair",
        metavar="fair",
        help="For which type of dataset (default: fair): fair, bias, original",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="For debugging",
    )
    parser.add_argument(
        "--record",
        type=str,
        default="True",
        help="Recording into csv file",
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
    parser.add_argument(
        "--query",
        type=int,
        default=0,
        metavar=0,
        help="Query 0: Full space, Query 1: Is there bias with respect to gender for people holding a house? Query 2: Is there bias with respect to people with no credit taken for female?",
    )

    # parse command-line arguments
    args = parser.parse_args()

    return args


def main():

    args = parse_arguments()

    prop = []
    if args.prop == "fairness":
        prop.append("checking-fairness")
    elif args.prop == "cegfa":
        prop.append("CEGFA")
    network_file = (
        "../../networks/german/"
        + args.rqfolder
        + "/"
        + args.datasettype
        + "/german_"
        + args.datasettype
        + "_"
        + args.serialnum
        + "_3_"
        + str(args.node)
        + "_fc.onnx"
    )
    csv_cegfa_posfix = ""
    if args.cegfa == "random":
        csv_cegfa_posfix = args.cegfa
        pt_model = FairHalfNet()  # useless variable
    elif args.cegfa == "saliencymap":
        print("-----------Loading model for saliency map analysis----------")
        network_file_pt = (
            "../../networks/german/"
            + args.rqfolder
            + "/"
            + args.datasettype
            + "/german_"
            + args.datasettype
            + "_"
            + args.serialnum
            + "_saliencymap"
            + "_3_"
            + str(args.node)
            + "_fc.pt"
        )
        pt_model = FairHalfNet()
        pt_model.load_state_dict(
            torch.load(network_file_pt, map_location=torch.device("cpu"))
        )
        pt_model.eval()
        print("-----------Loading finish----------")
        csv_cegfa_posfix = args.cegfa + "_" + str(args.smepsilon)
    elif args.cegfa == "deepimportance":
        print("-----------Loading model for deepimportance analysis----------")
        network_file_lrp = (
            "../../networks/german/"
            + args.rqfolder
            + "/"
            + args.datasettype
            + "/german_"
            + args.datasettype
            + "_"
            + args.serialnum
            + "_lrp"
            + "_3_"
            + str(args.node)
            + "_fc.txt"
        )
        lrp_model = read(network_file_lrp, "txt")
        network_file_pt = (
            "../../networks/german/"
            + args.rqfolder
            + "/"
            + args.datasettype
            + "/german_"
            + args.datasettype
            + "_"
            + args.serialnum
            + "_saliencymap"
            + "_3_"
            + str(args.node)
            + "_fc.pt"
        )
        pt_model = FairHalfNet()
        pt_model.load_state_dict(
            torch.load(network_file_pt, map_location=torch.device("cpu"))
        )
        pt_model.eval()
        csv_cegfa_posfix = args.cegfa + "_" + str(args.diepsilon)
        print("-----------Loading finish----------")

    csvfile = (
        "../../validate/german/"
        + args.rqfolder
        + "/"
        + args.datasettype
        + "/query_"
        + str(args.query)
        + "_"
        + prop[0]
        + "_"
        + args.datasettype
        + "_"
        + str(args.node)
        + "_"
        + args.serialnum
        + "_"
        + csv_cegfa_posfix
        + ".csv"
    )
    mara = marabouEncoding()

    # initial region for: account (qualitative), credit history (qualitative),
    # credit amount (quatitative), saving (qualitative), employment (qualitative),
    # geneder (qualitative), residence (quatitative), house (qualitative)
    # Query 1: Is there bias with respect to gender for people holding a house?
    # Query 2: Is there bias with respect to credit taken history for female?

    init_region = []
    if args.query == 0:
        init_region = [
            [0, 1, 2],  # account, 3
            [0, 1],  # credit history, 2
            [0, 1],  # credit amount, 1
            [0, 1, 2],  # saving, 3
            [0, 1, 2],  # employment, 3
            [0, 1],  # XXX Gender, 2
            [0, 1],  # residence, 1
            [0, 1],  # house, 2
        ]
    elif args.query == 1:
        init_region = [
            [0, 1, 2],  # account, 3
            [0, 1],  # credit history, 2
            [0, 1],  # credit amount, 1
            [0, 1, 2],  # saving, 3
            [0, 1, 2],  # employment, 3
            [0, 1],  # XXX Gender, 2
            [0, 1],  # residence, 1
            [-1, 1],  # house, 2
        ]
    elif args.query == 2:
        init_region = [
            [0, 1, 2],  # account, 3
            [0, 1],  # XXX credit history, 2
            [0, 1],  # credit amount, 1
            [0, 1, 2],  # saving, 3
            [0, 1, 2],  # employment, 3
            [0, -1],  # Gender, 2
            [0, 1],  # residence, 1
            [0, 1],  # house, 2
        ]
    prop.append(init_region)

    ##############################
    ### Verification procedure ###
    ##############################
    if args.record == "True":
        write_csv_header(csvfile)
    veri_dict.update(
        {
            "Specification": prop[0],
            "NetworkFile": network_file,
        }
    )
    if prop[0] == "checking-fairness":
        mara_time, counterExample = checking_fairness(prop, mara, network_file)
        print("mara time: " + str(mara_time))
        print(counterExample)
        veri_dict.update(
            {
                "TimeTaken": mara_time,
            }
        )
        if args.record == "True":
            write_line_csv(csvfile)
    elif prop[0] == "CEGFA":
        iter = 0
        # sensitive_feature = 5
        time_total = 0
        total_fair_region_list = []
        total_fair_region_set = set()
        all_counterexample = []
        input_dimension = 8
        total_input_dimension = 17
        total_fair_region_percent = 0
        region_deque = collections.deque()
        region_deque.append(init_region)
        cur_region = region_deque.popleft()

        # variables for saliency map
        adversary = JacobianSaliencyMapAttack(
            pt_model,
            num_classes=2,
            clip_min=0.0,
            clip_max=1.0,
            loss_fn=nn.CrossEntropyLoss(),
            theta=1.0,
            gamma=1.0,
            comply_cleverhans=False,
        )  # For Saliency map
        # map dimension, for saliency map
        dimension_map = {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 2,
            6: 3,
            7: 3,
            8: 3,
            9: 4,
            10: 4,
            11: 4,
            12: 5,
            13: 5,
            14: 6,
            15: 7,
            16: 7,
        }

        # variables for deepimportance analysis
        num_rel = 1
        subject_layer = -1

        while time_total < args.time:

            # Verification
            mara_time, counterExample = checking_fairness(
                prop, mara, network_file, args.query
            )
            time_total += mara_time

            #############################
            ####### Compute Region ######
            #############################
            if not counterExample:
                # For filter repetitive region
                # total_fair_region = filter_region(cur_region, sensitive_feature, total_fair_region)

                # Compute every time
                # print(cur_region)
                if not cur_region in total_fair_region_list:
                    total_fair_region_list.append(cur_region)
                    (
                        total_fair_region_percent,
                        total_fair_region_set,
                    ) = compute_region_percentage(cur_region, total_fair_region_set)
                    # print(total_fair_region_percent)
            # else:
            #     tmp_ce_list = []
            #     for i in range(2 * total_input_dimension):
            #         tmp_ce_list.append(counterExample[i])
            # XXX ignore for now
            # all_counterexample.append(tmp_ce_list)

            #############################
            ####### Update Region #######
            #############################
            prop.pop()
            if args.cegfa == "random":
                if counterExample:
                    new_region, other_region = random_split_region(
                        copy.deepcopy(cur_region), input_dimension
                    )

                    region_deque.append(new_region)
                    region_deque.append(other_region)
                    region_deque.append(cur_region)  # cyclic queue

            elif args.cegfa == "saliencymap":
                if counterExample:
                    choice = random.uniform(0, 1)
                    if choice < args.smepsilon:
                        # epsilon-explotation
                        split_dimension_list = saliency_split_dimension(
                            adversary, counterExample, total_input_dimension
                        )

                        flag_split = False
                        for split_dimension in set(split_dimension_list):
                            # Can be unsplittable, filtering at first
                            split_dimension = dimension_map[split_dimension]
                            if (
                                (
                                    split_dimension in [0, 3, 4]
                                    and cur_region[split_dimension].count(-1) == 2
                                )
                                or (
                                    split_dimension in [1, 7]
                                    and cur_region[split_dimension].count(-1) == 1
                                )
                                or split_dimension == 5
                            ):  # unsplittable and skip
                                continue

                            new_region, other_region = split_region(
                                input_dimension,
                                split_dimension,
                                copy.deepcopy(cur_region),
                            )
                            region_deque.append(new_region)
                            region_deque.append(other_region)
                            region_deque.append(cur_region)
                            flag_split = True
                        if not flag_split:
                            # Exploration step
                            new_region, other_region = random_split_region(
                                copy.deepcopy(cur_region), input_dimension
                            )

                            region_deque.append(new_region)
                            region_deque.append(other_region)
                            region_deque.append(cur_region)  # cyclic queue
                        if args.debug:
                            veri_dict.update(
                                {
                                    "JSMADimension": set(split_dimension_list),
                                }
                            )

                    else:
                        # Exploration
                        new_region, other_region = random_split_region(
                            copy.deepcopy(cur_region), input_dimension
                        )

                        region_deque.append(new_region)
                        region_deque.append(other_region)
                        region_deque.append(cur_region)  # cyclic queue

            elif args.cegfa == "deepimportance":
                if counterExample:
                    choice = random.uniform(0, 1)
                    if choice < args.diepsilon:
                        # epsilon-explotation
                        split_dimension = deepimportance_split_dimension(
                            pt_model,
                            lrp_model,
                            num_rel,
                            counterExample,
                            total_input_dimension,
                            subject_layer,
                        )
                        split_dimension = dimension_map[split_dimension]
                        if (
                            (
                                split_dimension in [0, 3, 4]
                                and cur_region[split_dimension].count(-1) == 2
                            )
                            or (
                                split_dimension in [1, 7]
                                and cur_region[split_dimension].count(-1) == 1
                            )
                            or split_dimension == 5
                        ):  # unsplittable
                            # Exploration step
                            new_region, other_region = random_split_region(
                                copy.deepcopy(cur_region), input_dimension
                            )

                            region_deque.append(new_region)
                            region_deque.append(other_region)
                            region_deque.append(cur_region)  # cyclic queue
                        else:
                            new_region, other_region = split_region(
                                input_dimension,
                                split_dimension,
                                copy.deepcopy(cur_region),
                            )
                            region_deque.append(new_region)
                            region_deque.append(other_region)
                            region_deque.append(cur_region)

                        if args.debug:
                            veri_dict.update(
                                {
                                    "DeepimportanceDimension": str(split_dimension),
                                }
                            )

                    else:
                        # Exploration
                        new_region, other_region = random_split_region(
                            copy.deepcopy(cur_region), input_dimension
                        )

                        region_deque.append(new_region)
                        region_deque.append(other_region)
                        region_deque.append(cur_region)

            else:
                print("CEGFA method not found!")
                break

            # If no ce found and deque is empty
            if region_deque:
                # print(region_deque)
                cur_region = region_deque.popleft()
                prop.append(cur_region)
            else:
                prop.append(cur_region)

            veri_dict.update(
                {
                    "Counterexample": all_counterexample,
                    "FairRegion": total_fair_region_list,  # fair region should be list but str, maybe buggy
                    "TimeTaken": time_total,
                    "FairRegionPercent": total_fair_region_percent,
                }
            )

            if args.debug:
                veri_dict.update(
                    {
                        "DebugCurRegion": cur_region,
                    }
                )

            if args.record == "True":
                write_line_csv(csvfile)
            iter += 1

            # if iter == 1:
            #     print(region_deque)
            #     break
            if (
                not counterExample and iter == 1
            ):  # All region are fair, no need to continue
                print(region_deque)
                break


if __name__ == "__main__":
    main()
