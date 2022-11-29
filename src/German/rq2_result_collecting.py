import pandas as pd
import matplotlib.pyplot as plt


datasettype = ["bias"]  # "bias", "fair"
query = ["query_0"]
infix = "_CEGFA_bias_"  # "_CEGFA_bias_", "_CEGFA_fair_"
node = [20]  # 15
exp = "_exp"
method = ["_random", "_deepimportance", "_saliencymap"]
epsilon = [0.1, 0.2, 0.5, 1.0]


for dt in datasettype:
    for q in query:
        for n in node:
            all_method_list = []
            for m in method:
                if m != "_random":
                    for ep in epsilon:
                        total_list = [0 for i in range(1800)]
                        avg_num = 9
                        for e in range(1, 10):
                            iter = 0
                            fileName = (
                                "../../validate/german/rq2/"
                                + dt
                                + "/"
                                + q
                                + infix
                                + str(n)
                                + exp
                                + str(e)
                                + m
                                + "_"
                                + str(ep)
                                + ".csv"
                            )
                            print(fileName)
                            df = pd.read_csv(fileName)
                            if df["FairRegionPercent"][0] == 1.0:
                                avg_num -= 1
                                print("total fair: " + fileName)
                                continue
                            
                            # for region, time in zip(
                            #     df["FairRegionPercent"], df["TimeTaken"]
                            # ):
                            #     if time > iter and iter < 1800:
                            #         total_list[iter] += region
                            #         iter += 1

                            i = 0
                            for iter in range(1800):
                                if iter % 2 != 0 and iter > 0:
                                    total_list[iter] = total_list[iter-1]
                                else:
                                    total_list[iter] += df["FairRegionPercent"][i]
                                    i += 1
                                iter += 1
                        # print(my_list)
                        my_list = [x / avg_num for x in total_list]
                        # print(my_list)
                        # print(avg_num)
                        all_method_list.append(my_list)
                else:
                    total_list = [0 for i in range(1800)]
                    avg_num = 9
                    for e in range(1, 10):
                        iter = 0
                        fileName = (
                            "../../validate/german/rq2/"
                            + dt
                            + "/"
                            + q
                            + infix
                            + str(n)
                            + exp
                            + str(e)
                            + m
                            + ".csv"
                        )
                        print(fileName)
                        df = pd.read_csv(fileName)
                        if df["FairRegionPercent"][0] == 1.0:
                            avg_num -= 1
                            continue
                        
                        # for region, time in zip(
                        #     df["FairRegionPercent"], df["TimeTaken"]
                        # ):
                        #     if time > iter and iter < 1800:
                        #         total_list[iter] += region
                        #         iter += 1
                        i = 0
                        for iter in range(1800):
                            if iter % 2 != 0 and iter > 0:
                                total_list[iter] = total_list[iter-1]
                            else:
                                total_list[iter] += df["FairRegionPercent"][i]
                                i += 1
                            iter += 1
                        
                    my_list = [x / avg_num for x in total_list]
                    all_method_list.append(my_list)
                # plt.figure(figsize=(10,5))

            time_list = [i for i in range(1800)]
            makeevery = 300
            plt.plot(
                time_list,
                all_method_list[0],
                marker="o",
                markevery=makeevery,
                color="r",
                label="random",
            )
            plt.plot(
                time_list,
                all_method_list[1],
                marker="v",
                markevery=makeevery,
                color="lime",
                label="IN-0.1",
            )
            plt.plot(
                time_list,
                all_method_list[2],
                marker="P",
                markevery=makeevery,
                color="tan",
                label="IN-0.2",
            )
            plt.plot(
                time_list,
                all_method_list[3],
                marker="s",
                markevery=makeevery,
                color="c",
                label="IN-0.5",
            )
            plt.plot(
                time_list,
                all_method_list[4],
                marker="p",
                markevery=makeevery,
                color="indigo",
                label="IN-1.0",
            )
            plt.plot(
                time_list,
                all_method_list[5],
                marker="*",
                markevery=makeevery,
                color="violet",
                label="SM-0.1",
            )
            plt.plot(
                time_list,
                all_method_list[6],
                marker="H",
                markevery=makeevery,
                color="navy",
                label="SM-0.2",
            )
            plt.plot(
                time_list,
                all_method_list[7],
                marker="^",
                markevery=makeevery,
                color="peru",
                label="SM-0.5",
            )
            plt.plot(
                time_list,
                all_method_list[8],
                marker="D",
                markevery=makeevery,
                color="brown",
                label="SM-1.0",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Fairness Score (%)")
            plt.legend()
            plt.grid(True)
            plt.savefig("../../validate/german/fig/rq2/german_20_bias.pdf", dpi=300)
            plt.show()


# query_1_CEGFA_bias_10_exp1_random
# Training Script
# python3 compas_fairness_training.py --lr 1.1 --gamma 0.7 --epochs 10 --datasettype fair --rqfolder rq2 --save-model --serialnum exp1 --seed 1
