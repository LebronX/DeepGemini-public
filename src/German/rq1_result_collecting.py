import pandas as pd

datasettype = ["bias", "fair"]
query = ["query_1", "query_2"]
infix = "_CEGFA_"
node = [10, 20, 30, 40, 50, 60]
exp = "_exp"
posfix = "_random.csv"

for dt in datasettype:
    for q in query:
        for n in node:
            avg_time = 0
            for e in range(1, 10):
                fileName = (
                    "../../validate/german/rq1/"
                    + dt
                    + "/"
                    + q
                    + infix
                    + dt
                    + "_"
                    + str(n)
                    + exp
                    + str(e)
                    + posfix
                )
                df = pd.read_csv(fileName)
                avg_time += df["TimeTaken"]
            avg_time /= 9
            print(fileName)
            print(avg_time)


# query_1_CEGFA_bias_10_exp1_random
# Training Script
# python3 compas_fairness_training.py --lr 1.1 --gamma 0.7 --epochs 10 --datasettype fair --rqfolder rq2 --save-model --serialnum exp1 --seed 1
