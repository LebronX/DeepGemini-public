import subprocess

query = 2
for datasettype in ["bias", "fair"]:
    for node in [10, 20, 30, 40, 50, 60]:
        for i in range(1, 10):
            command = "python3 run-test.py --time 600 --prop cegfa --debug --record True --cegfa random --rqfolder rq1"
            command = command + " --query " + str(query)
            command = command + " --datasettype " + datasettype
            command = command + " --node " + str(node)
            command = command + " --serialnum exp" + str(i)
            print(command)
            # Popen: Parallel; Call: Sequential
            p = subprocess.call(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
    # for line in p.stdout.readlines():
    #     print(line)

# Training Script
# python3 german_fairness_training.py --lr 1.1 --gamma 0.7 --epochs 10 --datasettype fair --rqfolder rq3 --save-model --serialnum exp1 --seed 1
