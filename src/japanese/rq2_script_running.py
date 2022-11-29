import subprocess

query = 1
time = 1800
cegfa = "deepimportance"  # random / saliencymap / deepimportance
epsilon = [0.1, 0.2, 0.5, 1.0]  #

for e in epsilon:
    for node in [10]:  # 10, 30
        for i in range(1, 10):
            command = (
                "python3 run-test.py --prop cegfa --debug --record True --rqfolder rq2"
            )
            command = command + " --cegfa " + cegfa
            if cegfa == "saliencymap":
                command = command + " --smepsilon " + str(e)
            if cegfa == "deepimportance":
                command = command + " --diepsilon " + str(e)
            command = command + " --time " + str(time)
            command = command + " --query " + str(query)
            command = command + " --node " + str(node)
            command = command + " --serialnum exp" + str(i)
            command += " >/dev/null 2>&1 "
            print(command)
            # Popen: Parallel; Call: Sequential
            p = subprocess.call(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
    # for line in p.stdout.readlines():
    #     print(line)

# Training Script
# python3 japanese_fairness_training.py --lr 1.1 --gamma 0.7 --epochs 10 --datasettype fair --rqfolder rq2 --save-model --serialnum exp1 --seed 1
