import subprocess

query = 0
time = 1800
cegfa = "random"  # random / saliencymap / deepimportance

for node in [200]:  # 80, 100, 150, 200
    for i in range(1, 11):
        command = (
            "python3 run-test.py --prop cegfa --debug --record True --rqfolder rq3"
        )
        command = command + " --cegfa " + cegfa
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
