import numpy as np
import os


dir = "./Data/GraphData/"
files = os.listdir(dir)

for file in files:
    data = open(dir + file, "r").read()
    data = data.splitlines()
    data = data[1:]

    arr = np.zeros([len(data),2])

    for i in range(len(data)):
        line = data[i].split(",")
        print(line[1])
        arr[i][0] = line[1]
        arr[i][1] = line[2]

    name = file.split(".")[0]
    np.save(dir + name, arr)
