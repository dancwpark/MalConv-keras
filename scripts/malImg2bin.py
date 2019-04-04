import numpy as np
import os
import sys
from utils import *

# MalImg data path
datapath = "/home/dancwpark/data/malimg.npz"
dataset = np.load(datapath)
features, labels = load_data(dataset=dataset, standardize=False)

# Save file path
savePre = "/home/dancwpark/data/malbin/"
csvFile = "/home/dancwpark/data/malbin.csv"

# Data for csv file
csvData = []

# For each file in features, change the file to a string of hex bytes
for i, img in enumerate(features):
    binArray = toBinaryArray(img)
    hexArray = binaryToHexArray(binArray)
    hexString = "".join(hexArray)
    hexData = hexString.decode("hex")
    filename = savePre+str(i)+"_"+str(labels[i])+"_040419"
    with open(filename, 'wb') as f:
        f.write(hexData)
    csvData.append(filename+", "+str(labels[i]))

with open(csvFile, 'w') as f:
    for line in csvData:
        f.write(line+"\n")
