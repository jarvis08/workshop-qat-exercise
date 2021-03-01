import torch
import torchvision
import numpy as np

"""
pytorch's sequence of params : weight, bias, weight, bias, ...
layer 0 ~ 9 : CONV x 5
layer from 10 : FC x 3
torch need to transpose for Colmajor
Darknet uses Rowmajor(order='C' in TF) in CONV
Darknet uses Colmajor(order='F' in TF) in FC 
"""

model = torchvision.models.alexnet(pretrained=True)                    
with open("../backup/imagenet_alexnet.txt", 'w') as f:
    layer = 0
    #for param in model.parameters():
    for name, param in model.named_parameters():
        if layer < 10:
            print("Parsing CONV layer..")
            flattened = param.data.flatten()
            #np.savetxt(f, flattened, fmt='%f', newline=',') # loading file with delimiter ',' takes much longer time than \n
            #np.savetxt(f, flattened, fmt='%f', delimiter=',')
            np.savetxt(f, flattened, fmt='%f')
        else:
            print("Parsing FC layer..")
            if 'bias' not in name:
                flattened = param.data.transpose(1, 0).flatten()
                np.savetxt(f, flattened, fmt='%f')
            else:
                flattened = param.data.flatten()
                np.savetxt(f, flattened, fmt='%f')
        layer += 1
    f.write('\n')


