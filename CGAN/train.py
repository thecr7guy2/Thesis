import torch
import torch.nn as nn
import torch.nn.functional as F


# creating a method to onehot encode the labels
def one_hot(labels, n_classes):
    return F.one_hot(labels, n_classes)


###################
### test method ###
# labels = torch.arange(8)  # 8 because of the batch size
# n_classes = 10
# test = one_hot(labels, n_classes)
# print(test)
####################

