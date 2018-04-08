import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_1D(nn.Module):

    def __init__(self, args):
        super(CNN_1D, self).__init__()
        self.args = args

        Ci = 1 # only one input channel to network
        Co = args.kernel_num
        Ks = args.kernel_sizes
        C = args.num_classes

        # Convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv1d(Ci, Co, K) for K in Ks])
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        # Fully connected layers
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        #if self.training:
        #    x = Variable(x)
        # Batch size, number of input channels (1), width of input
        x = x.unsqueeze(1)  # (N, Ci=1, W)

        # Apply relu to all feature maps
        x = [F.relu(conv(x)) for conv in self.convs1]  # [(N, Co, L), ...]*len(Ks)

        # Apply max pooling to the feature maps
        x = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in x]  # [(N, Co), ...]*len(Ks)

        # Concatentate all of the pooled layers
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
