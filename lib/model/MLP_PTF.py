import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP_PTF(nn.Module):
    def __init__(self, 
                 filter_channels, 
                 merge_layer=0,
                 res_layers=[],
                 norm='group',
                 num_parts=2,
                 last_op=None):
        super(MLP_PTF, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op

        self.num_parts = num_parts

        # TODO: add normalization?
        self.fc_parts_0 = nn.Conv1d(filter_channels[0], 512, 1)
        self.fc_parts_1 = nn.Conv1d(512, 256, 1)
        self.fc_parts_out = nn.Conv1d(256, num_parts, 1)
        self.fc_parts_softmax = nn.Softmax(1)

        for l in range(0, len(filter_channels)-1):
            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    filter_channels[l]*num_parts+filter_channels[0],
                    filter_channels[l+1]*num_parts,
                    1,
                    groups=num_parts))
               
            else:
                if l == 0:
                    self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1] * num_parts,
                    1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l]* num_parts,
                        filter_channels[l+1] * num_parts,
                        1, groups=num_parts))
#             if l != len(filter_channels)-2:
#                 if norm == 'group':
#                     self.norms.append(nn.GroupNorm(32, filter_channels[l+1]))
#                 elif norm == 'batch':
#                     self.norms.append(nn.BatchNorm1d(filter_channels[l+1]))

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature
        phi = None

        # part
        net_parts = F.leaky_relu(self.fc_parts_0(feature))
        net_parts = F.leaky_relu(self.fc_parts_1(net_parts))
        out_parts = self.fc_parts_out(net_parts)
#         print("shape in mlp", out_parts.shape)

        parts_softmax = self.fc_parts_softmax(out_parts)
#         print("out_parts:",parts_softmax[0, :, 0])

        for i, f in enumerate(self.filters):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                y = F.leaky_relu(y)         
            if i == self.merge_layer:
                phi = y.clone()
#         print("y (before):", y[0, :, 0])
        y *= parts_softmax
        y = y.mean(1)
        y = y.view(y.shape[0], 1, y.shape[1])
#         print("y shape(after):", y.shape)
        
        if self.last_op is not None:
            y = self.last_op(y)

        return y, out_parts
