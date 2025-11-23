from typing import Dict, List, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from region_loss import RegionLoss
from cfg import *

class MaxPoolStride1(nn.Module):
    def __init__(self) -> None:
        super(MaxPoolStride1, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride: int = 2) -> None:
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        #Simen: edited as suggested here: https://github.com/marvis/pytorch-yolo2/issues/129#issue-350726531
        #x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        #x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        #x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        #x = x.view(B, hs*ws*C, H/hs, W/ws)
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self) -> None:
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile: str) -> None:
        super(Darknet, self).__init__()
        self.blocks: List[Dict[str, Any]] = parse_cfg(cfgfile)
        self.models: nn.ModuleList = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss: Union[RegionLoss, nn.Module] = self.models[len(self.models)-1]

        self.width: int = int(self.blocks[0]['width'])
        self.height: int = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            # Type ignore: accessing RegionLoss-specific attributes
            self.anchors: List[float] = self.loss.anchors  # type: ignore[union-attr, assignment]
            self.num_anchors: int = self.loss.num_anchors  # type: ignore[union-attr, assignment]
            self.anchor_step: float = self.loss.anchor_step  # type: ignore[union-attr, assignment]
            self.num_classes: int = self.loss.num_classes  # type: ignore[union-attr, assignment]

        self.header: torch.Tensor = torch.IntTensor([0,0,0,0])
        self.seen: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self) -> None:
        print_cfg(self.blocks)

    def create_network(self, blocks: List[Dict[str, Any]]) -> nn.ModuleList:
        models = nn.ModuleList()

        prev_filters = 3
        out_filters: List[int] = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                #Simen: edit as sugessted here: https://github.com/marvis/pytorch-yolo2/issues/129#issue-350726531
                #pad = (kernel_size-1)/2 if is_pad else 0
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model: nn.Module = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                model_mp: nn.Module
                if stride > 1:
                    model_mp = nn.MaxPool2d(pool_size, stride)
                else:
                    model_mp = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model_mp)
            elif block['type'] == 'avgpool':
                model_avg: nn.Module = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model_avg)
            elif block['type'] == 'softmax':
                model_sm: nn.Module = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model_sm)
            elif block['type'] == 'cost':
                model_cost: nn.Module
                if block['_type'] == 'sse':
                    model_cost = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model_cost = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model_cost = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model_cost)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                model_fc: nn.Module
                if block['activation'] == 'linear':
                    model_fc = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model_fc = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model_fc = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model_fc)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile: str) -> None:
        fp = open(weightfile, 'rb')
        header: NDArray[np.int32] = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = int(self.header[3].item())
        buf: NDArray[np.float32] = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    # Type ignore for module indexing as we know it's a Sequential
                    start = load_conv_bn(buf, start, model[0], model[1])  # type: ignore[index]
                else:
                    start = load_conv(buf, start, model[0])  # type: ignore[index, arg-type]
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])  # type: ignore[index, arg-type]
                else:
                    start = load_fc(buf, start, model)  # type: ignore[arg-type]
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile: str, cutoff: int = 0) -> None:
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])  # type: ignore[index, arg-type]
                else:
                    save_conv(fp, model[0])  # type: ignore[index, arg-type]
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fp, model)  # type: ignore[arg-type]
                else:
                    save_fc(fp, model[0])  # type: ignore[index, arg-type]
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
