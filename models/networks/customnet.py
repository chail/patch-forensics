'''
Customized version of pytorch resnet, alexnets.
'''

import numpy, torch, math, os
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet
from torchvision.models.alexnet import model_urls as alexnet_model_urls

def make_patch_resnet(depth, layername, num_classes=2):
    def change_out(layers):
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers)
                      if n == layername][0]
        if layername.startswith('layer'):
            bn = list(layer.modules())[-1 if depth < 50 else -2] # find final batchnorm
            assert(isinstance(bn, nn.BatchNorm2d))
            num_ch = bn.num_features
        else:
            num_ch = 64
        layers[ind+1:] = [('convout', nn.Conv2d(num_ch, num_classes, kernel_size=1))]
        return layers
    model = CustomResNet(depth, modify_sequence=change_out)
    return model

def make_patch_xceptionnet(layername, num_classes=2):
    def change_out(layers):
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers)
                      if n == layername][0]
        if layername.startswith('block'):
            module_list = list(layer.modules())
            bn = module_list[-1] # hack to find final batchnorm
            if not isinstance(bn, nn.BatchNorm2d):
                bn = module_list[-2]
            assert(isinstance(bn, nn.BatchNorm2d))
            num_ch = bn.num_features
        elif layername.startswith('relu'):
            bn = layers[ind-1][1]
            assert(isinstance(bn, nn.BatchNorm2d))
            num_ch = bn.num_features
        else:
            raise NotImplementedError
        layers[ind+1:] = [('convout', nn.Conv2d(num_ch, num_classes, kernel_size=1))]
        return layers
    model = CustomXceptionNet(modify_sequence=change_out)
    return model

def make_xceptionnet_long():
    # a modified xception net with blocks of kernel size 1
    from . import xception
    def change_out(layers):
        channels = [3, 32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728,
                    728, 728, 1024, 1536, 2048]
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers)
                      if n == 'block2'][0]
        new_layers = [
            # made all strides = 1
            ('pblock3', xception.PixelBlock(channels[4], channels[5],
                                      2,1,start_with_relu=True,grow_first=True)),
            ('pblock4', xception.PixelBlock(channels[5], channels[6],
                                      3,1,start_with_relu=True,grow_first=True)),
        ]
        num_ch = channels[9]
        new_layers.append(('convout', nn.Conv2d(num_ch, 2, kernel_size=1)))
        layers[ind+1:] = new_layers
        return layers
    model = CustomXceptionNet(modify_sequence=change_out)
    return model

class CustomResNet(nn.Module):
    '''
    Customizable ResNet, compatible with pytorch's resnet, but:
     * The top-level sequence of modules can be modified to add
       or remove or alter layers.
     * Extra outputs can be produced, to allow backprop and access
       to internal features.
     * Pooling is replaced by resizable GlobalAveragePooling so that
       any size can be input (e.g., any multiple of 32 pixels).
     * halfsize=True halves striding on the first pooling to
       set the default size to 112x112 instead of 224x224.
    '''
    def __init__(self, size=None, block=None, layers=None, num_classes=1000,
            extra_output=None, modify_sequence=None, halfsize=False):
        standard_sizes = {
            18: (resnet.BasicBlock, [2, 2, 2, 2]),
            34: (resnet.BasicBlock, [3, 4, 6, 3]),
            50: (resnet.Bottleneck, [3, 4, 6, 3]),
            101: (resnet.Bottleneck, [3, 4, 23, 3]),
            152: (resnet.Bottleneck, [3, 8, 36, 3])
        }
        assert (size in standard_sizes) == (block is None) == (layers is None)
        if size in standard_sizes:
            block, layers = standard_sizes[size]
        if modify_sequence is None:
            modify_sequence = lambda x: x
        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer # for recent resnet
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        sequence = modify_sequence([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2,
                padding=3, bias=False)),
            ('bn1', norm_layer(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, stride=1 if halfsize else 2,
                padding=1)),
            ('layer1', self._make_layer(block, 64, layers[0])),
            ('layer2', self._make_layer(block, 128, layers[1], stride=2)),
            ('layer3', self._make_layer(block, 256, layers[2], stride=2)),
            ('layer4', self._make_layer(block, 512, layers[3], stride=2)),
            ('avgpool', GlobalAveragePool2d()),
            ('fc', nn.Linear(512 * block.expansion, num_classes))
        ])
        super(CustomResNet, self).__init__()
        for name, layer in sequence:
            setattr(self, name, layer)
        self.extra_output = extra_output

    def _make_layer(self, block, channels, depth, stride=1):
        return resnet.ResNet._make_layer(self, block, channels, depth, stride)

    def forward(self, x):
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

class CustomXceptionNet(nn.Module):
    '''
    Customizable Xceptionnet, compatible with https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
    but:
     * The top-level sequence of modules can be modified to add
       or remove or alter layers.
     * Extra outputs can be produced, to allow backprop and access
       to internal features.
     * halfsize=True halves striding on the first convolution to
       allow 151x151 images to be processed rather than 299x299 only.
    '''
    def __init__(self, channels=None, num_classes=1000,
            extra_output=None, modify_sequence=None, halfsize=False):
        from . import xception
        if channels is None:
            channels = [3, 32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728,
                        728, 728, 1024, 1536, 2048]
        assert(len(channels) == 17)
        if modify_sequence is None:
            modify_sequence = lambda x: x

        sequence = modify_sequence([
            ('conv1', nn.Conv2d(channels[0], channels[1], kernel_size=3,
                                stride=1 if halfsize else 2, padding=0,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(channels[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(channels[1], channels[2], 3, bias=False)),
            ('bn2', nn.BatchNorm2d(channels[2])),
            ('relu2', nn.ReLU(inplace=True)),
            ('block1', xception.Block(channels[2],
                                      channels[3],2,2,start_with_relu=False,grow_first=True)),
            ('block2', xception.Block(channels[3], channels[4],
                                      2,2,start_with_relu=True,grow_first=True)),
            ('block3', xception.Block(channels[4], channels[5],
                                      2,2,start_with_relu=True,grow_first=True)),
            ('block4', xception.Block(channels[5], channels[6],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block5', xception.Block(channels[6], channels[7],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block6', xception.Block(channels[7], channels[8],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block7', xception.Block(channels[8], channels[9],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block8', xception.Block(channels[9], channels[10],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block9', xception.Block(channels[10], channels[11],
                                      3,1,start_with_relu=True,grow_first=True)),
            ('block10', xception.Block(channels[11], channels[12],
                                       3,1,start_with_relu=True,grow_first=True)),
            ('block11', xception.Block(channels[12], channels[13],
                                       3,1,start_with_relu=True,grow_first=True)),
            ('block12', xception.Block(channels[13], channels[14],
                                       2,2,start_with_relu=True,grow_first=False)),
            ('conv3', xception.SeparableConv2d(channels[14], channels[15],
                                               3,1,1)),
            ('bn3', nn.BatchNorm2d(channels[15])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', xception.SeparableConv2d(channels[15], channels[16],
                                               3,1,1)),
            ('bn4', nn.BatchNorm2d(channels[16])),
            ('relu4', nn.ReLU(inplace=True)),
            ('avgpool', GlobalAveragePool2d()), # does adaptive_avg_pool and flatten
            ('fc', nn.Linear(channels[16], num_classes))
        ])

        super(CustomXceptionNet, self).__init__()
        for name, layer in sequence:
            setattr(self, name, layer)
        self.extra_output = extra_output

    def forward(self, x):
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), int(numpy.prod(x.size()[1:])))
        return x

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()
    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

if __name__ == '__main__':
    import torch.utils.model_zoo as model_zoo
    # Verify that at the default settings, pytorch standard pretrained
    # models can be loaded into each of the custom nets.
    print('Loading resnet18')
    model = CustomResNet(18)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
    print('Loading resnet34')
    model = CustomResNet(34)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet34']))
    print('Loading resnet50')
    model = CustomResNet(50)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    print('Loading resnet101')
    model = CustomResNet(101)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
    print('Loading resnet152')
    model = CustomResNet(152)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet152']))

    print('Loading xceptionnet')
    model = CustomXceptionNet()
    model.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'))

