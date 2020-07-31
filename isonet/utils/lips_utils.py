import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# Local imports
from isonet.models import *
from isonet.models.resnet import BasicBlock

# Taken from
# The Singular Values of Convolutional Layers, Sedghi et al., ICLR 2019
# https://arxiv.org/pdf/1805.10408.pdf
def singular_values(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def conv_lipschitz(conv_layer, input_shape):
    kernel = conv_layer.weight.permute(2, 3, 0, 1).cpu().detach().numpy()
    # For the input shape, say [3, 32, 32] just pass [32, 32]
    sing_vals = singular_values(kernel, input_shape[1:])
    lip_const = sing_vals.max()

    rand_input = torch.randn((1,*input_shape), device=conv_layer.weight.device)
    output_shape = list(conv_layer(rand_input).shape[1:])
    return lip_const, output_shape


def resblock_lipschitz(block, input_shape):
    inn_consts = OrderedDict()
    # Conv1
    lip_const, output_shape = conv_lipschitz(block.f.a, input_shape)
    inn_consts.update({ 'l_conv1' : lip_const })
    # Conv2
    lip_const, output_shape = conv_lipschitz(block.f.b, output_shape)
    inn_consts.update({ 'l_conv2' : lip_const })
    # Create DataFrame for printint individual Lipschitz constants
    df = pd.DataFrame({
        'layer_name' : list(inn_consts.keys()), 
        'lipschitz_constant' : list(inn_consts.values())
    })
    print('Inner Lipschitz constant of the block:\n', df.to_string(index=False))
    return inn_consts, output_shape


def basicblock_lipschitz(block, input_shape):
    inn_consts = OrderedDict()
    # Conv1
    lip_const, output_shape = conv_lipschitz(block.conv1, input_shape)
    inn_consts.update({ 'l_conv1' : lip_const })
    # BatchNorm1
    inn_consts.update({ 'l_bn1'   : bn_lipschitz(block.bn1) })
    # Conv2
    lip_const, output_shape = conv_lipschitz(block.conv2, output_shape)
    inn_consts.update({ 'l_conv2' : lip_const })
    # BatchNorm2
    inn_consts.update({ 'l_bn2'   : bn_lipschitz(block.bn2) })
    # Compute 'total' Lipschitz constant up until now
    inn_consts.update({ 'before_shortcut' : np.prod( list(inn_consts.values()) ) })
    # If there is a short cut there also a residual connection!
    if len(block.shortcut) != 0:
        lip_const, output_shape = conv_lipschitz(block.shortcut[0], input_shape)
        inn_consts.update({ 'l_shortcut_conv1' : lip_const })

        inn_consts.update({ 'l_shortcut_bn1' : bn_lipschitz(block.shortcut[1]) })
        # for residual connection
        shortcut_lip = inn_consts['l_shortcut_conv1'] * inn_consts['l_shortcut_bn1']

    else:
        # no residual connection
        shortcut_lip = 0

    inn_consts.update({ 'after_shortcut' : inn_consts['before_shortcut'] + shortcut_lip } )
    # Create DataFrame for printint individual Lipschitz constants
    df = pd.DataFrame({
        'layer_name' : list(inn_consts.keys()), 
        'lipschitz_constant' : list(inn_consts.values())
    })
    print('Inner Lipschitz constant of the block:\n', df.to_string(index=False))
    return inn_consts, output_shape

'''
Lipschitz constant for average pooling. Taken from Section 2.3.1 of
https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2016/2016-29_fp.pdf
'''
def avg_pool_lipschitz(pool_layer, input_shape):
    n_pool_elements = input_shape[1] * input_shape[2]
    lip_const = n_pool_elements ** (-1/2)

    rand_input = torch.randn((1,*input_shape))
    output_shape = list(pool_layer(rand_input).shape[1:])
    return lip_const, output_shape


def linear_lipschitz(linear_layer, shape):
    weight = linear_layer.weight.cpu().detach().numpy()
    sing_vals = np.linalg.svd(weight, compute_uv=False)
    lip_const = max(sing_vals)

    rand_input = torch.randn((1,*shape), device=linear_layer.weight.device)
    output_shape = list(linear_layer(rand_input).shape[1:])
    return lip_const, output_shape

# Based on 
# Lipschitz Continuous Neural Networks, Gouk et al., 2018
# https://arxiv.org/pdf/1804.04368.pdf
def bn_lipschitz(bn_layer):
    gamma = bn_layer.weight.cpu().detach().numpy()
    var = bn_layer.running_var.cpu().detach().numpy()
    sqrt_var = np.sqrt(var)
    quot = np.abs(gamma / sqrt_var)
    lip_const = quot.max()
    return lip_const


def resnet18_lipschitz(model, input_shape):
    # Starting
    layers = OrderedDict()
    layers['conv1']                 = model.conv1
    layers['bn']                    = model.bn1
    # Layer 1
    layers['layer1.basic_block1']   = list(model.layer1.named_children())[0][1]
    layers['layer1.basic_block2']   = list(model.layer1.named_children())[1][1]
    # Layer 2
    layers['layer2.basic_block1']   = list(model.layer2.named_children())[0][1]
    layers['layer2.basic_block2']   = list(model.layer2.named_children())[1][1]
    # Layer 3
    layers['layer3.basic_block1']   = list(model.layer3.named_children())[0][1]
    layers['layer3.basic_block2']   = list(model.layer3.named_children())[1][1]
    # Layer 4
    layers['layer4.basic_block1']   = list(model.layer4.named_children())[0][1]
    layers['layer4.basic_block2']   = list(model.layer4.named_children())[1][1]
    # Linear
    layers['linear']                = model.linear

    lips_constants = OrderedDict()
    shape = input_shape
    print('Computing Lipschitz constant of each layer...')
    for idx, (layer_name, layer) in enumerate(layers.items()):
        print(50 * '-')
        print(f'Layer #{idx}: "{layer_name}". Inp. shape: {shape}. Layer is', end=' ')
        if isinstance(layer, nn.Conv2d):
            print('Convolution.', end=' ')
            lip_const, shape = conv_lipschitz(layer, shape)
        elif isinstance(layer, nn.BatchNorm2d):
            print('BatchNorm.', end=' ') # shape doesn't change
            lip_const = bn_lipschitz(layer)
        elif isinstance(layer, BasicBlock):
            print('BasicBlock.', end=' ')
            inn_consts, shape = basicblock_lipschitz(layer, shape)
            lip_const = inn_consts['after_shortcut']
        elif isinstance(layer, nn.Linear):
            print('Linear.', end=' ')
            # In ResNet18 there's only one Linear layer: before the output.
            # Before passing the feature, there's an average pooling with a kernel
            # of size equal to that of the spatial dimensions
            shape = shape[:1]
            lip_const, shape = linear_lipschitz(layer, shape)


        print(f'Output shape: {shape}')
        lips_constants[layer_name] = lip_const

    # See individual constants:
    df = pd.DataFrame({
        'layer_name' : list(lips_constants.keys()), 
        'lipschitz_constant' : list(lips_constants.values())
    })
    print('\nAll Lipschitz constants: \n', df.to_string(index=False))

    tot_lipschitz = np.prod(list(lips_constants.values()))
    print(f'\n>> Total Lipschitz constant of the network: {tot_lipschitz:4.3f}')
    return tot_lipschitz


def isonet18_lipschitz(model, input_shape):
    # Starting
    layers = OrderedDict()
    layers['stem']          = model.stem.conv # stem has conv, relu and maxpool
    # Stage 1. Blocks have conv, relu, conv, relu
    layers['stage1.block1'] = model.s1.b1
    layers['stage1.block2'] = model.s1.b2
    # Stage 2
    layers['stage2.block1'] = model.s2.b1
    layers['stage2.block2'] = model.s2.b2
    # Stage 3
    layers['stage3.block1'] = model.s3.b1
    layers['stage3.block2'] = model.s3.b2
    # Stage 4
    layers['stage4.block1'] = model.s4.b1
    layers['stage4.block2'] = model.s4.b2
    # Linear. The head has avgpool, dropout and fc
    layers['avg_pool']      = model.head.avg_pool
    layers['linear']        = model.head.fc

    lips_constants = OrderedDict()
    shape = input_shape
    print('Computing Lipschitz constant of each layer...')
    for idx, (layer_name, layer) in enumerate(layers.items()):
        print(50 * '-')
        print(f'Layer #{idx}: "{layer_name}". Inp. shape: {shape}. Layer is', end=' ')
        if isinstance(layer, nn.Conv2d):
            print('Convolution.', end=' ')
            lip_const, shape = conv_lipschitz(layer, shape)
        elif isinstance(layer, ResBlock):
            print('ResBlock.', end=' ')
            inn_consts, shape = resblock_lipschitz(layer, shape)
            lip_const = inn_consts['l_conv1'] * inn_consts['l_conv2']
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            lip_const, shape = avg_pool_lipschitz(layer, shape)
        elif isinstance(layer, nn.Linear):
            print('Linear.', end=' ')
            # In ResNet18 there's only one Linear layer: before the output.
            # Before passing the feature, there's an average pooling with a kernel
            # of size equal to that of the spatial dimensions
            shape = shape[:1]
            lip_const, shape = linear_lipschitz(layer, shape)

        print(f'Output shape: {shape}')
        lips_constants[layer_name] = lip_const

    # See individual constants:
    df = pd.DataFrame({
        'layer_name' : list(lips_constants.keys()), 
        'lipschitz_constant' : list(lips_constants.values())
    })
    print('\nAll Lipschitz constants: \n', df.to_string(index=False))

    tot_lipschitz = np.prod(list(lips_constants.values()))
    print(f'\n>> Total Lipschitz constant of the network: {tot_lipschitz:4.3f}')
    return tot_lipschitz