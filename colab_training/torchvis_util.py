"""
Torch visualization utilities for guided backpropagation and saliency maps
Adapted from MarkoArsenovic/DeepLearning_PlantDiseases for modern PyTorch
"""
from collections import OrderedDict
from enum import Enum
from functools import partial
import torch
import torch.nn as nn


class GradType(Enum):
    """
    Gradient calculation types for saliency map generation
    Based on Simonyan et al. (2013) and related work
    """
    NAIVE = 0   # Plain gradient (Simonyan et al., 2013)
    GUIDED = 1  # Guided backpropagation 
    DECONV = 2  # Deconvolution


def augment_module(net: nn.Module):
    """
    Augment a PyTorch module with hooks for gradient computation and visualization
    
    Args:
        net: PyTorch neural network module
        
    Returns:
        tuple: (vis_param_dict, reset_state_func, remove_handles_func)
    """
    layer_dict, remove_forward = _augment_module_pre(net)
    vis_param_dict, remove_backward = _augment_module_post(net, layer_dict)

    def remove_handles():
        remove_forward()
        remove_backward()

    def reset_state():
        for x, y in layer_dict.items():
            print('clearing {}'.format(x))
            assert isinstance(y, dict)
            y.clear()

    return vis_param_dict, reset_state, remove_handles


def _forward_hook(m, in_, out_, module_name, callback_dict):
    """Forward hook to capture ReLU activations"""
    callback_dict[module_name]['output'] = out_.data.clone()


def _augment_module_pre(net: nn.Module):
    """
    Add forward hooks to capture intermediate activations
    
    Args:
        net: PyTorch neural network module
        
    Returns:
        tuple: (callback_dict, remove_handles_func)
    """
    callback_dict = OrderedDict()
    forward_hook_remove_func_list = []

    for x, y in net.named_modules():
        if not isinstance(y, nn.Sequential) and y is not net:
            if isinstance(y, nn.ReLU):
                callback_dict[x] = {}
                forward_hook_remove_func_list.append(
                    y.register_forward_hook(
                        partial(_forward_hook, module_name=x, callback_dict=callback_dict)
                    )
                )

    def remove_handles():
        for x in forward_hook_remove_func_list:
            x.remove()

    return callback_dict, remove_handles


def _backward_hook(m: nn.Module, grad_in, grad_out, module_name, callback_dict, vis_param_dict):
    """
    Backward hook for guided backpropagation and gradient modification
    
    Args:
        m: Module where hook is attached
        grad_in: Input gradients
        grad_out: Output gradients  
        module_name: Name of the module
        callback_dict: Dictionary storing forward activations
        vis_param_dict: Visualization parameters
    """
    # Get visualization parameters
    layer = vis_param_dict['layer']
    method = vis_param_dict['method']
    
    if module_name not in callback_dict:
        return
    
    # Ensure we have proper gradient tensors
    if not isinstance(grad_in, tuple) or not isinstance(grad_out, tuple):
        return
        
    if len(grad_out) == 0:
        return
        
    # Work on the actual grad_out - clone for safety
    grad_out_actual = grad_out[0].clone()

    # Apply gradient modification based on method
    if isinstance(m, nn.ReLU):
        new_grad = grad_out_actual
        
        # Get saved activations from forward pass
        if 'output' in callback_dict[module_name]:
            response = callback_dict[module_name]['output']
            
            if method == GradType.NAIVE:
                # Standard backpropagation - zero negative activations
                new_grad = new_grad.clone()
                new_grad[response <= 0] = 0
            elif method == GradType.GUIDED:
                # Guided backpropagation - zero both negative activations and negative gradients
                new_grad = new_grad.clone()
                new_grad[response <= 0] = 0
                new_grad[grad_out_actual <= 0] = 0
            elif method == GradType.DECONV:
                # Deconv - only zero negative gradients
                new_grad = new_grad.clone()
                new_grad[grad_out_actual <= 0] = 0
        
        return (new_grad,) + grad_in[1:] if len(grad_in) > 1 else (new_grad,)
    
    elif isinstance(m, nn.Linear):
        # For linear layers, modify gradient computation if needed
        if len(grad_in) > 0:
            return grad_in
    
    return grad_in


def _augment_module_post(net: nn.Module, callback_dict: dict):
    """
    Add backward hooks for gradient modification
    
    Args:
        net: PyTorch neural network module
        callback_dict: Dictionary from forward hooks
        
    Returns:
        tuple: (vis_param_dict, remove_handles_func)
    """
    backward_hook_remove_func_list = []

    vis_param_dict = dict()
    vis_param_dict['layer'] = None
    vis_param_dict['index'] = None
    vis_param_dict['method'] = GradType.NAIVE

    for x, y in net.named_modules():
        if not isinstance(y, nn.Sequential) and y is not net:
            backward_hook_remove_func_list.append(
                y.register_backward_hook(
                    partial(_backward_hook, 
                           module_name=x, 
                           callback_dict=callback_dict, 
                           vis_param_dict=vis_param_dict)
                )
            )

    def remove_handles():
        for x in backward_hook_remove_func_list:
            x.remove()

    return vis_param_dict, remove_handles