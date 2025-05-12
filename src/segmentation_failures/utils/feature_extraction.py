# credit: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/dev-ood_detection/nnunet_ext/calibration/mahalanobis/ActivationSeeker.py # noqa: B950

import torch.nn as nn


class ActivationSeeker:
    def __init__(self, to_cpu=False):
        self.activation = {}
        self.handles = []
        self.to_cpu = to_cpu

    def get_activation(self, name):
        def hook(model, input, output):
            activation = output.detach()
            if self.to_cpu:
                activation = activation.cpu()
            self.activation[name] = activation

        return hook

    def attach_hooks(self, model, hook_name_paths_dict):
        for param_name, param_path in hook_name_paths_dict.items():
            child, child_path = get_module_recursive(model, param_path)
            handle = child.register_forward_hook(self.get_activation(param_name))
            self.handles.append(handle)

    def get_data_activations(self, model=None, inputs=None):
        if model is not None and inputs is not None:
            model(inputs)
        activation_dict = dict(self.activation)
        return activation_dict

    def get_dl_activations(self, agent, dl):
        ix = 0
        dl_activations = []
        for data in dl:
            dl_activations.append(self.get_data_activations(agent, data))
            ix += 1
            if ix == 2:
                break
        return dl_activations

    def remove_handles(self):
        while len(self.handles) > 0:
            handle = self.handles.pop()
            handle.remove()


def get_module_recursive(parent_module, child_path):
    r"""Extracts a specific module from a model and the module's name. Also
    returns the path that is a module (as lower-level paths may be passed)
    Args:
        parent_module (torch.nn.Module): a PyTorch model or parent module
        child_path (str): the name of a module as extracted from the parent
        when named_parameters() is called recursively, i.e. the module is not a
        direct child, and therefore not an attribute, of the parent module
    """
    module = parent_module
    new_child_path = []
    for module_name in child_path.split("."):
        child_module = getattr(module, module_name)
        if isinstance(child_module, nn.Module):
            module = child_module
            new_child_path.append(module_name)
        else:
            break
    return module, ".".join(new_child_path)
