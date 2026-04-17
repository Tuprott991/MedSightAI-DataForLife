# Adapted from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
import torch.nn as nn


class FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers, return_gradients=True):
        self.model = model
        self.target_layers = target_layers
        self.return_gradients = return_gradients
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                if self.return_gradients:
                    x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers."""

    def __init__(self, model, feature_module, target_layers, return_gradients=True):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(
            self.feature_module, target_layers, return_gradients
        )

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            print("inside feature_module sublayer:", name)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x
