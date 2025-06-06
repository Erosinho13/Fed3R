import os
import types
from collections import OrderedDict

import torch
from functools import partial
from timm.models.efficientnet import mobilenetv2_100
from torchinfo import summary


def print_summary(self, train_batch_size):
    device = next(self.parameters()).device
    print(summary(self, input_size=(train_batch_size, 3, 224, 224), device=device))


def split_server_and_client_params(self, client_mode):
    device = next(self.parameters()).device
    if self.is_on_client is not None:
        raise ValueError('This model has already been split across clients and server.')
    assert client_mode in [None, 'feature_extractor', 'classifier', 'all']

    # Set requires_grad based on `train_mode`
    if client_mode is None:
        def is_on_client(name):
            return False
    elif client_mode in ['feature_extractor']:
        def is_on_client(name):
            return name not in ['classifier.weight', 'classifier.bias']  # everything except final fc
    elif client_mode in ['classifier']:
        def is_on_client(name):
            return name in ['classifier.weight', 'classifier.bias']  # final fc
    elif client_mode in ['all']:
        is_on_client = lambda _: True
    else:
        raise ValueError(f'Unknown client_mode: {client_mode}')

    def is_on_server(name):
        return not is_on_client(name)

    self.is_on_client = is_on_client
    self.is_on_server = is_on_server
    self.to(device)


def client_parameters(self):
    return [p for (n, p) in self.named_parameters() if self.is_on_client(n)]


def server_parameters(self):
    return [p for (n, p) in self.named_parameters() if self.is_on_server(n)]


def client_named_parameters(self):
    return [(n, p) for (n, p) in self.named_parameters() if self.is_on_client(n)]


def server_named_parameters(self):
    return [(n, p) for (n, p) in self.named_parameters() if self.is_on_server(n)]


def client_state_dict(self):
    return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_client(n))


def server_state_dict(self):
    return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_server(n))


def client_params_requires_grad_(self, requires_grad):
    for p in self.client_parameters():
        p.requires_grad_(requires_grad)


def server_params_requires_grad_(self, requires_grad):
    for p in self.server_parameters():
        p.requires_grad_(requires_grad)


def add_pfl_attributes(model):
    model.is_on_client = None
    model.print_summary = types.MethodType(print_summary, model)
    model.split_server_and_client_params = types.MethodType(split_server_and_client_params, model)
    model.client_parameters = types.MethodType(client_parameters, model)
    model.server_parameters = types.MethodType(server_parameters, model)
    model.client_named_parameters = types.MethodType(client_named_parameters, model)
    model.server_named_parameters = types.MethodType(server_named_parameters, model)
    model.client_state_dict = types.MethodType(client_state_dict, model)
    model.server_state_dict = types.MethodType(server_state_dict, model)
    model.client_params_requires_grad_ = types.MethodType(client_params_requires_grad_, model)
    model.server_params_requires_grad_ = types.MethodType(server_params_requires_grad_, model)


def save_classifier(model, root, config_name):
    torch.save(model.classifier.state_dict(), os.path.join(root, f'{config_name}_classifier.pt'))


def load_classifier(model, root, config_name):
    state_dict = torch.load(os.path.join(root, f'{config_name}_classifier.pt'), map_location=torch.device('cpu'))
    try:
        model.classifier.load_state_dict(state_dict)
    except RuntimeError:
        model.classifier = torch.nn.Linear(in_features=1280, out_features=1203)
        model.classifier.load_state_dict(state_dict)


def mobilenetv2(num_classes, pretrained_path='model_ckpts/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth',
                return_features=False, force_num_classes=False):
    norm_layer = partial(torch.nn.GroupNorm, num_groups=8, eps=1e-6)
    model = mobilenetv2_100(pretrained=False, norm_layer=norm_layer, drop_rate=0.2, drop_path_rate=0.2)
    add_pfl_attributes(model)
    # force this size to match pretrained size
    model.classifier = torch.nn.Linear(in_features=1280, out_features=2028 if not force_num_classes else num_classes)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            try:
                model.load_state_dict(state_dict['server_model_state_dict'], strict=True)
            except:
                model.classifier = torch.nn.Linear(in_features=1280, out_features=1203)
        print("Reloading pretrained weights from", pretrained_path)
    # model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)
    if return_features:
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
    return model
