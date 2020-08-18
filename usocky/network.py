from torchvision import models
import torch.nn as nn


def select_model(network_name):
    if network_name == "vgg16":
        net, params_to_update = vgg16_model()

    if network_name == "resnet50":
        net, params_to_update = resnet_model()

    if network_name == "alexnet":
        net, params_to_update = alex_model()

    return net, params_to_update


def vgg16_model():
    # ネットワークモデルのロード
    net = models.vgg16_bn(pretrained=True)

    net.classifier[6] = nn.Linear(in_features=4096, out_features=1)
    net.classifier[2] = nn.Dropout(p=0.6)
    net.classifier[5] = nn.Dropout(p=0.6)

    # 調整するパラメータの名前

    update_param_names = [
        "features.19.weight",
        "features.19.bias",
        "features.21.weight",
        "features.21.bias",
        "features.24.weight",
        "features.24.bias",
        "features.26.weight",
        "features.26.bias",
        "features.28.weight",
        "features.28.bias",
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
        "classifier.6.weight",
        "classifier.6.bias",
    ]

    params_to_update = create_update_param_list(net, update_param_names)

    return net, params_to_update


def create_update_param_list(network, update_param_names):

    params_to_update = []

    for name, param in network.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    return params_to_update


def resnet_model():
    net = models.resnet50()
    net.fc = nn.Linear(in_features=2048, out_features=1)

    update_param_names = [
        "layer4.1.conv1.weight",
        "layer4.1.bn1.weight",
        "layer4.1.bn1.bias",
        "layer4.1.conv2.weight",
        "layer4.1.bn2.weight",
        "layer4.1.bn2.bias",
        "layer4.1.conv3.weight",
        "layer4.1.bn3.weight",
        "layer4.1.bn3.bias",
        "layer4.2.conv1.weight",
        "layer4.2.bn1.weight",
        "layer4.2.bn1.bias",
        "layer4.2.conv2.weight",
        "layer4.2.bn2.weight",
        "layer4.2.bn2.bias",
        "layer4.2.conv3.weight",
        "layer4.2.bn3.weight",
        "layer4.2.bn3.bias",
        "fc.weight",
        "fc.bias",
    ]

    params_to_update = create_update_param_list(net, update_param_names)

    return net, params_to_update


def alex_model():
    net = models.alexnet(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=1)

    update_param_names = [
        "classifier.1.weight",
        "classifier.1.bias",
        "classifier.4.weight",
        "classifier.4.bias",
        "classifier.6.weight",
        "classifier.6.bias",
    ]

    params_to_update = create_update_param_list(net, update_param_names)

    return net, params_to_update


if __name__ == "__main__":
    net = alex_model()

    print(net)
    for name, param in net.named_parameters():
        print(name)
