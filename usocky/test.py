from torchvision import models

net = models.vgg16(pretrained=True)

for name, param in net.named_parameters():
    print(name)

print(net)
