from torchvision import models

resnet = models.vgg16(pretrained=True)

print(resnet)