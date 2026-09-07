from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as T

resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)

for name, param in resnet.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

resnet = resnet.to(device)

train_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])
test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

train_dataset = FoodDataset(root=root, split="train", transform=train_transform)
valid_dataset = FoodDataset(root=root, split="val", transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

n_epoch = 25
train(resnet, n_epoch, optimizer, scheduler)
