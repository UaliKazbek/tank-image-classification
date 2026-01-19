import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision 
import torchvision.models as models

from sklearn.metrics import classification_report
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_efficientnet = models.EfficientNet_B0_Weights.DEFAULT
transform = weights_efficientnet.transforms()

train_set = torchvision.datasets.ImageFolder(r"D:\dataset\train", transform=transform)
val_set = torchvision.datasets.ImageFolder(r"D:\dataset\val", transform=transform)
test_set = torchvision.datasets.ImageFolder(r"D:\dataset\test", transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)



# train_classifier
model = models.efficientnet_b0(weights='DEFAULT')

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, 10)
)

model = model.to(device)
loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)



# model training
EPOCHS = 8
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
count = 0
best_loss = None

for epoch in range(EPOCHS):
    model.train()
    running_train_loop = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for images, targets in train_loop:
        images = images.to(device)
        targets = targets.to(torch.long).to(device)

        pred = model(images)
        loss = loss_model(pred, targets)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        running_train_loop.append(loss.item())
        mean_train_loss = sum(running_train_loop)/len(running_train_loop)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f'EPOCH {epoch+1}/{EPOCHS}, train_loss: {mean_train_loss:.4f}')

    running_train_acc = true_answer / len(train_set)
    train_acc.append(running_train_acc)
    train_loss.append(mean_train_loss)

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        running_val_loop = []
        true_answer = 0
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(torch.long).to(device)

            pred = model(images)
            loss = loss_model(pred, targets)

            running_val_loop.append(loss.item())
            mean_val_loss = sum(running_val_loop) / len(running_val_loop)

            true_answer += (pred.argmax(dim=1) == targets).sum().item()

            preds = pred.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        running_val_acc = true_answer / len(val_set)
        val_acc.append(running_val_acc)
        val_loss.append(mean_val_loss)

    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f'EPOCH {epoch+1}/{EPOCHS}, train_loss: {mean_train_loss:.4f}, train_acc: {running_train_acc:.4f}, val_loss: {mean_val_loss:.4f}, val_acc: {running_val_acc:.4f}, lr: {lr}')
    
    print(classification_report(all_targets, all_preds, digits=3))

    if best_loss is None:
        best_loss = mean_val_loss


    if mean_val_loss < best_loss:
        best_loss = mean_val_loss
        count = 0

        checkpoint = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }

        torch.save(model.state_dict(), f'efficientnet_state_dict.pt')
        print(f'на {epoch+1} эпохе модель сохранила значение функция потерь на валидации {mean_val_loss:.4f}')
    else:
        count +=1



# model with a trained classifier
param_model = torch.load(r'C:\Users\STARLINECOMP\projects\classification\efficientnet_learned_classifier.pt')

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, 10)
)

model.load_state_dict(param_model)
model = model.to(device)



# defrosting
for param in model.parameters():
    param.requires_grad = False

for param in model.features[-2:].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,
    weight_decay=1e-4
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
loss_model = nn.CrossEntropyLoss()



#model training
EPOCHS = 7
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
count = 0
best_loss = None

for epoch in range(EPOCHS):
    model.train()
    running_train_loop = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for images, targets in train_loop:
        images = images.to(device)
        targets = targets.to(torch.long).to(device)

        pred = model(images)
        loss = loss_model(pred, targets)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        running_train_loop.append(loss.item())
        mean_train_loss = sum(running_train_loop)/len(running_train_loop)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f'EPOCH {epoch+1}/{EPOCHS}, train_loss: {mean_train_loss:.4f}')

    running_train_acc = true_answer / len(train_set)
    train_acc.append(running_train_acc)
    train_loss.append(mean_train_loss)

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        running_val_loop = []
        true_answer = 0
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(torch.long).to(device)

            pred = model(images)
            loss = loss_model(pred, targets)

            running_val_loop.append(loss.item())
            mean_val_loss = sum(running_val_loop) / len(running_val_loop)

            true_answer += (pred.argmax(dim=1) == targets).sum().item()

            preds = pred.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        running_val_acc = true_answer / len(val_set)
        val_acc.append(running_val_acc)
        val_loss.append(mean_val_loss)

    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f'EPOCH {epoch+1}/{EPOCHS}, train_loss: {mean_train_loss:.4f}, train_acc: {running_train_acc:.4f}, val_loss: {mean_val_loss:.4f}, val_acc: {running_val_acc:.4f}, lr: {lr}')
    
    print(classification_report(all_targets, all_preds, digits=3))

    if best_loss is None:
        best_loss = mean_val_loss


    if mean_val_loss < best_loss:
        best_loss = mean_val_loss
        count = 0

        # checkpoint = {
        #     'model': model.state_dict(),
        #     'opt': opt.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        # }

        torch.save(model.state_dict(), f'efficientnet_state_dict.pt')
        print(f'на {epoch+1} эпохе модель сохранила значение функция потерь на валидации {mean_val_loss:.4f}')
    else:
        count +=1

