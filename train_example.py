import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloader import MRBrainSDataset
from dataloader.augmentations import *

from torchvision import models
from torchsummary import summary
# import your model here
#from yourmodel import YourSegModel
from model import MRBrainNet

# set hyper parameters
data_path = osp.join("/home/cv_wyz/data/", "MRBrainS")
learning_rate = 1e-8
momentum = 0.99
weight_decay = 0.005
batch_size = 16
num_workers = 4
num_epochs = 100
checkpoint_dir = "checkpoints"
save_frequency = 10
use_cuda = True

cuda = torch.cuda.is_available() and use_cuda

os.makedirs(checkpoint_dir, exist_ok=True)

# define your data loader to load the trainning data
data_aug = Compose([
           RandomHorizontallyFlip(0.5),
           RandomRotate(10),
           Scale(256),
           ])
train_loader = DataLoader(MRBrainSDataset(
        data_path, split='train', is_transform=True, img_norm=True, augmentations=data_aug), batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(MRBrainSDataset(
        data_path, split='val', is_transform=True, img_norm=True, augmentations=data_aug), batch_size=batch_size, shuffle=True, num_workers=num_workers
)




# define your model
# TODO:
#model = YourSegModel()
#model.train()
#model.cuda()
model = MRBrainNet(n_classes=9)

pretrained_model = "pretrained_model/vgg16-397923af.pth"
vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load(pretrained_model))
model.copy_params_from_vgg16(vgg16)

if cuda:
    model = model.cuda()

summary(model, (3, 256, 256))

model.train()

# define your criterion
criterion = nn.CrossEntropyLoss()
criterion.cuda()

# define your optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,  # set your learning rate here
    momentum=momentum, # set your momentum here
    weight_decay=weight_decay # set your modentum here
)

optimizer.zero_grad()
for i in range(num_epochs):
    total_loss = 0
    for index, (img, mask) in enumerate(train_loader):
        image = Variable(img).cuda()
        optimizer.zero_grad()
        output = model(image)
        
        loss = criterion(output, mask)
        total_loss += loss

        loss.backward()
        optimizer.step()
        print("[Epoch: %d/%d, Batch: %d/%d] [Losses: %.6f" % (i, num_epochs, index, len(train_loader), total_loss/(image.size(0)*(index+1))))
        
    if num_epochs % save_frequency == 0:
        # save your model
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_dir + '/model_{}.pth'.format(num_epochs))
