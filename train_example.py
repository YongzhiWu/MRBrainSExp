import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from dataloader import MRBrainSDataset
from dataloader.augmentations import *
# import your model here
from yourmodel import YourSegModel

# define your data loader to load the trainning data
data_aug = Compose([
           RandomHorizontallyFlip(0.5),
           RandomRotate(10),
           Scale(256),
           ])
train_loader = DataLoader(MRBrainSDataset('MRBrainS', split='train', is_transform=True, img_norm=True, augmentations=data_aug), batch_size=1)




# define your model
# TODO:
model = YourSegModel()
model.train()
model.cuda()

# define your criterion
criterion = nn.CrossEntropyLoss()
criterion.cuda()

# define your optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,  # set your learning rate here
    momentum=args.momentum, # set your momentum here
    weight_decay=args.weight_decay # set your modentum here
)

optimizer.zero_grad()
for i in range(num_epochs):
    for index, (img, mask) in enumerate(train_loader):
        image = Variable(image).cuda()
        optimizer.zero_grad()
        # ...
        # ...
        # ...
        loss.backward()
        optimizer.step()
    if num_epochs % save_frequecy == 0:
        # save your model
        state_dict = model.state_dict()
        torch.save(state_dict, 'modelname_{}.pth'.format(num_epochs))
