import os
import os.path as osp

from distutils.version import LooseVersion
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from dataloader import MRBrainSDataset
from dataloader.augmentations import *
from utils.metrics import Score, averageMeter

from torchvision import models
from torchsummary import summary
# import your model here
#from yourmodel import YourSegModel
from model import MRBrainNet

def cross_entropy2d(res, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = res.size()
    nt, ht, wt = target.size()
    
    if h > ht and w > wt:
        target = target.unsequeeze(1)
        target = F.unsample(target, size=(h,w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt:
        res = F.upsample(res, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("only support upsampling")
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(res, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.contiguous().view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250, size_average=False)
    if size_average:
        loss /= mask.float().data.sum()
    return loss

def valid():
    pass

# set hyper parameters
data_path = osp.join("/home/cv_wyz/data/", "MRBrainS")
learning_rate = 1e-3
momentum = 0.99
weight_decay = 0.005
batch_size = 32
num_workers = 4
num_epochs = 500
checkpoint_dir = "checkpoints"
log_interval = 5
save_frequency = 50
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
criterion = cross_entropy2d

# define your optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,  # set your learning rate here
    momentum=momentum, # set your momentum here
    weight_decay=weight_decay # set your modentum here
)

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer.zero_grad()
for i in range(num_epochs):
    total_loss = 0
    for index, (img, mask) in enumerate(train_loader):
        if cuda:
            image = img.cuda()
            mask = mask.cuda()
        #image = Variable(img.type(Tensor))
        #mask = Variable(mask.type(Tensor))
        optimizer.zero_grad()
        output = model(image)
        
        loss = criterion(output, mask)
        total_loss += loss

        loss.backward()
        optimizer.step()
        
        
        if index % log_interval == 0:
            print("[Train Epoch: %d/%d, Batch: %d/%d] [Losses: %.6f]" % (i, num_epochs, index, len(train_loader), total_loss/(image.size(0)*(index+1))))
    
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for img , mask in val_loader:
            if cuda:
                img = img.cuda()
                mask = mask.cuda()
            output = model(img)
            output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=True)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(probs, dim=1)
            pred = pred.cpu().data[0].numpy()
            
            label = mask.cpu().data[0].numpy()
            pred = np.asarray(pred, dtype=np.int)
            label = np.asarray(label, dtype=np.int)
            gts.append(label)
            preds.append(preds)
    
    whole_brain_preds = np.dstack(preds)
    whole_brain_gts = np.dstack(gts)
    running_metrics = Score(9)
    running_metrics.update(whole_brain_gts, whole_brain_preds)
    scores, class_iou = running_metrics.get_scores()
    mIoU = np.nanmean(class_iou[1::])
    mean_dice = (mIoU * 2) / (mIoU + 1)
    
    print("[Valid] [mean IoU: %.6f, mean dice: %.6f]" % (mIoU, mean_dice))
            
    if i % save_frequency == 0:
        # save your model
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_dir + '/model_{}.pth'.format(i))
