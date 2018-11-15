import torch
import torch.nn as nn
from dataloader import MRBrainSDataset
from utils.metrics import Score

val_loader = torch.utils.data.DataLoader(MRBrainSDataset('path/to/MRBrainS', split='val', is_transform=True, img_norm=True, augmentations=None), batch_size=1)


# return mean IoU, mean dice
def evaluate(model, val_loader):
    model.eval()
    model.cuda()
    preds = []
    gts = []
    for i, (img, mask) in val_loader:
        img = Variable(img).cuda()
        h, w = mask[1::]
        with torch.no_grad():
            output = model(img)
            output = F.interpolate(output, size=(h, w),  mode='bilinear', align_corners=True)
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
    return mIoU, mean_dice
