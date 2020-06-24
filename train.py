import torch
import torch.nn as nn
import pdb, os, argparse
from torch.autograd import Variable
from datetime import datetime
from model.model import model_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr

bce_loss = torch.nn.BCELoss(size_average=True)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.3, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
parser.add_argument('--start_epoch', default=37, type=int)
parser.add_argument('--total_depth', type=int, default=10, help='total depth')
parser.add_argument('--total_length', type=int,default=4, help='length of regions')
parser.add_argument('--total_width', type=int,default=4, help='width of regions')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
model = model_VGG()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# iamge roots #
image_root = ''
gt_root = ''
depth_root = ''
boundary_root = ''
pre_cheak_root = ''

train_loader = get_loader(image_root, gt_root,depth_root,boundary_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
CE = torch.nn.BCEWithLogitsLoss(reduce = False)
BCE = torch.nn.BCEWithLogitsLoss()

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts,depth_1,bdrs = pack
        images = Variable(images)
        gts = Variable(gts)
        depth = Variable(depth_1)
        bdrs = Variable(bdrs)
        images = images.cuda()
        gts = gts.cuda()
        depth = depth.cuda()
        n, c, h, w = images.size()
        depth1 = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        depth1 = depth1.transpose(3, 1)
        depth1 = depth1.transpose(3, 2)
        bdrs = bdrs.cuda()
        det_dps, dets,bdr_p,atts5 = model(images,depth1,depth)
        loss_bdr = BCE(bdr_p, bdrs)
        loss4 = loss_bdr
        max_pool1 = nn.MaxPool2d(4, stride=None)
        max_pool2 = nn.MaxPool2d(4, stride=None)
        max_pool3 = nn.MaxPool2d(2, stride=None)
        upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        detts = torch.nn.functional.sigmoid(bdr_p)
        gtts = bdrs * 1
        result_pool = max_pool1(detts)
        result_pool = upsample(result_pool)
        result_pool2 = max_pool2(gtts)
        result_pool2 = upsample(result_pool2)
        result_ = torch.max(result_pool, result_pool2)
        result1 = result_pool * result_pool2
        result = result_ - result1
        result_p = max_pool3(result)
        resultp = upsample2(result_p)
        loss1 = BCE(det_dps, gts)
        loss_sal = CE(det_dps, gts)
        loss_sals = CE(dets, gts)
        loss2 = BCE(dets, gts)
        loss3 = torch.mul(loss_sals,resultp).mean()
        n_, _, _, _ = gts.size()
        res_f = torch.zeros((n_,1,256, 256))
        loss5 = bce_loss(atts5, gts)
        for jj in range(opt.total_depth):
            res = depth * 255
            target = gts
            target = target * 255
            res1 = (res >= (255.0 / opt.total_depth) * jj)
            res1 = res1.type(torch.FloatTensor)
            res1 = res1.cuda()
            res3 = (res <= (255.0 / opt.total_depth) * (jj + 1))
            res3 = res3.type(torch.FloatTensor)
            res3 = res3.cuda()
            res2 = res * res1 * res3
            res2[res2 > 0] = 255
            res_sim = res2 * (target / 255)
            res_res = res2
            total = target.mean(dim=3)
            total = total.mean(dim=2)
            res_sim = res_sim.mean(dim = 3)
            weight = torch.div(res_sim.mean(dim=2),total)
            weight = torch.unsqueeze(weight, -1)
            weight = torch.unsqueeze(weight, -1)
            res_f = res_f.cuda()
            res__ = torch.mul(res_res , weight)
            res_f = res_f + res__
        res_f = res_f / 255
        pre_hard_region = torch.mul(loss_sal, res_f).mean()
        loss_hard_region = pre_hard_region
        loss = loss1 + loss2 + loss3*0.3 + loss4  + loss5 + loss_hard_region*0.3
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        if i % 5 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}  Loss4: {:0.4f} Loss5: {:0.4f} Loss6: {:0.4f}  Loss: {:0.4f}  Step: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data,loss3.data,  loss4.data, loss5.data,loss_hard_region.data,loss.data, i+(epoch-1)*total_step ))
    save_path = 'H:\kalili\Recoverit 2019-11-11 at 09.10.02\I(NTFS)\PAPER_CVPR\F_CODES_SUM\Final_cheak\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + '%d' % epoch +  '_w.pth' )
        
progress = range(opt.start_epoch+1 , opt.epoch)
for epoch in progress:
    if opt.param == True:
        if epoch!= 1:
             print("\nloading parameters")
             model.load_state_dict(torch.load(pre_cheak_root + '%d'% (epoch-1) + '_w.pth'))
    print(epoch)
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    print("train.")