import torch
import torch.nn.functional as F
import time
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.model import model_VGG
from data import test_dataset
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = ''
model = model_VGG()
model.load_state_dict(torch.load(''))
model.cuda()
model.eval()
test_datasets = [ '\\DUT-RGBD\\test_data']
#test_datasets = [ '\\LFSD']
#test_datasets = [ '\\NJUD\\test_data']
#test_datasets = [ '\\NLPR\\test_data']
#test_datasets = [ '\\RGBD135']
#test_datasets = [ '\\SSD']
#test_datasets = [ '\\STEREO']
#test_datasets = [ '\\DUTS-TEST']
#time_start=time.time()
for dataset in test_datasets:
    save_path = '' + dataset + '\\results\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '\\images\\'
    gt_root = dataset_path + dataset + '\\gts\\'
    depth_root = dataset_path + dataset + '\\depths\\'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,depth, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        depth /= (depth.max() + 1e-8)
        image = Variable(image).cuda()
        depth = Variable(depth).cuda()
        n,c, h, w = image.size()
        depth1 = depth.view(n,h, w, 1).repeat(1,1, 1, c)
        depth1 = depth1.transpose(3, 2)
        depth1 = depth1.transpose(2, 1)
        time_start = time.time()
        _, res, _, _ = model(image, depth1, depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        time_end = time.time()
        res = res.data.sigmoid().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print(name)
        misc.imsave(save_path + name, res)
        print('totally cost:',time_end-time_start,'s')
