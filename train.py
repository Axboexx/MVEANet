from turtledemo.penrose import start

import torch
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import os
import argparse
import json
from datetime import datetime

from model.MV_EdgeGlanceNet2 import build_MVEGN2


from model.MVANet import MVANet
from utils.dataset_strategy_fpn import get_loader
from utils.misc import adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms
import torch.nn as nn
from torch.cuda import amp

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=0.00001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--model_name', type=str, default='MVANet', help='model name for saving')  # 新增模型名称参数
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training (default: None)')
parser.add_argument('--dataset', type=str, default=None, help='select dataset（foodpix、foodseg103）')
opt = parser.parse_args()

print('Generator Learning Rate: {}'.format(opt.lr_gen))
save_path = f'./saved_model/{opt.model_name}/'
os.makedirs(save_path, exist_ok=True)
best_models_dir = os.path.join(save_path, 'best_models')
os.makedirs(best_models_dir, exist_ok=True)

best_models = []
latest_checkpoint_path = None

# build models
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
generator = None
if 'mvanet' in opt.model_name:
    generator = MVANet()
elif 'mvegn2' in opt.model_name:
    generator = build_MVEGN2()
elif 'mvegn3' in opt.model_name:
    generator = build_MVEGN3()
elif 'mvegn4' in opt.model_name:
    generator = build_MVEGN4()
elif 'mvegn5' in opt.model_name:
    generator = build_MVEGN5()
elif 'mvegn' in opt.model_name:
    generator = build_MVEGN()
#
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

start_epoch = 1
if opt.resume:
    checkpoint = torch.load(opt.resume)
    generator.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}, lr: {checkpoint['lr']:.6f}")

image_root = None
gt_root = None
if opt.dataset == 'foodpix':
    image_root = '/22liuchengxu/dataset/foodpixcompleted/images/validation/'
    gt_root = '/22liuchengxu/dataset/foodpixcompleted/annotations/validation_l/'
elif opt.dataset == 'foodseg103':
    image_root = '/22liuchengxu/dataset/FoodSeg103/Images/img_dir/validation/'
    gt_root = '/22liuchengxu/dataset/FoodSeg103/Images/ann_dir/validation_l/'
elif opt.dataset == 'foodpix-mvegn':
    image_root = '/22liuchengxu/dataset/foodpixcompleted-mvegn/images/training/'
    gt_root = '/22liuchengxu/dataset/foodpixcompleted-mvegn/annotations/training/'
elif opt.dataset == 'foodpix-mvegn2':
    image_root = '/22liuchengxu/dataset/foodpixcompleted-mvegn/images/validation/'
    gt_root = '/22liuchengxu/dataset/foodpixcompleted-mvegn/annotations/validation/'
elif opt.dataset =='foodseg103-mvegn2':
    image_root='/22liuchengxu/dataset/FoodSeg103/Images/img_dir/validation/'
    gt_root='/22liuchengxu/dataset/FoodSeg103/Images/ann_dir/validation/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
to_pil = transforms.ToPILImage()
## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
use_fp16 = False
scaler = amp.GradScaler(enabled=use_fp16)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


training_data_file = os.path.join(save_path, 'training_data.txt')
if not opt.resume:
    with open(training_data_file, 'w') as f:
        f.write('Epoch\tLoss\tLearning_Rate\tTimestamp\n')

for epoch in range(start_epoch, opt.epoch + 1):
    torch.cuda.empty_cache()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        torch.cuda.empty_cache()
        for rate in size_rates:
            torch.cuda.empty_cache()
            generator_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                    align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 4, mode='nearest')
            target_2 = F.upsample(gts, size=h // 8, mode='nearest').cuda()
            target_3 = F.upsample(gts, size=h // 16, mode='nearest').cuda()
            target_4 = F.upsample(gts, size=h // 32, mode='nearest').cuda()
            target_5 = F.upsample(gts, size=h // 64, mode='nearest').cuda()

            with amp.autocast(enabled=use_fp16):
                sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1 = generator.forward(
                    images)
                loss1 = structure_loss(sideout5, target_4)
                loss2 = structure_loss(sideout4, target_3)
                loss3 = structure_loss(sideout3, target_2)
                loss4 = structure_loss(sideout2, target_1)
                loss5 = structure_loss(sideout1, target_1)
                loss6 = structure_loss(final, gts)
                loss7 = structure_loss(glb5, target_5)
                loss8 = structure_loss(glb4, target_4)
                loss9 = structure_loss(glb3, target_3)
                loss10 = structure_loss(glb2, target_2)
                loss11 = structure_loss(glb1, target_2)
                loss12 = structure_loss(tokenattmap4, target_3)
                loss13 = structure_loss(tokenattmap3, target_2)
                loss14 = structure_loss(tokenattmap2, target_1)
                loss15 = structure_loss(tokenattmap1, target_1)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3 * (
                        loss7 + loss8 + loss9 + loss10 + loss11) + 0.3 * (loss12 + loss13 + loss14 + loss15)
                Loss_loc = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                Loss_glb = loss7 + loss8 + loss9 + loss10 + loss11
                Loss_map = loss12 + loss13 + loss14 + loss15

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    current_loss = loss_record.show().item()

    # 保存最新检查点
    new_checkpoint_path = os.path.join(save_path, f'latest_epoch_{epoch}_loss_{current_loss:.4f}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': generator_optimizer.state_dict(),
        'lr': generator_optimizer.param_groups[0]['lr'],
        'loss': float(current_loss),
        'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
    }
    torch.save(checkpoint, new_checkpoint_path)
    print(f"Saved checkpoint: {new_checkpoint_path}")

    # 验证新检查点
    if not os.path.exists(new_checkpoint_path):
        raise RuntimeError(f"Failed to save checkpoint: {new_checkpoint_path}")

    # 删除上一 epoch 的检查点
    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        try:
            os.remove(latest_checkpoint_path)
            print(f"Removed previous checkpoint: {latest_checkpoint_path}")
        except OSError as e:
            print(f"Warning: Failed to remove previous checkpoint {latest_checkpoint_path}: {e}")
    latest_checkpoint_path = new_checkpoint_path

    best_models.append(checkpoint)
    candidates_sorted = sorted(best_models, key=lambda x: x['loss'])[:3]
    if len(candidates_sorted) >= 3:
        if candidates_sorted[0]['loss'] != best_models[0]['loss'] or candidates_sorted[1]['loss'] != best_models[1][
            'loss'] or candidates_sorted[2]['loss'] != best_models[2]['loss']:
            best_models = candidates_sorted
            for file in os.listdir(best_models_dir):
                file_path = os.path.join(best_models_dir, file)
                os.remove(file_path)
            for idx, model in enumerate(best_models):
                path = os.path.join(best_models_dir, 'epoch_{}_loss_{}'.format(model['epoch'], model['loss']))
                torch.save(best_models[idx], path)
    else:
        best_models = candidates_sorted
        for file in os.listdir(best_models_dir):
            file_path = os.path.join(best_models_dir, file)
            os.remove(file_path)
        for idx, model in enumerate(best_models):
            path = os.path.join(best_models_dir, 'epoch_{}_loss_{}'.format(model['epoch'], model['loss']))
            torch.save(model, path)
        # 记录训练数据到文本文件
    with open(training_data_file, 'a') as f:
        f.write(
            f'{epoch}\t{current_loss:.4f}\t{generator_optimizer.param_groups[0]["lr"]:.6f}\t{datetime.now().isoformat()}\n')

        # 记录训练日志到 JSON
    log_entry = {
        'epoch': epoch,
        'loss': float(current_loss),
        'lr': float(generator_optimizer.param_groups[0]['lr']),
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(save_path, 'training_log.json'), 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
