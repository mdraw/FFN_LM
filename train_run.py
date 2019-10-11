import argparse
import time
import random
from torch.utils.data import DataLoader
from core.data.utils import *
from functools import partial
import os
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from core.models.ffn import FFN
from core.data import BatchCreator
import sys

import h5py
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).')
parser.add_argument('-d', '--data', type=str, default='./data1.h5', help='training data')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--delta', default=(5, 5, 5), help='delta offset')
parser.add_argument('--input_size', default=(31, 31, 31), help='input size')
parser.add_argument('--clip_grad_thr', type=float, default=0.6, help='grad clip threshold')
parser.add_argument('--save_path', type=str, default='./model', help='model save path')
parser.add_argument('--resume', type=str, default=None, help='resume training')
parser.add_argument('--interval', type=int, default=120, help='How often to save model (in seconds).')
parser.add_argument('--iter', type=int, default=1e100, help='training iteration')

parser.add_argument('--inspection_path', type=str, default='./inspection', help='inspect training process')
args = parser.parse_args()

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.inspection_path):
    os.makedirs(args.inspection_path)

def run():
    ""construct model"""
    model = FFN(in_channels=2, out_channels=1, input_size=args.input_size, delta=args.delta).cuda()

    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.save_path, 'ffn.pth')))

    """data path"""
    input_h5data = [args.data]

    """construct data loader"""
    train_dataset = BatchCreator(input_h5data, args.input_size, delta=args.delta, train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, pin_memory=True)
    
    #learning rate
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    best_loss = np.inf

    """obtain the data stream"""
    t_last = time.time()
    cnt = 0
    iter_i = 0
    tp = fp = tn = fn = 0
    while cnt < args.iter:
        cnt += 1
        for iter, (seeds, images, labels, offsets) in enumerate(
                get_batch(train_loader, args.batch_size, args.input_size,
                          partial(fixed_offsets, fov_moves=train_dataset.shifts))):

            t_curr = time.time()
            """positive_sample_weight"""
            pos_w = - torch.log((labels > 0.5).sum().float() / np.prod(labels.shape))
            slice = seeds[:, :, seeds.shape[2] // 2, :, :].sigmoid()
            seeds[:, :, seeds.shape[2] // 2, :, :] = slice
            labels = labels.cuda()

            input_data = torch.cat([images, seeds], dim=1)
            input_data = Variable(input_data.cuda())

            logits = model(input_data)
            updated = seeds.cuda() + logits

            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(updated, labels, pos_weight=pos_w)
            loss.backward()
            """clip_gradient"""
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_thr)
            optimizer.step()

            # update_seed(updated, seeds, model, offsets)
            seeds = updated


            pred_mask = (updated >= logit(0.9)).detach().cpu().numpy()
            true_mask = (labels > 0.5).cpu().numpy()

            iter_i += 1
            if iter_i % 1000 == 999:
                input_data_i = input_data.cpu().detach().numpy()
                updated_i = updated.cpu().detach().numpy()
                logits_i = logits.cpu().detach().numpy()
                inspection_save_path = 'inspection_' + str(cnt) + '_' + str(iter) + '.h5'
                with h5py.File(os.path.join(args.inspection_path, inspection_save_path), 'w') as f:
                    f.create_dataset('input_data', data=input_data_i, compression='gzip')
                    f.create_dataset('updated', data=updated_i, compression='gzip')
                    f.create_dataset('logits_out', data=logits_i, compression='gzip')
                    f.create_dataset('pred_mask', data=pred_mask, compression='gzip')
                    f.create_dataset('true_mask', data=true_mask, compression='gzip')

            true_bg = np.logical_not(true_mask)
            pred_bg = np.logical_not(pred_mask)
            tp += (true_mask & pred_mask).sum()
            fp += (true_bg & pred_mask).sum()
            fn += (true_mask & pred_bg).sum()
            tn += (true_bg & pred_bg).sum()
            precision = 1.0 * tp / max(tp + fp, 1)
            recall = 1.0 * tp / max(tp + fn, 1)
            accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
            count = round(iter / len(train_loader) * 50)
            sys.stdout.write('[Epoch {}], {}/{}: [{}{}] loss: {:.4}, Precision: {:.2f}%, Recall: {:.2f}%, '
                             'Accuracy: {:.2f}%\r'.format(cnt, (iter + 1) * args.batch_size, len(train_loader),
                                                          '#' * count, ' ' * (50 - count), loss.item(), precision*100,
                                                          recall*100, accuracy * 100))

            """save the best model!"""
            if best_loss > loss.item() or t_curr - t_last > args.interval:
                tp = fp = tn = fn = 0
                t_last = t_curr
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(args.save_path, 'ffn.pth'))
                print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                    precision * 100, recall * 100, accuracy * 100))


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)

    run()
