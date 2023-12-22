import sys
import os
sys.path.insert(0, os.getcwd())
import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

from loguru import logger


def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            dataset_opt['io_backend']['type'] = 'disk'
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            dataset_opt['io_backend']['type'] = 'disk'
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    prefetcher = CPUPrefetcher(train_loader)

    # create model
    exp_id = '0915-0703-0400-02'
    model = build_model(opt)  
    model.load_network(model.net_g, f'experiments/{exp_id}/models/net_g_latest.pth')

    # training
    ret = {'g': 0, 'c': 0, 'fg':0, 'sg':0}
    def compute_fg_metric(fg):
        b, t, c, h, w = fg.shape
        m = np.mean(np.sum(np.abs(fg).view(b, t, -1), axis=2), axis=0)
        m = m / m[t//2]
        return m, (np.sum(m)-1)/(t-1)

    def compute_sg_metric(sg):
        b, t, c, h, w = sg.shape
        m = np.mean(np.mean(np.abs(sg), axis=2), axis=0)
        m = m / m[t//2, h//2, w//2]
        return m, (np.sum(m)-1)/(t*h*w-1)

    total_count = 100
    for epoch in range(0, 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:

            model.feed_data(train_data)
            _b, _t, _c, _h, _w = model.lq.shape
            model.lq.requires_grad = True
            model.output = model.net_g(model.lq)

            # for grads among frequency
            loss = torch.sum(torch.abs(model.output - model.gt))
            loss.backward(retain_graph=True)
            g = model.lq.grad
            # g = torch.mean(torch.sum(torch.abs(g).view(_b, _t, -1), dim=2), dim=0)
            # g = g / g[_t//2]
            ret['fg'] += g.detach().cpu()
            
            # for grads among spatial
            loss = torch.sum(torch.abs(model.output[:, :, _h*4//2, _w*4//2] - model.gt[:, :, _h*4//2, _w*4//2]))
            loss.backward()
            g = model.lq.grad
            # g = torch.mean(torch.mean(torch.abs(g).view(_b, _t, _c, _h, _w), dim=2), dim=0)
            ret['sg'] += g.detach().cpu()

            ret['c'] += 1
            fg_metrices = compute_fg_metric(ret['fg'])
            sg_metrices = compute_sg_metric(ret['sg'])
            logger.info(f'total count {ret["c"]}, average fg metric: {fg_metrices[1]}, average sg metric: {sg_metrices[1]}')

            train_data = prefetcher.next()

            if ret['c'] >= total_count:
                break
        if ret['c'] >= total_count:
            break

    # logger.info(f'total count {ret["c"]}, average grads: {ret["g"]/ret["c"]}')

    import piclke
    with open(f'experiments/{exp_id}/fg.pickle', 'wb') as f:
        pickle.dump(ret['fg'].numpy(), f)
    with open(f'experiments/{exp_id}/sg.pickle', 'wb') as f:
        pickle.dump(ret['sg'].numpy(), f)

    fg_metrices = compute_fg_metric(ret['fg'])
    sg_metrices = compute_sg_metric(ret['sg'])
    logger.info(f'total count {ret["c"]}, average fg metric: {fg_metrices[1]}, average sg metric: {sg_metrices[1]}')
        
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
