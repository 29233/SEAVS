import torch
import time
import torch.nn
import os
import random
import numpy as np
from mmcv import Config
import argparse
import sys
from utils import pyutils
from utils.loss_util import LossUtil
from utility import mask_iou
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from loss import Loss
from utility import Eval_Fmeasure


def main():
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, dir_name)):
        os.mkdir(os.path.join(args.log_dir, dir_name))
    log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)
    checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)

    # model
    model = build_model(**cfg.model).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    model.train()
    # model.module.backbone.eval()
    logger.info("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    logger.info("Trainable params: %.2fM" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    logger.info("Decoder params: %.2fM" % (sum(p.numel() for p in model.decoder.parameters()) / 1e6))
    # log the computation cost
    # flops, params = profile(model, inputs=(input_a, input_v))
    # logger.info('FLOPs: %.2fG, Params: %.2fM' % (flops / 1e9, params / 1e6))
    # dataset
    train_dataset = build_dataset(**cfg.dataset.train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.dataset.train.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.process.num_works,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // cfg.dataset.train.batch_size) * \
               cfg.process.train_epochs
    val_dataset = build_dataset(**cfg.dataset.val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.dataset.val.batch_size,
                                                 shuffle=False,
                                                 num_workers=cfg.process.num_works,
                                                 pin_memory=True)

    # optimizer
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    loss_util = LossUtil(**cfg.loss)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_fscore = pyutils.AverageMeter('F_score')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    Fscore_list = []
    max_Fscore = 0
    for epoch in range(cfg.process.train_epochs):
        # if epoch == cfg.process.freeze_epochs:
        #     model.module.freeze_backbone(False)

        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, _ = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask_num = 5
            mask = mask.view(B * mask_num, 1, H, W)

            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            output = model(audio, imgs)
            loss, loss_dict = Loss(
                output, mask, cfg.loss.weight_dict)
            # loss, loss_dict = Context_align_Loss(output, mask, audio_context, visual_context, cfg.loss.weight_dict)
            loss_util.add_loss(loss, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                    global_step - 1, max_step, loss_util.pretty_out(), optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                if epoch == cfg.process.freeze_epochs:
                    model.module.tune_backbone()
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs, audio, mask, _ = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2],
                                   audio.shape[3], audio.shape[4])

                # [bs*5, 1, 224, 224]
                output = model(audio, imgs)

                miou = mask_iou(output.squeeze(1), mask)
                Fscore = Eval_Fmeasure(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})
                avg_meter_fscore.add({'F_score': Fscore})

            miou = (avg_meter_miou.pop('miou'))
            Fscore = (avg_meter_fscore.pop('F_score'))
            if miou > max_miou or Fscore + miou > max_Fscore + max_miou:
                model_save_path = os.path.join(
                    checkpoint_dir,
                    '{}_best_epoch{}_lr_{}_{}.pth'.format(args.session_name, cfg.process.train_epochs, cfg.optimizer.lr,
                                                          cfg.discribe.info))
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            Fscore_list.append(Fscore)
            max_Fscore = max(Fscore_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}, Fscore:{}, max_Fscore:{}'.format(
                epoch, miou, max_miou, Fscore, max_Fscore)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str,
                        default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='work_dir', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="MS3",
                        type=str, help="the MS3 setting")

    args = parser.parse_args()
    main()
