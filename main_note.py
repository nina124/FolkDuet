#!/usr/bin/env python3
import argparse
import glog as log
import os
import torch.optim as optim
from datasets import make_loaders_and_note_dict
from utils import latest_checkpoint
from project_util import exp_preparation
import numpy as np
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Music generation')

    parser.add_argument('--exp_dir', default='results/tmp',
                        help='experiment directory')
    parser.add_argument('--nemb_pitch', type=int, default=128,
                        help='dimension of pitch embedding')
    parser.add_argument('--nemb_duration', type=int, default=64,
                        help='dimension of duration embedding')
    parser.add_argument('--nemb_meta', type=int, default=32,
                        help='dimension of meta embedding')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of GRU layers')
    parser.add_argument('--nfc_left', type=int, default=512,
                        help='hidden layer size after the GRU sequences')
    parser.add_argument('--meta_nfc_cent', type=int, default=32,
                        help='hidden layer size after the meta')
    parser.add_argument('--nfc_cent', type=int, default=128,
                        help='hidden layer size after the partner central pitch')
    parser.add_argument('--pred_nfc', type=int, default=512,
                        help='hidden layer size before the final prediction')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip_grad', type=float, default=1.,
                        help='clip to prevent the too large grad in GRU')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--step_at', nargs='+', default=[10, 15],
                        help='drop lr at specified epochs', )
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for training')
    parser.add_argument('--check_dir', type=str, default='',
                        help='restore model from')
    parser.add_argument('--arch', type=str, default='Generator',
                        help='model architecture to use')
    parser.add_argument('--init_len', type=int, default=10,
                        help='length of the initial segment')
    parser.add_argument('--only_config', action='store_true',
                        help='only load the model configurations of check_dir')
    parser.add_argument('--folk', action='store_true',
                        help='use folk dataset')
    parser.add_argument('--raw', action='store_true',
                        help='use the raw data, not transposed')
    args = parser.parse_args()
    args.step_at = [int(e) for e in args.step_at]
    return args


def test(model, loader):
    model.eval()
    test_loss, test_acc, test_pitch_acc, test_duration_acc, count = 0, 0, 0, 0, 0
    for batch_index, batch in enumerate(loader):
        with torch.no_grad():
            if args.arch == 'BachHM':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                partner_right, pr_length, pr_recover_idx, output = \
                    map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model(
                    [self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central,
                     partner_right, pr_length, pr_recover_idx])
            elif args.arch == 'BachM':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                output = map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, meta_central])
            elif args.arch == 'StyleRewarder':
                self_left, self_length, meta_central, output = \
                    map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, meta_central])
            elif args.arch == 'Generator':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                output = map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, partner_left, partner_length,
                                  recover_idx, partner_central, meta_central])
            else:
                raise NotImplementedError
            pitch_y, duration_y = output[..., 0], output[..., 1]
            target = pitch_y * args.num_durations + duration_y
            pred = logit.argmax(1)
            acc = (pred == target).to(torch.float32).mean().item()
            pitch_action = pred // args.num_durations
            duration_action = pred % args.num_durations
            loss = F.cross_entropy(logit, target)
            pitch_acc = (pitch_action == pitch_y).to(torch.float32).mean().item()
            duration_acc = (duration_action == duration_y).to(torch.float32).mean().item()
            test_loss += loss
            test_acc += acc
            test_pitch_acc += pitch_acc
            test_duration_acc += duration_acc
        count += 1
    log.info('VALID: epoch_loss: %.4f, epoch_acc: %.4f' % (test_loss / count, test_acc / count))
    log.info('VALID: epoch_pitch_acc: %.4f, epoch_duration_acc: %.4f' %
             (test_pitch_acc / count, test_duration_acc / count))
    return test_acc / count, test_pitch_acc / count, test_duration_acc / count


def train(model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    for epoch_idx in range(args.epochs):
        log.info('Training for epoch %d' % epoch_idx)

        if epoch_idx in args.step_at:
            for group_idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                new_lr = lr * 0.1
                param_group['lr'] = new_lr
                log.info('Epoch %d, param group %d, %d params, cut learning rate from %f to %f' %
                         (epoch_idx, group_idx, len(param_group['params']), lr, new_lr))

        epoch_loss, epoch_acc, epoch_pitch_acc, epoch_duration_acc = 0, 0, 0, 0
        log.info('Epoch %d' % epoch_idx)
        model.train()
        count = 0
        for batch_index, batch in enumerate(train_loader):
            if args.arch == 'BachHM':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                partner_right, pr_length, pr_recover_idx, output = \
                    map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model(
                    [self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central,
                     partner_right, pr_length, pr_recover_idx])
            elif args.arch == 'BachM':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                output = map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, meta_central])
            elif args.arch == 'StyleRewarder':
                self_left, self_length, meta_central, output = \
                    map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, meta_central])
            elif args.arch == 'Generator':
                self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, \
                output = map(lambda item: None if item is None else item.cuda(), batch)
                logit, _ = model([self_left, self_length, partner_left, partner_length,
                                  recover_idx, partner_central, meta_central])
            else:
                raise NotImplementedError
            pitch_y, duration_y = output[..., 0], output[..., 1]
            target = pitch_y * args.num_durations + duration_y
            total_pred = logit.argmax(1)
            total_acc = (total_pred == target).to(torch.float32).mean().item()
            pitch_action = total_pred // args.num_durations
            duration_action = total_pred % args.num_durations
            loss = F.cross_entropy(logit, target)
            pitch_acc = (pitch_action == pitch_y).to(torch.float32).mean().item()
            duration_acc = (duration_action == duration_y).to(torch.float32).mean().item()
            loss.backward()
            epoch_loss += loss
            epoch_acc += total_acc
            epoch_pitch_acc += pitch_acc
            epoch_duration_acc += duration_acc

            # Clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, norm_type=2)

            # Optimize
            optimizer.step()
            optimizer.zero_grad()
            count += 1

        log.info('TRAIN: epoch_loss: %.4f, epoch_acc: %.4f' % (epoch_loss / count, epoch_acc / count))
        log.info('TRAIN: epoch_pitch_acc: %.4f, epoch_duration_acc: %.4f' %
                 (epoch_pitch_acc / count, epoch_duration_acc / count))

        test(model, valid_loader)

        # save model
        fname = os.path.join(args.exp_dir, 'epoch_%d.model' % epoch_idx)
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        torch.save(model.state_dict(), fname)
        log.info('Model at epoch %d saved to %s' % (epoch_idx, fname))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = eval('%s(args)' % args.arch)
    model = model.cuda()

    if args.check_dir:
        if not args.only_config:
            checkpoint = latest_checkpoint(args.check_dir)
            model.load_state_dict(torch.load(checkpoint), False)
            log.info('Model loaded from %s' % checkpoint)
    log.info('Start training')
    train(model)
    log.info('Training done')


if __name__ == '__main__':
    args = parse_args()
    args = exp_preparation(args)

    # Load data loaders and token dict
    loaders = make_loaders_and_note_dict(args, folk_data=args.folk, phases=('train', 'valid'))
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    main()
