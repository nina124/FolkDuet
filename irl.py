#!/usr/bin/env python3
import argparse
import glog as log
import os
import torch.optim as optim
from datasets import IRLLoaderNote, prepare_folk_note_data
from utils import latest_checkpoint, weights_init
from reward_model import IRLStyleRewardModel, train_folk_reward_model, MIRewardModel
from project_util import exp_preparation
import numpy as np
from sample import note_sample
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Music generation')

    parser.add_argument('--exp_dir', default='results/irl',
                        help='experiment directory')
    parser.add_argument('--arch', type=str, default='Generator',
                        help='model architecture to use')
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
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip_grad', type=float, default=1.,
                        help='clip to prevent the too large grad in GRU')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--lr_reward', type=float, default=1e-4,
                        help='learning rate for the reward_model')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=192,
                        help='batch size for training')
    parser.add_argument('--bach_both', type=str, default='',
                        help='bach both model checkpoint')
    parser.add_argument('--bach_self', type=str, default='',
                        help='bach self model checkpoint')
    parser.add_argument('--reward_dir', type=str, default='',
                        help='reward model checkpoint')
    parser.add_argument('--check_dir', type=str, default='',
                        help='restore model from')
    parser.add_argument('--init_len', type=int, default=10,
                        help='length of the initial segment')
    parser.add_argument('--only_config', action='store_true',
                        help='only load the model configurations of check_dir')
    parser.add_argument('--gamma', type=float, default=1.,
                        help='discount factor for RL')
    parser.add_argument('--entropy_beta', type=float, default=0.05,
                        help='entropy regularization weight for RL')
    parser.add_argument('--inter_alpha', type=float, default=0.5,
                        help='weight for the InterRewarder')
    parser.add_argument('--raw', action='store_true',
                        help='use the raw data, not transposed')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='note sequence length for RL training')
    args = parser.parse_args()
    return args


def ac_learn(model, optimizer, values, log_probs, entropies, rewards):
    tau = 1.
    poclicy_loss = 0.
    value_loss = 0.
    for value, log_prob, entropy, reward in zip(values, log_probs, entropies, rewards):
        R = value[-1]
        gae = torch.zeros(1).cuda()
        for i in reversed(range(len(reward))):
            R = args.gamma * R + reward[i].item()
            advantage = R - value[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = args.gamma * value[i + 1].data - value[i].data + reward[i].item()
            gae = gae * args.gamma * tau + delta_t
            poclicy_loss = poclicy_loss - log_prob[i] * gae - args.entropy_beta * entropy[i]

    loss = poclicy_loss + 0.5 * value_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, norm_type=2)
    optimizer.step()
    model.zero_grad()
    return poclicy_loss.item(), value_loss.item(), grad_norm


def train(model):
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, amsgrad=True)

    batch_count, train_count_d = 0, 0
    epoch_idx = 0
    while epoch_idx < args.epochs:
        epoch_idx += 1
        log.info('Training for epoch %d' % epoch_idx)
        log.info('Generator sampling data')
        train_loader.generate_data(model)
        log.info('Reset rewarder fc parameters')
        reward_agents.apply(weights_init)
        log.info('Generator finish sampling data')
        reward_agents.train()
        for r_epoch_idx in range(4):
            rewards = np.zeros(2)
            losses = np.zeros(2)
            count = 0
            for batch in train_loader.get_rewarder_samples():
                reward, loss = train_folk_reward_model(reward_agents.model, reward_optimizer, batch,
                                                       pretrain=epoch_idx == 1)
                rewards[:] += reward
                count += args.batch_size // 2
            rewards /= (count * args.seq_length)
            losses /= args.seq_length
            log.info('generation reward %.4f' % (rewards[1]))
            log.info('train_data reward %.4f' % (rewards[0]))
        reward_agents.eval()

        model.train()
        batch_greedy_acc, batch_steps, batch_reward_mi, batch_reward_model = 0, 0, 0, 0
        batch_p_loss, batch_v_loss, batch_grad_norm = 0, 0, 0
        last_train_count, train_count = 0, 0
        samples, batch_stop, epoch_stop = train_loader.batch_step(None)
        values = []
        log_probs = []
        entropies = []
        rewards = []
        for _ in range(args.batch_size):
            values.append([])
            log_probs.append([])
            entropies.append([])
            rewards.append([])

        rl_index2value_index = {r: i for i, r in enumerate(train_loader.rl_index)}
        prev_rl_index = train_loader.rl_index.copy()
        while True:
            if batch_stop:
                for s, ind in enumerate(prev_rl_index):
                    index = rl_index2value_index[ind]
                    values[index].append(torch.zeros(1).cuda())
                p_loss, v_loss, grad_norm = \
                    ac_learn(model, optimizer, values, log_probs, entropies, rewards)
                batch_p_loss += p_loss
                batch_v_loss += v_loss
                batch_grad_norm += grad_norm
                train_count += 1
                log.info('train-count: %d' % train_count)
                log.info('p_loss: %.4f' % (batch_p_loss/batch_steps))
                log.info('v_loss: %.4f' % (batch_v_loss/batch_steps))
                log.info('grad_norm: %.4f' % (batch_grad_norm/(train_count - last_train_count)))
                log.info('model reward: %f' % (batch_reward_model / batch_steps))
                log.info('reward_mi_model: %f' % (batch_reward_mi.mean() / batch_steps))
                log.info('greedy_acc: %.4f' % (batch_greedy_acc / batch_steps))
                log.info('')
                batch_count += 1
                last_train_count = train_count
                if epoch_stop:
                    break
                else:
                    samples, batch_stop, epoch_stop = train_loader.batch_step(None)
                    values = []
                    log_probs = []
                    entropies = []
                    rewards = []
                    for _ in range(len(samples[0])):
                        values.append([])
                        log_probs.append([])
                        entropies.append([])
                        rewards.append([])
                    rl_index2value_index = {r: i for i, r in enumerate(train_loader.rl_index)}
                    prev_rl_index = train_loader.rl_index.copy()
                    batch_greedy_acc, batch_steps, batch_reward_mi, batch_reward_model = 0, 0, 0, 0
                    batch_p_loss, batch_v_loss, batch_grad_norm = 0, 0, 0
                # torch.cuda.empty_cache()
            else:
                self_left, self_length, partner_left, partner_length, recover_idx, \
                    partner_central, meta_central, output, partner_right, pr_length, pr_recover_idx = samples

                logit, value = model([self_left, self_length, partner_left, partner_length,
                                      recover_idx, partner_central[..., 0], meta_central])
                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)
                entropy = -(log_prob * prob).sum(1)
                action = prob.multinomial(num_samples=1).data
                pitch_action = action // args.num_durations
                duration_action = action % args.num_durations
                log_prob = log_prob.gather(1, action)
                for i, ind in enumerate(train_loader.rl_index):
                    ind = rl_index2value_index[ind]
                    values[ind].append(value[i])
                    log_probs[ind].append(log_prob[i])
                    entropies[ind].append(entropy[i])
                prev_pred = torch.cat([pitch_action.detach(), duration_action.detach()], -1)
                samples, batch_stop, epoch_stop = train_loader.batch_step(prev_pred)

                # modelagent
                inp = [self_left, self_length, partner_left, partner_length, recover_idx,
                       partner_central[..., 0], meta_central]
                reward_models = reward_agents.get_batch_reward(inp, action)
                reward = reward_models.mean(1)  # [B]

                # mutula information reward
                mi_reward_models = mi_reward_agents.get_batch_reward(
                    [self_left, self_length, partner_left, partner_length, recover_idx,
                     partner_central, meta_central, partner_right, pr_length, pr_recover_idx],
                    action)

                reward += args.inter_alpha * mi_reward_models  # [B]

                for i, ind in enumerate(train_loader.rl_index):
                    ind = rl_index2value_index[ind]
                    rewards[ind].append(reward[i])

                batch_reward_model += reward_models.sum(0)
                batch_reward_mi += mi_reward_models.sum(0)

                batch_steps += len(reward)

        if epoch_idx % 10 == 0:
            note_sample(model, args, 'Epoch%d-sample' % epoch_idx, count=10, greedy=False)
            fname = os.path.join(args.exp_dir, 'epoch_%d.model' % epoch_idx)
            if not os.path.exists(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
            torch.save(model.state_dict(), fname)
            fname = os.path.join(args.exp_dir, 'rewarder_epoch_%d.model' % epoch_idx)
            torch.save(reward_agents.model.state_dict(), fname)
            log.info('Model at epoch %d saved to %s' % (epoch_idx, fname))
        model.train()

    fname = os.path.join(args.exp_dir, 'epoch_%d.model' % epoch_idx)
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    torch.save(model.state_dict(), fname)
    fname = os.path.join(args.exp_dir, 'rewarder_epoch_%d.model' % epoch_idx)
    torch.save(reward_agents.model.state_dict(), fname)
    log.info('Model at epoch %d saved to %s' % (epoch_idx, fname))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = eval('%s(args)' % args.arch)
    model = model.cuda()

    if args.check_dir:
        if not args.only_config:
            checkpoint = latest_checkpoint(args.check_dir)
            model.load_state_dict(torch.load(checkpoint), True)
            log.info('Model loaded from %s' % checkpoint)

    log.info('Start training')
    train(model)
    log.info('Training done')


if __name__ == '__main__':
    args = parse_args()
    args.folk = True
    assert args.dropout == 0, 'do not use dropout in RL'
    args = exp_preparation(args)

    # Load data loaders and token dict
    log.info('Load reward model:')
    reward_agents = IRLStyleRewardModel(args.reward_dir)
    rewarder_vars = [p for name, p in reward_agents.model.named_parameters() if 'encoders' not in name]
    reward_optimizer = optim.SGD(rewarder_vars, lr=args.lr_reward, momentum=0.9)
    mi_reward_agents = MIRewardModel(args.bach_self, args.bach_both)
    self_feature, _ = prepare_folk_note_data(args, 'train', raw=args.raw, segment=False)
    partner_feature = None
    train_loader = IRLLoaderNote(self_feature, partner_feature, args)

    main()
