from project_util import get_model
from models import *
import torch


class IRLStyleRewardModel(object):
    def __init__(self, reward_checkdir):
        self.reward_check_dir = reward_checkdir
        self.model, self.reward_args = get_model(reward_checkdir, load_weight=True)
        self.model.eval()

    def get_batch_reward(self, model_input, action):
        with torch.no_grad():
            self_left = model_input[0]
            logit = self.model.reward(self_left)
            logit = F.sigmoid(logit)
            reward = torch.gather(logit, 1, action).cpu().numpy()
        return reward

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def apply(self, func):
        self.model.apply(func)


class MIRewardModel(object):
    def __init__(self, bach_self, bach_both):
        model, args = get_model(bach_self)
        model.eval()
        self.bach_self_model = model
        model, args = get_model(bach_both)
        model.eval()
        self.bach_both_model = model

    def get_batch_reward(self, model_input, action, skip=False):
        with torch.no_grad():
            if not skip:
                for i in [0, 2, 5, 7]:
                    item = model_input[i].clone()
                    if len(item) > 0:
                        item[..., 0] = item[..., 0] + 12
                    model_input[i] = item
            both_logit, _ = self.bach_both_model(model_input)
            self_logit, _ = self.bach_self_model([model_input[0], model_input[1], model_input[6]])

            self_reward = torch.gather(torch.log(torch.clamp(F.softmax(self_logit, -1), min=1e-20, max=1.)), 1, action)
            both_reward = torch.gather(torch.log(torch.clamp(F.softmax(both_logit, -1), min=1e-20, max=1.)), 1, action)

            reward = torch.clamp(both_reward - self_reward, min=-1., max=1.)
            return reward[:, 0]


def train_folk_reward_model(reward_model, reward_optimizer, batch, num_durations=10, pretrain=False, temperature=1):
    self_left, target, logp, mask = batch
    pitch_y, duration_y = target[..., 0], target[..., 1]
    target = pitch_y * num_durations + duration_y
    half_bs = len(self_left) // 2

    reward_model.eval()
    reward = reward_model.reward(self_left[half_bs:], return_seq=True)
    reward = F.sigmoid(reward)
    reward = torch.sum(reward.gather(-1, target[half_bs:, :, None])[..., 0] * mask[half_bs:], 1)
    if pretrain:
        weight = torch.cat([torch.full((half_bs,), 1./half_bs, dtype=torch.float).cuda(),
                            -torch.full((half_bs,), 1./half_bs, dtype=torch.float).cuda()], 0)
    else:
        weight = torch.cat([torch.full((half_bs,), 1./half_bs, dtype=torch.float).cuda(),
                            -F.softmax((reward - logp)/temperature, 0)], 0)

    reward_model.train()
    pred = reward_model.reward(self_left, return_seq=True)
    pred = F.sigmoid(pred)
    reward = torch.sum(pred.gather(-1, target[..., None])[..., 0] * mask, 1)
    loss_seq = -reward * weight.data  # [B,]
    loss = loss_seq.sum()
    loss.backward()
    reward_optimizer.step()
    reward_optimizer.zero_grad()
    return map(lambda item: item.view(2, -1).sum(1).data.cpu().numpy(), [reward, loss_seq])

