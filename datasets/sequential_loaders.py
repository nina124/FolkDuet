import numpy as np
from datasets import get_note_meta, seq_sort_by_length_and_pad
import torch.nn.functional as F
import torch


class LoaderNote(object):
    def __init__(self, self_feature, partner_feature, args):
        self.dataset_self = np.array(self_feature)
        if partner_feature is not None:
            self.dataset_partner = np.array(partner_feature)
        else:
            self.dataset_partner = None
        self.train_dataset_self = None
        self.train_dataset_partner = None
        self.size = len(self.dataset_self)
        self.bs = args.batch_size
        self.init_len = args.init_len
        self.sample_ind = 0
        self.self_pred_index = np.array([self.init_len] * self.bs)
        self.self_pred_offsets = [0] * self.bs
        self.partner_pred_index = None
        self.partner_offsets = None
        self.total_offsets = None
        self.sequence_length = args.seq_length
        self.current_samples = []
        self.epoch_index = -1
        self.prev_rl_index = None
        self.rl_index = []
        index2pitch, pitch2index, index2duration, _ = get_note_meta(args.folk)
        self.index2pitch = index2pitch
        self.index2duration = index2duration
        self.rest_ind = pitch2index['rest']
        self.meta = torch.arange(16).view(1, 16).cuda()
        self.clip = False

    def batch_step(self, prev_pred=None):
        if self.sample_ind == 0:
            self.shuffle()
        skip_offset = False
        if len(self.rl_index) == 0:
            self.current_samples = [[self.train_dataset_self[i].astype(np.long),
                                     self.train_dataset_partner[i].astype(np.long)]
                                    for i in range(self.sample_ind, min(len(self.train_dataset_self),
                                                                        self.sample_ind + self.bs))]
            self.sample_ind += len(self.current_samples)
            self.self_pred_index = np.full((len(self.current_samples),), self.init_len)
            self.self_pred_offsets = [list(np.cumsum([0, ] + [self.index2duration[i] for i in sample[0][:pred_ind, 1]]))
                                      for sample, pred_ind in zip(self.current_samples, self.self_pred_index)]
            skip_offset = True
            self.partner_offsets = [list(np.cumsum([0, ] + [self.index2duration[i] for i in sample[1][:, 1]]))
                                    for sample in self.current_samples]
            self.partner_pred_index = [sum(self.partner_offsets[i] < self.self_pred_offsets[i][-1])
                                       for i in range(len(self.current_samples))]
            self.total_offsets = [sum([self.index2duration[i] for i in sample[1][:, 1]])
                                  for sample in self.current_samples]
            self.rl_index = list(range(len(self.current_samples)))

        self_left, self_length, partner_left, partner_length, partner_central, meta_central, \
            partner_right, pr_length, output = [], [], [], [], [], [], [], [], []
        self.prev_rl_index = self.rl_index.copy()
        for s, i in enumerate(self.prev_rl_index):
            pred_ind = self.self_pred_index[i]
            if prev_pred is not None:
                self.current_samples[i][0][pred_ind-1] = prev_pred[s].cpu().numpy()  # pitch, duration
            prev_pred_s = self.current_samples[i][0][pred_ind - 1]
            duration = self.index2duration[prev_pred_s[1]]
            if not skip_offset:
                self.self_pred_offsets[i].append(self.self_pred_offsets[i][-1] + duration)
            pred_offset = self.self_pred_offsets[i][-1]
            while self.partner_pred_index[i] < len(self.current_samples[i][1]) \
                    and self.partner_offsets[i][self.partner_pred_index[i]] <= pred_offset:
                self.partner_pred_index[i] += 1
            if (self.clip and pred_ind == self.sequence_length + self.init_len) or \
                    (not self.clip and pred_offset >= self.total_offsets[i]):
                self.rl_index.remove(i)
                self.current_samples[i] = [self.current_samples[i][0][:pred_ind],
                                           self.current_samples[i][1][:self.partner_pred_index[i]]]
                self.self_pred_index[i] += 1
                continue

            if pred_ind == len(self.current_samples[i][0]):
                self.current_samples[i][0] = np.concatenate(
                    [self.current_samples[i][0], self.current_samples[i][0][-1:]], 0)
            meta_central.append(self.meta[0, int(4*pred_offset) % 16])
            self_left.append(self.current_samples[i][0][:pred_ind])
            self_length.append(len(self_left[-1]))
            partner_ind = self.partner_pred_index[i]
            last_partner_duration = self.index2duration[self.current_samples[i][1][partner_ind - 1, 1]]
            if self.partner_offsets[i][self.partner_pred_index[i]] - last_partner_duration >= pred_offset:
                partner_ind -= 1
            partner_left.append(self.current_samples[i][1][:(partner_ind-1)])
            partner_length.append(partner_ind-1)
            partner_central.append(self.current_samples[i][1][partner_ind - 1])
            partner_right.append(self.current_samples[i][1][:partner_ind-1-len(self.current_samples[i][1]):-1])
            pr_length.append(len(partner_right[-1]))
            output.append(self.current_samples[i][0][pred_ind])
            self.self_pred_index[i] += 1
        if len(self.rl_index) == 0:
            if self.sample_ind >= len(self.train_dataset_self):
                self.sample_ind = 0
                return None, True, True
            return None, True, False

        partner_left, partner_length, recover_idx = seq_sort_by_length_and_pad(partner_left, partner_length)
        partner_right, pr_length, pr_recover_idx = seq_sort_by_length_and_pad(partner_right, pr_length)
        return list(map(lambda item: torch.from_numpy(np.array(item).astype(np.long)).cuda(),
                        [self_left, self_length, partner_left, partner_length, recover_idx, partner_central,
                         meta_central, output, partner_right, pr_length, pr_recover_idx])), False, False

    def shuffle(self):
        ind = np.arange(len(self.dataset_self))
        np.random.shuffle(ind)
        if self.dataset_partner is None:
            gap = 12
            self.train_dataset_self = []
            self.train_dataset_partner = []
            for i in range(len(self.dataset_self)):
                pitch, duration = self.dataset_self[ind[i]][:, 0].copy(), self.dataset_self[ind[i]][:, 1].copy()
                if self.clip:
                    max_clip = len(pitch) - self.sequence_length - self.init_len + 1
                    if max_clip <= 0:
                        continue
                    clip_pos = np.random.randint(0, max_clip)
                    pitch = pitch[clip_pos:]
                    duration = duration[clip_pos:]
                bass_pitch = pitch.copy()
                min_shift = -pitch.min()
                max_shift = self.rest_ind - pitch[pitch != self.rest_ind].max()
                if max_shift - min_shift <= gap:
                    continue
                else:
                    min_shift = np.random.randint(max(-12, min_shift), min(max_shift - gap, 12))
                    max_shift = min_shift + gap
                    pitch[pitch != self.rest_ind] += max_shift
                    bass_pitch[bass_pitch != self.rest_ind] += min_shift

                if np.random.random() < 0.5:
                    self.train_dataset_self.append(np.stack([pitch, duration], -1))
                    self.train_dataset_partner.append(np.stack([bass_pitch, duration], -1))
                else:
                    self.train_dataset_partner.append(np.stack([pitch, duration], -1))
                    self.train_dataset_self.append(np.stack([bass_pitch, duration], -1))
        else:
            self.train_dataset_self = []
            self.train_dataset_partner = []
            for i in range(len(self.dataset_self)):
                pitch, duration = self.dataset_self[ind[i]][:, 0].copy(), self.dataset_self[ind[i]][:, 1].copy()
                partner_pitch, partner_duration = self.dataset_partner[ind[i]][:, 0].copy(), \
                                                  self.dataset_partner[ind[i]][:, 1].copy()
                if self.clip:
                    offset = np.cumsum([0, ] + [self.index2duration[i] for i in duration[:-1]]) * 4
                    partner_offset = np.cumsum([0, ] + [self.index2duration[i] for i in partner_duration[:-1]]) * 4
                    max_clip = len(pitch) - self.init_len - self.sequence_length + 1
                    if max_clip <= 0:
                        continue
                    clip_pos = np.random.randint(0, max_clip)
                    while offset[clip_pos] not in partner_offset:
                        clip_pos = np.random.randint(0, max_clip)
                    pitch = pitch[clip_pos:]
                    duration = duration[clip_pos:]
                    partner_clip_pos = np.where(partner_offset == offset[clip_pos])[0].item()
                    partner_pitch = partner_pitch[partner_clip_pos:]
                    partner_duration = partner_duration[partner_clip_pos:]

                self.train_dataset_self.append(np.stack([pitch, duration], -1))
                self.train_dataset_partner.append(np.stack([partner_pitch, partner_duration], -1))
        self.epoch_index += 1
        self.rl_index = []


class IRLLoaderNote(LoaderNote):
    def __init__(self, self_feature, partner_feature, args):
        super(IRLLoaderNote, self).__init__(self_feature, partner_feature, args)
        self.negative_samples_self = []
        self.negative_sample_logp = []
        self.clip = True

    def generate_data(self, generator_model):
        self.sample_ind = 0
        old_bs = self.bs
        self.bs = 256
        samples, batch_stop, epoch_stop = self.batch_step(None)
        self.negative_samples_self = []
        self.negative_sample_logp = []
        negative_log_probs = []
        for i in range(len(self.rl_index)):
            negative_log_probs.append([])
        while True:
            if batch_stop:
                self.negative_samples_self.extend([sample[0] for sample in self.current_samples])
                self.negative_sample_logp.extend(negative_log_probs)
                if epoch_stop:
                    break
                else:
                    samples, batch_stop, epoch_stop = self.batch_step(None)
                    negative_log_probs = []
                    for i in range(len(self.rl_index)):
                        negative_log_probs.append([])
            else:
                self_left, self_length, partner_left, partner_length, recover_idx, \
                    partner_central, meta_central, output = samples[:8]
                with torch.no_grad():
                    logit, _ = generator_model([self_left, self_length, partner_left, partner_length,
                                                recover_idx, partner_central[..., 0], meta_central])
                    prob = F.softmax(logit, dim=-1)
                    action = prob.multinomial(num_samples=1).data
                    log_prob = torch.gather(F.log_softmax(logit, dim=-1), -1, action).squeeze(-1)
                duration_action = action % len(self.index2duration)
                pitch_action = action // len(self.index2duration)
                prev_pred = torch.cat([pitch_action.detach(), duration_action.detach()], -1)
                for s, ind in enumerate(self.rl_index):
                    negative_log_probs[ind].append(log_prob[s].data)
                samples, batch_stop, epoch_stop = self.batch_step(prev_pred)
        self.bs = old_bs

    def get_rewarder_samples(self):
        ind = np.arange(len(self.dataset_self))
        np.random.shuffle(ind)
        self.dataset_self = self.dataset_self[ind]
        positive_ind, negative_ind = 0, 0
        bs = self.bs // 2

        def prepare_batch(start_ind, positive):
            if positive:
                dataset = self.dataset_self
            else:
                dataset = self.negative_samples_self
            self_parts, targets, logp = [], [], []
            while len(self_parts) < bs:
                sample = dataset[start_ind].copy()

                if len(sample) < self.init_len + self.sequence_length:
                    start_ind += 1
                    continue
                if positive:
                    max_clip = len(sample) - self.init_len - self.sequence_length + 1
                    clip_pos = np.random.randint(0, max_clip)
                    sample = sample[clip_pos:clip_pos + self.sequence_length + self.init_len]
                    pitch = sample[:, 0]
                    min_shift = -pitch.min()
                    max_shift = self.rest_ind - pitch[pitch != self.rest_ind].max()
                    min_shift = np.random.randint(max(-12, min_shift), min(max_shift - 12, 12))
                    pitch[pitch != self.rest_ind] += min_shift
                    sample[:, 0] = pitch
                else:
                    logp.append(torch.stack(self.negative_sample_logp[start_ind]).sum())
                start_ind += 1
                self_parts.append(sample[:-1])
                targets.append(sample[1:])
            self_parts, targets = \
                map(lambda item: torch.from_numpy(np.array(item)).cuda(), [self_parts, targets])
            if positive:
                return self_parts, targets, start_ind
            else:
                logp = torch.from_numpy(np.array(logp)).cuda()
                return self_parts, targets, logp, start_ind

        while positive_ind + bs < len(self.train_dataset_self) and negative_ind + bs < len(self.negative_samples_self):
            positive_self, positive_target, positive_ind = prepare_batch(positive_ind, positive=True)
            negative_self, negative_target, logp, negative_ind = prepare_batch(negative_ind, positive=False)
            batch = [torch.cat([positive_self, negative_self], 0), torch.cat([positive_target, negative_target], 0),
                     logp]
            mask = torch.zeros((batch[0].shape[:2])).cuda()
            mask[:, -self.sequence_length:] = 1.
            batch.append(mask)
            yield batch

