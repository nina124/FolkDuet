from torch.utils.data import Dataset
from datasets import prepare_bach_note_data, prepare_folk_note_data
from collections import OrderedDict
import numpy as np
import torch
import glog


class BachDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return (self.data[i][index] for i in range(self.length))

    def __len__(self):
        return len(self.data[0])


def seq_sort_by_length_and_pad(part, length):
    index = sorted(range(len(length)), key=lambda idx: length[idx], reverse=True)
    recover_idx = sorted(range(len(index)), key=lambda idx: index[idx])
    part, length = map(lambda item: [item[idx] for idx in index], [part, length])
    if length[-1] == 0:
        zero_len = sum(np.array(length) == 0)
        length = length[:-zero_len]
        part = part[:-zero_len]
    if len(length) > 0:
        max_length = max(length)
        part = list(map(lambda item: np.concatenate([item, np.full((max_length - len(item), item.shape[1]),
                                                                   0, dtype=np.long)]), part))
    return part, length, recover_idx


def pad_collate_note_single_part(batch):
    self_left, meta_central, output = map(list, zip(*batch))
    self_length = [len(i) for i in self_left]
    index = sorted(range(len(self_length)), key=lambda idx: self_length[idx], reverse=True)
    self_left, self_length, meta_central, output = \
        map(lambda item: [item[idx] for idx in index], [self_left, self_length, meta_central, output])
    max_length = max(self_length)
    self_left = list(map(lambda item: np.concatenate([item, np.full((max_length-len(item), item.shape[1]),
                                                                    0, dtype=np.long)]), self_left))
    output = [i[-1] for i in output]
    meta_central = [i[-1] for i in meta_central]
    self_left, self_length, meta_central, output = map(lambda item: torch.from_numpy(np.array(item).astype(np.long)),
                                                       [self_left, self_length, meta_central, output])
    return self_left, self_length, meta_central, output


def pad_collate_note_tow_parts(batch):
    batch = list(map(list, zip(*batch)))
    self_left, partner_left, partner_central, meta_central, output = batch
    output = [i[-1] for i in output]
    self_length = [len(i) for i in self_left]
    partner_length = [len(i) for i in partner_left]
    index = sorted(range(len(self_length)), key=lambda idx: self_length[idx], reverse=True)
    self_left, partner_left, self_length, partner_length, partner_central, meta_central, output = \
        map(lambda item: [item[idx] for idx in index] if item is not None else None,
            [self_left, partner_left, self_length, partner_length, partner_central, meta_central, output])
    max_length = max(self_length)
    self_left = list(map(lambda item: np.concatenate([item, np.full((max_length-len(item), item.shape[1]),
                                                                    0, dtype=np.long)]), self_left))
    partner_left, partner_length, recover_idx = seq_sort_by_length_and_pad(partner_left, partner_length)

    self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, output = \
        map(lambda item: torch.from_numpy(np.array(item).astype(np.long)) if item is not None else None,
            [self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, output])
    return self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central, output


def pad_collate_note_three_seq(batch):
    self_left, partner_left, partner_central, meta_central, partner_right, output = map(list, zip(*batch))
    output = [i[-1] for i in output]
    self_length = [len(i) for i in self_left]
    pl_length = [len(i) for i in partner_left]
    pr_length = [len(i) for i in partner_right]
    index = sorted(range(len(self_length)), key=lambda idx: self_length[idx], reverse=True)
    self_left, partner_left, partner_right, self_length, pl_length, pr_length, partner_central, meta_central, output = \
        map(lambda item: [item[idx] for idx in index],
            [self_left, partner_left, partner_right, self_length, pl_length, pr_length,
             partner_central, meta_central, output])
    max_length = max(self_length)
    self_left = list(map(lambda item: np.concatenate([item, np.full((max_length-len(item), item.shape[1]),
                                                                    0, dtype=np.long)]), self_left))
    partner_left, pl_length, pl_recover_idx = seq_sort_by_length_and_pad(partner_left, pl_length)
    partner_right, pr_length, pr_recover_idx = seq_sort_by_length_and_pad(partner_right, pr_length)

    self_left, self_length, partner_left, pl_length, pl_recover_idx, partner_central, meta_central, \
    partner_right, pr_length, pr_recover_idx, output = \
        map(lambda item: torch.from_numpy(np.array(item).astype(np.long)),
            [self_left, self_length, partner_left, pl_length, pl_recover_idx, partner_central, meta_central,
             partner_right, pr_length, pr_recover_idx, output])
    return self_left, self_length, partner_left, pl_length, pl_recover_idx, partner_central, meta_central, \
           partner_right, pr_length, pr_recover_idx, output


def make_loaders_and_note_dict(args, folk_data=False, phases=('train', 'test')):
    loaders = dict()
    for phase in phases:
        if folk_data:
            if args.arch == 'StyleRewarder':
                self_left, meta_central, output = prepare_folk_note_data(args, phase, raw=args.raw or phase == 'valid')
                dataset = BachDataset([self_left, meta_central, output])
                collate_fn = pad_collate_note_single_part
            else:
                self_left, partner_left, partner_central, meta_central, output = \
                    prepare_folk_note_data(args, phase, raw=args.raw)
                dataset = BachDataset([self_left, partner_left, partner_central, meta_central, output])
                collate_fn = pad_collate_note_tow_parts
        else:
            if args.arch == 'BachHM':
                self_left, partner_left, partner_central, meta_central, partner_right, output = \
                    prepare_bach_note_data(args, phase, raw=phase == 'valid')
                dataset = BachDataset(
                    [self_left, partner_left, partner_central, meta_central, partner_right, output])
                collate_fn = pad_collate_note_three_seq
            else:
                self_left, partner_left, partner_central, meta_central, output = \
                    prepare_bach_note_data(args, phase, raw=phase == 'valid')
                dataset = BachDataset([self_left, partner_left, partner_central, meta_central, output])
                collate_fn = pad_collate_note_tow_parts
        if phase == 'train':
            loaders[phase] = torch.utils.data.DataLoader(dataset, pin_memory=True, shuffle=True,
                                                         batch_size=args.batch_size, num_workers=1,
                                                         collate_fn=collate_fn)
        else:
            loaders[phase] = torch.utils.data.DataLoader(dataset, pin_memory=True, shuffle=False,
                                                         batch_size=512, num_workers=1,
                                                         collate_fn=collate_fn)
    return loaders


if __name__ == '__main__':
    args = OrderedDict()
    args.batch_size = 16
    args.segment_len = 10
    phase = 'train'
    loaders = make_loaders_and_note_dict(args, phases=['train', 'test'])
    for loader in loaders.values():
        print(len(loader))
        for batch in loader:
            for item in batch:
                glog.info(item.shape)
            break
        break

    print(1)
