from music21 import *
from datasets import get_note_meta
from utils import md5
import numpy as np
import pickle
import glog
import os


def full_partner_segment_note_(self_feature, partner_feature, init_len=10):
    _, _, index2duration, _ = get_note_meta(folk=False)
    self_duration = self_feature[:, 1]
    self_duration = [index2duration[i] for i in self_duration]
    self_offsets = [0] + list(np.cumsum(self_duration[:-1]))
    partner_duration = partner_feature[:, 1]
    partner_duration = [index2duration[i] for i in partner_duration]
    partner_offsets = [0] + list(np.cumsum(partner_duration[:-1]))
    s_list, pl_list, pc_list, mc_list, pr_list, o_list = [], [], [], [], [], []

    pred_ind = init_len
    while pred_ind < len(self_feature):
        s_list.append(self_feature[:pred_ind])
        o_list.append(self_feature[1:pred_ind + 1])
        partner_ind = sum(partner_offsets < self_offsets[pred_ind])
        partner = partner_feature[:partner_ind]
        pl_list.append(partner[:-1])
        pc_list.append(partner[-1])
        pr_list.append(partner_feature[:partner_ind - 1 - len(partner_feature):-1])
        mc_list.append((4 * self_offsets[pred_ind]) % 16)
        pred_ind += 1
    return s_list, pl_list, pc_list, mc_list, pr_list, o_list


def full_partner_segment_note(args, self_data, partner_data):
    glog.info('Segmenting data')
    init_len = args.init_len
    self_left, partner_left, partner_central, meta_central, partner_right, output = [], [], [], [], [], []
    for self_f, partner_f in zip(self_data, partner_data):
        s_list, pl_list, pc_list, mc_list, pr_list, o_list = full_partner_segment_note_(
            self_f, partner_f, init_len)
        self_left += s_list
        partner_left += pl_list
        partner_central += pc_list
        meta_central += mc_list
        partner_right += pr_list
        output += o_list
    return self_left, partner_left, partner_central, meta_central, partner_right, output


def segment_note_(self_feature, partner_feature, seed_len=10):
    _, _, index2duration, _ = get_note_meta(folk=False)
    self_duration = self_feature[:, 1]
    self_duration = [index2duration[i] for i in self_duration]
    self_offsets = [0] + list(np.cumsum(self_duration[:-1]))
    partner_duration = partner_feature[:, 1]
    partner_duration = [index2duration[i] for i in partner_duration]
    partner_offsets = [0] + list(np.cumsum(partner_duration[:-1]))
    s_list, p_list, pc_list, mc_list, o_list = [], [], [], [], []

    pred_ind = seed_len
    while pred_ind < len(self_feature):
        s_list.append(self_feature[:pred_ind])
        o_list.append(self_feature[1:pred_ind + 1])
        partner_ind = sum(partner_offsets < self_offsets[pred_ind])
        partner = partner_feature[:partner_ind]
        p_list.append(partner[:-1])
        pc_list.append(partner[-1, 0])
        mc_list.append((4 * self_offsets[pred_ind]) % 16)
        pred_ind += 1
    return s_list, p_list, pc_list, mc_list, o_list


def segment_note(args, self_data, partner_data):
    glog.info('Segmenting data')
    init_len = args.init_len
    self_left, partner_left, partner_central, meta_central, output = [], [], [], [], []
    for self_f, partner_f in zip(self_data, partner_data):
        s_list, p_list, pc_list, mc_list, o_list = segment_note_(
            self_f, partner_f, init_len)
        self_left += s_list
        partner_left += p_list
        partner_central += pc_list
        meta_central += mc_list
        output += o_list
    return self_left, partner_left, partner_central, meta_central, output


def prepare_bach_note_data(args, phase, raw=False, segment=True):
    pitches, durations, metas, pitches_raw, durations_raw, metas_raw = initialization_note(dataset_path=None)
    if phase == 'train':
        start = 0
        end = int(len(pitches) * 0.95)
    elif phase == 'valid':
        start = int(len(pitches) * 0.95)
        end = len(pitches)
    elif phase == 'all':
        start = 0
        end = len(pitches)
    else:
        assert False, 'phase undefined'
    if raw:
        pitches_raw = pitches_raw[start:end]
        durations_raw = durations_raw[start:end]
        data = [[p, d] for p, d in zip(pitches_raw, durations_raw)]
    else:
        pitches = sum(pitches[start:end], [])
        durations = sum(durations[start:end], [])
        data = [[p, d] for p, d in zip(pitches, durations)]
    if segment:
        head = 'bach'
        if not raw:
            head += '_trans'
        if args.arch == 'BachHM':
            head += '_full_partner'
        segment_data_path = '%s_%s_seq.pkl' % (head, phase)
        segment_data_path = os.path.join('data', segment_data_path)
        if not os.path.exists(segment_data_path):
            self_data, partner_data = pair_note(data)
            if args.arch == 'BachHM':
                self_left, partner_left, partner_central, meta_central, partner_right, outputs = \
                    full_partner_segment_note(args, self_data, partner_data)
                glog.info('pickle.dumping {}'.format(segment_data_path))
                pickle.dump([self_left, partner_left, partner_central, meta_central, partner_right, outputs],
                            open(segment_data_path, 'wb'))
            else:
                self_left, partner_left, partner_central, meta_central, output = \
                    segment_note(args, self_data, partner_data)
                glog.info('pickle.dumping {}'.format(segment_data_path))
                pickle.dump([self_left, partner_left, partner_central, meta_central, output],
                            open(segment_data_path, 'wb'))
        else:
            with open(segment_data_path, 'rb') as f:
                if args.arch == 'BachHM':
                    self_left, partner_left, partner_central, meta_central, partner_right, outputs = pickle.load(f)
                else:
                    self_left, partner_left, partner_central, meta_central, output = pickle.load(f)
            glog.info('segment dataset load from %s' % segment_data_path)
        glog.info('md5 of %s: %s' % (segment_data_path, md5(segment_data_path)))
        if args.arch == 'BachHM':
            return self_left, partner_left, partner_central, meta_central, partner_right, outputs
        else:
            return self_left, partner_left, partner_central, meta_central, output
    else:
        self_data, partner_data = pair_note(data)
        self_data = [item.astype(np.int) for item in self_data]
        partner_data = [item.astype(np.int) for item in partner_data]
        return self_data, partner_data


def pair_note(chorales, ks=None):
    glog.info('Pairing data')
    self_data = []
    partner_data = []
    ks_paired = []
    for ind, (pitch, duration) in enumerate(chorales):
        for i in range(len(pitch)):
            for j in range(len(pitch)):
                if i == j:
                    continue
                self_data.append(
                    np.concatenate([pitch[i].reshape(-1, 1), duration[i].reshape(-1, 1)], -1).astype(np.uint8))
                partner_data.append(
                    np.concatenate([pitch[j].reshape(-1, 1), duration[j].reshape(-1, 1)], -1).astype(np.uint8))
                if ks is not None:
                    ks_paired.append(ks[ind])
    if ks is None:
        return self_data, partner_data
    else:
        return self_data, partner_data, ks_paired


def initialization_note(dataset_path=None):
    if dataset_path is None:
        dataset_path = 'data/bach_note_folked.pickle'
    with open(dataset_path, 'rb') as f:
        pitches, durations, metas, pitches_raw, durations_raw, metas_raw = pickle.load(f)
    glog.info('dataset load from %s (%s)' % (dataset_path, md5(dataset_path)))
    return pitches, durations, metas, pitches_raw, durations_raw, metas_raw

