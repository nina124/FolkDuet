import numpy as np
import pickle
import os
import glog
from utils import md5


def segment_note_(self_feature, meta, seed_len=10):
    _, _, index2duration, _ = get_note_meta(folk=True)
    s_list, mc_list, o_list = [], [], []
    pred_ind = seed_len
    while pred_ind < len(self_feature):
        s_list.append(self_feature[:pred_ind])
        mc_list.append(meta[1:pred_ind+1])
        o_list.append(self_feature[1:pred_ind+1])
        pred_ind += 1
    return s_list, mc_list, o_list


def segment_note(args, self_data, metas):
    glog.info('Segmenting data')
    init_len = args.init_len
    self_left, meta_central, output = [], [], []
    for self_f, meta in zip(self_data, metas):
        s_list, mc_list, o_list = segment_note_(self_f, meta, init_len)
        self_left += s_list
        meta_central += mc_list
        output += o_list
    return self_left, meta_central, output


def segment_two_note_(self_feature, partner_feature, meta, seed_len=10):
    with open('data/folk_bach_note.info', 'rb') as f:
        _, _, index2duration, _ = pickle.load(f)
    self_duration = self_feature[:, 1]
    self_duration = [index2duration[i] for i in self_duration]
    self_offsets = [0] + list(np.cumsum(self_duration[:-1]))
    partner_duration = partner_feature[:, 1]
    partner_duration = [index2duration[i] for i in partner_duration]
    partner_offsets = np.array([0] + list(np.cumsum(partner_duration[:-1])))
    s_list, p_list, pc_list, mc_list, o_list = [], [], [], [], []

    pred_ind = seed_len
    while pred_ind < len(self_feature):
        s_list.append(self_feature[:pred_ind])
        o_list.append(self_feature[1:pred_ind+1])
        partner_ind = sum(partner_offsets < self_offsets[pred_ind])
        partner = partner_feature[:partner_ind]
        p_list.append(partner[:-1])
        pc_list.append(partner[-1, 0])
        mc_list.append(meta[pred_ind])
        pred_ind += 1
    return s_list, p_list, pc_list, mc_list, o_list


def segment_two_note(self_data, partner_data, metas):
    glog.info('Segmenting data')
    init_len = 10
    self_left, partner_left, partner_central, meta_central, output = [], [], [], [], []
    for ind, (self_f, partner_f, meta) in enumerate(zip(self_data, partner_data, metas)):
        s_list, p_list, pc_list, mc_list, o_list = segment_two_note_(self_f, partner_f, meta, init_len)
        self_left += s_list
        partner_left += p_list
        partner_central += pc_list
        meta_central += mc_list
        output += o_list
    return self_left, partner_left, partner_central, meta_central, output


def random_duet(pitches_raw, durations_raw, metas_raw, augment=10):
    with open('data/folk_note.info', 'rb') as f:
        i2p, p2i, i2d, d2i = pickle.load(f)
    rest_ind = p2i['rest']
    np.random.seed(1024)
    self = []
    partner = []
    metas = []
    for ind in range(len(pitches_raw)):
        pitch = pitches_raw[ind]
        duration = durations_raw[ind]
        meta = metas_raw[ind]
        n = 0
        bass_pitch = pitch.copy()
        while n < augment:
            pitch_ = pitch.copy()
            bass_pitch_ = bass_pitch.copy()
            min_shift = - bass_pitch_.min()
            max_shift = rest_ind - pitch_[pitch_ != rest_ind].max()
            possible_gaps = list(range(12, min(24, max_shift - min_shift)))
            gap = np.random.choice(possible_gaps)
            if gap > 0:
                pitch_[pitch_ != rest_ind] += gap
            else:
                bass_pitch_[bass_pitch_ != rest_ind] += gap

            min_shift = -bass_pitch_.min()
            max_shift = rest_ind - pitch_[pitch_ != rest_ind].max()
            if min_shift < max_shift:
                shift = np.random.randint(min_shift, max_shift)
                pitch_[pitch_ != rest_ind] += shift
                bass_pitch_[bass_pitch_ != rest_ind] += shift
            assert pitch_.max() <= 49
            assert bass_pitch_.min() >= 0
            if np.random.random() < 0.5:
                self.append(np.stack([pitch_, duration], -1))
                partner.append(np.stack([bass_pitch_, duration], -1))
            else:
                partner.append(np.stack([pitch_, duration], -1))
                self.append(np.stack([bass_pitch_, duration], -1))
            metas.append(meta)
            n += 1
    return self, partner, metas


def prepare_folk_note_data(args, phase, raw=True, segment=True, paired=False):
    pitches, durations, metas, pitches_raw, durations_raw, metas_raw = initialization_note(dataset_path=None)
    if phase == 'train':
        start = 0
        end = int(len(pitches) * 0.8)
    elif phase == 'valid':
        start = int(len(pitches) * 0.8)
        end = int(len(pitches) * 0.9)
    elif phase == 'test':
        start = int(len(pitches) * 0.9)
        end = len(pitches)
    else:
        assert False, 'phase undefined'
    if raw:
        pitches_raw = pitches_raw[start:end]
        durations_raw = durations_raw[start:end]
        metas_raw = metas_raw[start:end]
        metas = metas_raw
        data = [np.stack([p, d], -1) for p, d in zip(pitches_raw, durations_raw)]
    else:
        pitches = sum(pitches[start:end], [])
        durations = sum(durations[start:end], [])
        metas = sum(metas[start:end], [])
        data = [np.stack([p, d], -1) for p, d in zip(pitches, durations)]
    if segment:
        if args.arch == 'StyleRewarder':
            head = 'folk'
            if not raw:
                head += '_trans'
            segment_data_path = 'data/%s_%s_seq.pkl' % (head, phase)
            if not os.path.exists(segment_data_path):
                self_left, meta_central, output = segment_note(args, data, metas)
                with open(segment_data_path, 'wb') as f:
                    pickle.dump([self_left, meta_central, output], f)
                glog.info('pickle.dumping {}'.format(segment_data_path))
            else:
                with open(segment_data_path, 'rb') as f:
                    self_left, meta_central, output = pickle.load(f)
                glog.info('segment dataset load from %s' % segment_data_path)
            glog.info('md5 of %s: %s' % (segment_data_path,  md5(segment_data_path)))
            return self_left, meta_central, output
        else:
            segment_data_path = 'data/folk_randomtrans_%s_seq.pkl' % phase
            if not os.path.exists(segment_data_path):
                self_data, partner_data, metas = random_duet(pitches_raw, durations_raw, metas,
                                                             augment=20 if phase == 'train' else 10)
                self_left, partner_left, partner_central, meta_central, output = \
                    segment_two_note(self_data, partner_data, metas)
                with open(segment_data_path, 'wb') as f:
                    pickle.dump([self_left, partner_left, partner_central, meta_central, output], f)
                glog.info('pickle.dumping {}'.format(segment_data_path))
            else:
                with open(segment_data_path, 'rb') as f:
                    self_left, partner_left, partner_central, meta_central, output = pickle.load(f)
                glog.info('segment dataset load from %s' % segment_data_path)
            glog.info('md5 of %s: %s' % (segment_data_path, md5(segment_data_path)))
            return self_left, partner_left, partner_central, meta_central, output
    else:
        if paired:
            index2pitch, pitch2index, index2duration, duration2index = get_note_meta(True)
            rest_ind = pitch2index['rest']
            gap = 12
            test_self = []
            test_partner = []
            for i in range(len(data)):
                test_self.append(data[i])
                partner = data[i].copy()
                pitch = partner[:, 0]
                duration = partner[:, 1]
                min_shift = -pitch.min()
                max_shift = rest_ind - pitch[pitch != rest_ind].max()
                if min_shift <= -gap and max_shift > gap:
                    if min_shift + max_shift > 0:
                        shift = gap
                    else:
                        shift = -gap
                elif min_shift <= -gap:
                    shift = -gap
                else:
                    shift = gap
                partner_pitch = pitch.copy()
                partner_pitch[pitch != rest_ind] += shift
                partner = np.stack([partner_pitch, duration], -1)
                test_partner.append(partner)
            return test_self, test_partner

        return data, data


def initialization_note(dataset_path=None):
    if dataset_path is None:
        dataset_path = 'data/folk_note.pickle'
    with open(dataset_path, 'rb') as f:
        pitches, durations, metas, pitches_raw, durations_raw, metas_raw = pickle.load(f)
    glog.info('dataset load from %s (%s)' % (dataset_path, md5(dataset_path)))
    return pitches, durations, metas, pitches_raw, durations_raw, metas_raw


def get_note_meta(folk):
    if folk:
        with open('data/folk_note.info', 'rb') as f:
            i2p, p2i, i2d, d2i = pickle.load(f)
    else:
        with open('data/folk_bach_note.info', 'rb') as f:
            i2p, p2i, i2d, d2i = pickle.load(f)
    return i2p, p2i, i2d, d2i


