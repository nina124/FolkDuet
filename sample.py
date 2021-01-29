import glog as log
import os
import argparse
from project_util import indexed_note_to_score, get_model
from datasets import prepare_bach_note_data, get_note_meta, prepare_folk_note_data
from music21 import midi
import numpy as np
from models import *


def save_music(indexed_music, filename, ks=None, folk=False):
    score = indexed_note_to_score(indexed_music, ks, folk)
    mf = midi.translate.music21ObjectToMidiFile(score)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()


def note_sample(model, args, tag='', phase='valid', count=10, inds=None, greedy=False):
    if 'folk' in args and args.folk:
        test_self, test_partner = prepare_folk_note_data(args, phase, raw=True, segment=False, paired=True)
    else:
        test_self, test_partner = prepare_bach_note_data(args, 'valid', raw=True, segment=False)
    if inds is None:
        inds = np.linspace(0, len(test_self)-1, count).astype(np.int)
    if not os.path.exists(os.path.join(args.exp_dir, 'samples')):
        os.makedirs(os.path.join(args.exp_dir, 'samples'))
    for i in inds:
        music = note_sample_(model, args, test_self[i], test_partner[i], greedy=greedy)
        save_music(music, os.path.join(args.exp_dir, 'samples', '%s-from%d.mid' % (tag, i)), folk=args.folk)
    log.info('%d sample music written in %s' % (len(inds), os.path.join(args.exp_dir, 'samples')))


def note_sample_(model, args, self, partner, index2duration=None, greedy=False):
    if index2duration is None:
        _, _, index2duration, _ = get_note_meta('folk' in args and args.folk)
    self = torch.from_numpy(self[:, :2]).cuda().long().unsqueeze(0)
    pred_ind = args.init_len
    pred = self[:, :pred_ind]
    pred_offset = sum([index2duration[i.item()] for i in self[0, :pred_ind, 1]])
    partner_rev = torch.from_numpy(partner[::-1].copy()).cuda().long().unsqueeze(0)
    partner = torch.from_numpy(partner[:, :2]).cuda().long().unsqueeze(0)
    partner_offsets = [index2duration[i.item()] for i in partner[0, :, 1]]
    partner_offsets = np.concatenate([[0], np.cumsum(partner_offsets)])
    total_length = partner_offsets[-1]
    meta = torch.arange(16).view(1, 16).cuda()
    while pred_offset < total_length:
        partner_ind = sum(partner_offsets < pred_offset)
        current_partner = partner[:, :(partner_ind - 1)]
        partner_central = partner[:, partner_ind - 1, 0]
        meta_central = meta[:, int(4 * pred_offset) % 16]
        if args.arch == 'BachHM':
            logit, _ = model([pred, None, current_partner, None, None, partner[:, partner_ind-1],
                              meta_central, partner_rev[:, :-partner_ind], None, None])
        elif args.arch == 'Generator':
            logit, _ = model([pred, None, current_partner, None, None, partner_central, meta_central])
        elif args.arch == 'BachM':
            logit, _ = model([pred, None, meta_central])
        elif args.arch == 'StyleRewarder':
            logit, _ = model([pred, None, meta_central])
        else:
            raise NotImplementedError
        if greedy:
            total_pred = logit.argmax(1, keepdim=True)
        else:
            logit = F.softmax(logit, 1)
            total_pred = logit.multinomial(num_samples=1).data
        pitch = total_pred // args.num_durations
        duration = total_pred % args.num_durations
        pred = torch.cat([pred, torch.cat([pitch, duration], -1).unsqueeze(1)], 1)
        pred_offset += index2duration[duration.item()]
        pred_ind += 1
        if pred_ind > 10000:
            break
    self = pred.data.cpu().numpy()
    partner = partner.data.cpu().numpy()
    pitches = [partner[0, :, 0], self[0, :, 0]]
    durations = [partner[0, :, 1], self[0, :, 1]]
    music = (pitches, durations)
    return music


def main():
    model, model_arg = get_model(os.path.join(main_args.check_dir))
    model_arg.folk = True
    test_self, test_partner = prepare_folk_note_data(model_arg, 'test', raw=True, segment=False, paired=True)
    _, pitch2index, index2duration, _ = get_note_meta(True)
    inds = np.random.choice(len(test_self), main_args.num)
    if not os.path.exists(main_args.save_dir):
        os.makedirs(main_args.save_dir)
    for i in inds:
        human_part = test_partner[i].copy()
        machine_part = test_self[i].copy()
        music = note_sample_(model, model_arg, machine_part, human_part, index2duration)
        filename = os.path.join(main_args.save_dir, 'sample-id%d.mid' % i)
        save_music(music, filename, folk=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample music')
    parser.add_argument('--check_dir', default='results/pretrained')
    parser.add_argument('--save_dir', default='generated_samples/')
    parser.add_argument('--num', default=10)
    return parser.parse_args()


if __name__ == '__main__':
    main_args = parse_args()
    main()


