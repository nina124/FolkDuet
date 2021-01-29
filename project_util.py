from argparse import Namespace
import json
import os
from utils import latest_checkpoint
from datasets import get_note_meta
import glog
import sys
from models import *
from music21 import note, pitch, stream, duration


def get_model(check_dir, load_weight=True):
    if os.path.isfile(check_dir):
        config_path = os.path.join(os.path.join(os.path.dirname(check_dir), 'config.json'))
    else:
        config_path = os.path.join(check_dir, 'config.json')
    args = Namespace(**json.load(open(config_path, 'r')))
    i2p, p2i, i2d, _ = get_note_meta(args.folk)
    args.num_pitches = len(i2p)
    args.num_durations = len(i2d)
    args.num_tokens = [args.num_pitches, args.num_durations]
    args.index2pitch = i2p
    args.pitch2index = p2i
    model = eval('%s(args)' % args.arch)
    model_name = latest_checkpoint(check_dir)
    if load_weight:
        state_dict = torch.load(model_name)
        model.load_state_dict(state_dict)
        glog.info('load model weight from %s' % model_name)
    else:
        glog.info('load model config from %s' % config_path)
    model = model.cuda()
    model.eval()
    return model, args


def exp_preparation(args):
    if args.check_dir:
        if os.path.isfile(args.check_dir):
            config_path = os.path.join(os.path.join(os.path.dirname(args.check_dir), 'config.json'))
        else:
            config_path = os.path.join(args.check_dir, 'config.json')
        args_old = json.load(open(config_path))
        args = vars(args)
        # assert args['arch'] == args_old['arch']
        for key in ['nhid', 'nemb_pitch', 'nemb_duration', 'nemb_meta', 'nlayers',
                    'meta_nfc_cent', 'nfc_cent', 'nfc_left', 'pred_nfc']:
            if key in args_old:
                args[key] = args_old[key]
        args = Namespace(**args)
    args.nembs = [args.nemb_pitch, args.nemb_duration]
    i2p, p2i, i2d, _ = get_note_meta(args.folk)
    args.num_pitches = len(i2p)
    args.num_durations = len(i2d)
    args.num_tokens = [args.num_pitches, args.num_durations]
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    for folder in ['files', ]:
        if not os.path.exists(os.path.join(args.exp_dir, folder)):
            os.makedirs(os.path.join(args.exp_dir, folder))
    os.system('cp -r *py datasets %s' % os.path.join(args.exp_dir, 'files'))
    glog.info('Command line is: {}'.format(' '.join(sys.argv)))

    glog.info('Called with args:')
    items = [i for i in dir(args) if not i.startswith('_')]
    for i in items:
        glog.info('%s: %s' % (i, str(eval('args.' + i))))

    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(args, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    return args


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    return note.Note(pitch.Pitch(int(note_or_rest_string)))


def indexed_note_to_score(music, ks=None, folk=False):
    pitches, durations = music
    index2pitch, _, index2duration, _ = get_note_meta(folk=folk)
    score = stream.Score()
    max_duration = max([sum([index2duration[i] for i in dur]) for dur in durations])
    for voice_index, (pitch, dur) in enumerate(zip(pitches, durations)):
        total_duration = 0
        part = stream.Part(id='part' + str(voice_index))
        if ks is not None:
            part.append(ks)
        for p, d in zip(pitch, dur):
            f = standard_note(index2pitch[p])
            f.duration = duration.Duration(index2duration[d])
            total_duration += index2duration[d]
            part.append(f)
        if total_duration < max_duration:
            n = note.Rest()
            n.duration = duration.Duration(max_duration - total_duration)
            part.append(n)
        score.insert(part)
    return score

