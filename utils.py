from __future__ import division
import os
import re
import torch.nn as nn
import hashlib


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def latest_checkpoint(check_dir):
    if os.path.isfile(check_dir):
        return check_dir
    checkpoint_names = [f for f in os.listdir(check_dir) if '.model' in f]
    inds = [int(re.findall(r'(\d+).', i)[0]) for i in checkpoint_names]
    if len(inds) > 0:
        max_id = max(inds)
        for f in checkpoint_names:
            if str(max_id) in f:
                return os.path.join(check_dir, f)
    return None


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

