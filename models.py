import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqBaseModel(nn.Module):
    def __init__(self, args):
        super(SeqBaseModel, self).__init__()
        self.num_tokens = len(args.num_tokens)
        self.encoders = nn.ModuleList([nn.Embedding(num_tokens, n_emb) for n_emb, num_tokens in
                                       zip(args.nembs, args.num_tokens)])
        self.rnn = nn.GRU(sum(args.nembs), args.nhid, args.nlayers, dropout=args.dropout, batch_first=True)

    def seq(self, part, part_length, recover_idx=None, bs=1):
        if len(part) == 0:
            return torch.zeros((bs, self.args.nhid)).cuda()
        part = torch.cat([self.encoders[i](part[:, :, i]) for i in range(self.num_tokens)], -1)
        if len(part) > 1:
            part = torch.nn.utils.rnn.pack_padded_sequence(part, part_length, batch_first=True)
            part, _ = self.rnn(part)
            part, _ = torch.nn.utils.rnn.pad_packed_sequence(part, batch_first=True)
            part_length = part_length.view(len(part), 1, 1).expand(-1, 1, part.shape[-1])
            part = F.dropout(torch.gather(part, 1, part_length - 1), p=self.args.dropout).view(len(part), -1)
            if bs > len(part):
                part = torch.cat([part, torch.zeros((bs - len(part), part.shape[-1])).cuda()])
            if recover_idx is not None:
                part = part[recover_idx]
        else:
            part, _ = self.rnn(part)
            part = F.dropout(part[:, -1, :], p=self.args.dropout).view(len(part), -1)
            if bs > len(part):
                part = torch.cat([part, torch.zeros((bs - len(part), part.shape[-1])).cuda()])
        return part


class Generator(SeqBaseModel):
    def __init__(self, args):
        super(Generator, self).__init__(args)
        self.meta_encoder = nn.Embedding(16, args.nemb_meta)
        self.central_meta_fc = nn.Linear(args.nemb_meta, args.meta_nfc_cent)
        self.central_fc = nn.Linear(args.nembs[0], args.nfc_cent)
        self.left_fc = nn.Linear(2 * args.nhid, args.nfc_left)
        self.fc = nn.Linear(args.nfc_left + args.nfc_cent + args.meta_nfc_cent, args.pred_nfc)
        self.pred = nn.Linear(args.pred_nfc, args.num_pitches * args.num_durations)
        self.value = nn.Linear(args.pred_nfc, 1)
        self.args = args

    def forward(self, inputs):
        self_left, self_length, partner_left, partner_length, recover_idx, partner_central, meta_central = inputs
        self_left = self.seq(self_left, self_length)
        partner_left = self.seq(partner_left, partner_length, recover_idx, bs=len(self_left))
        left_feature = self.left_fc(torch.cat([self_left, partner_left], -1))
        meta_central_emb = self.meta_encoder(meta_central)
        meta_central_feature = self.central_meta_fc(F.dropout(meta_central_emb, p=self.args.dropout))
        central_emb = self.encoders[0](partner_central)
        central_feature = self.central_fc(F.dropout(central_emb, p=self.args.dropout))
        feature = torch.cat((left_feature, central_feature, meta_central_feature), dim=-1)
        last_feature = F.relu(self.fc(feature))
        pred = self.pred(F.dropout(last_feature, p=self.args.dropout))
        value = self.value(F.dropout(last_feature, p=self.args.dropout))
        return pred, value


class StyleRewarder(SeqBaseModel):
    def __init__(self, args):
        super(StyleRewarder, self).__init__(args)
        self.left_fc = nn.Linear(args.nhid, args.nfc_left)
        self.fc = nn.Linear(args.nfc_left, args.pred_nfc)
        self.pred = nn.Linear(args.pred_nfc, args.num_pitches * args.num_durations)
        self.args = args

    def forward(self, inputs):
        self_left, self_length, _ = inputs
        self_left = self.seq(self_left, self_length)
        feature = self.left_fc(self_left)
        pred = self.pred(F.relu(self.fc(feature)))
        return pred, None

    def reward(self, self_part, return_seq=False):
        self_emb = torch.cat([self.encoders[i](self_part[:, :, i]) for i in range(self.num_tokens)], -1)
        self_part, _ = self.rnn(self_emb)
        if not return_seq:
            self_part = self_part[:, -1, :].view(len(self_part), -1)
        feature = self.left_fc(self_part)
        pred = self.pred(F.relu(self.fc(feature)))
        return pred


class BachM(SeqBaseModel):
    def __init__(self, args):
        super(BachM, self).__init__(args)
        self.meta_encoder = nn.Embedding(16, args.nemb_meta)
        self.central_meta_fc = nn.Linear(args.nemb_meta, args.meta_nfc_cent)
        self.left_fc = nn.Linear(args.nhid, args.nfc_left)
        self.fc = nn.Linear(args.nfc_left + args.meta_nfc_cent, args.pred_nfc)
        self.pred = nn.Linear(args.pred_nfc, args.num_pitches * args.num_durations)
        self.args = args

    def forward(self, inputs):
        self_left, self_length, meta_central = inputs
        self_left = self.seq(self_left, self_length)
        left_feature = self.left_fc(self_left)
        meta_central_emb = self.meta_encoder(meta_central)
        meta_central_feature = self.central_meta_fc(F.dropout(meta_central_emb, p=self.args.dropout))
        feature = torch.cat((left_feature, meta_central_feature), dim=-1)
        pred = self.pred(F.dropout(F.relu(self.fc(feature)), p=self.args.dropout))
        return pred, None


class BachHM(SeqBaseModel):
    def __init__(self, args):
        super(BachHM, self).__init__(args)
        self.meta_encoder = nn.Embedding(16, args.nemb_meta)
        self.central_meta_fc = nn.Linear(args.nemb_meta, args.meta_nfc_cent)
        self.central_fc = nn.Linear(sum(args.nembs), args.nfc_cent)
        self.rnn_fc = nn.Linear(3 * args.nhid, args.nfc_left)
        self.fc = nn.Linear(args.nfc_left + args.nfc_cent + args.meta_nfc_cent, args.pred_nfc)
        self.pred = nn.Linear(args.pred_nfc, args.num_pitches * args.num_durations)
        self.args = args

    def forward(self, inputs):
        self_left, self_length, partner_left, p_left_length, p_left_recover_idx, \
        partner_central, meta_central, partner_right, p_right_length, p_right_recover_idx = inputs
        partner_left = self.seq(partner_left, p_left_length, p_left_recover_idx, bs=len(self_left))
        partner_right = self.seq(partner_right, p_right_length, p_right_recover_idx, bs=len(self_left))
        self_left = self.seq(self_left, self_length)
        rnn_feature = torch.cat([self_left, partner_left, partner_right], -1)
        rnn_feature = self.rnn_fc(rnn_feature)

        central_emb = torch.cat([self.encoders[i](partner_central[..., i]) for i in range(self.num_tokens)], -1)
        central_feature = self.central_fc(F.dropout(central_emb, p=self.args.dropout))

        meta_central_emb = self.meta_encoder(meta_central)
        meta_central_feature = self.central_meta_fc(F.dropout(meta_central_emb, p=self.args.dropout))
        feature = torch.cat((rnn_feature, central_feature, meta_central_feature), dim=-1)
        pred = self.pred(F.dropout(F.relu(self.fc(feature)), p=self.args.dropout))
        return pred, None
