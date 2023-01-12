import argparse
import gc
import logging
import sys
import typing
from math import sqrt
from os import PathLike

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYVX'

log = logging.getLogger('casp_eval')

def setup_logging(save_path: typing.Optional[PathLike] = None,
                  log_level: typing.Union[str, int] = 'info',
                  formatter: typing.Optional[logging.Formatter] = None) -> None:
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper())
    else:
        level = log_level

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if formatter is None:
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s.%(msecs)03d [%(process)d:%(threadName)s] <%(name)s> - %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S")

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    if save_path is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        file_handler = logging.FileHandler(save_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def setup_args(_args=None):
    argparser = argparse.ArgumentParser("Evaluation on CASP Targets")
    argparser.add_argument("--templates", required=True,
                           help="Path to fasta file with template sequences")
    argparser.add_argument("--query", required=True, help="Path to fasta file with query sequence(s)")
    argparser.add_argument("--model_weights", required=True, help="Path to file with model checkpoint")
    argparser.add_argument("--device", default='cpu', help="CUDA device, if available")
    argparser.add_argument("--minibatch_size", default=128, type=int, help="Minibatch size, depends on available memory")
    argparser.add_argument("--log_file")
    argparser.add_argument("--log_level", default='info')
    argparser.add_argument("--out_file", help="Path for CSV output; stdout if not specified")
    return argparser.parse_args(_args)


def make_mask(seq1: torch.Tensor, seq2: torch.Tensor, _mask: torch.Tensor) -> torch.Tensor:
    mask1 = (seq1 == 0.0).all(dim=-1)
    mask2 = (seq2 == 0.0).all(dim=-1)
    mask = mask1.unsqueeze(dim=2) | mask2.unsqueeze(dim=1)
    mask = ~mask
    shape = (min(_mask.shape[0], mask.shape[1]), min(_mask.shape[1], mask.shape[2]))
    mask[...] = _mask[0, -1]
    mask[:, :shape[0], :shape[1]] = _mask[:shape[0], :shape[1]]
    return mask


def soft_align(z1: torch.Tensor, z2: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    _z1_z2 = z1.matmul(z2.transpose(1, 2)) / sqrt(z1.shape[2])
    z1_z2 = torch.zeros_like(_z1_z2) - 1_000
    z1_z2[mask] = _z1_z2[mask]
    alpha = F.softmax(z1_z2, dim=1)
    beta = F.softmax(z1_z2, dim=2)
    _a = alpha + beta - alpha * beta
    a = torch.zeros_like(_a)
    a[mask] = _a[mask]
    score = (a * z1_z2).sum(dim=(1, 2)) / a.sum(dim=(1, 2))
    return score

class OrdinalEncodingForEmbedding(nn.Module):
    def __init__(self):
        super(OrdinalEncodingForEmbedding, self).__init__()
        self.register_buffer('dev', torch.zeros(1))
        self.amino_acid_ordinal = {a: i+1 for i, a in enumerate(AMINO_ACIDS)} # zero is padding_idx in nn.Embedding

    def __call__(self, seq):
        return torch.tensor([self.amino_acid_ordinal[a] for a in seq], dtype=torch.int, device=self.dev.device)


class EmbeddingLstmModel(nn.Module):

    def __init__(self, embedding_size: int, d_input: int, d_lstm_out: int, num_lstm_layers: int, d_dense_out: int,
                 n_cat: int,
                 clip: float = 0.0, pos_weight=None, dropout=0.0, init_bias=0.0):
        super().__init__()
        self.encoding = OrdinalEncodingForEmbedding()
        self.embedding = nn.Embedding(22, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(d_input, d_lstm_out, bidirectional=True, num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout)
        self.dense = nn.Linear(2 * d_lstm_out, d_dense_out, bias=False)
        self.sigmoid_scale = nn.parameter.Parameter(torch.empty(1, n_cat))
        nn.init.xavier_uniform_(self.sigmoid_scale)
        self.skip_weight = nn.parameter.Parameter(torch.tensor(0.5))
        self.mask = nn.parameter.Parameter(torch.ones(100, 100, dtype=torch.bool), requires_grad=False)
        if init_bias != 0.0:
            self.sigmoid_bias = nn.parameter.Parameter(torch.tensor(init_bias).expand(1, n_cat))
        else:
            self.sigmoid_bias = nn.parameter.Parameter(torch.empty(1, n_cat))
            nn.init.xavier_uniform_(self.sigmoid_bias)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.bce_logits_loss = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
        if clip != 0.0:
            for p in self.parameters():
                if p.requires_grad:
                    p.register_hook(lambda g: torch.clamp(g, -clip, clip))

    def forward(self, mb) -> torch.Tensor:
        self.lstm.flatten_parameters()
        enc1 = self.context_sensitive_encoding(mb['seq1'])
        enc2 = self.context_sensitive_encoding(mb['seq2'])
        mask = make_mask(enc1, enc2, self.mask)
        algn_score = soft_align(enc1, enc2, mask).unsqueeze(1)
        logits = algn_score * self.sigmoid_scale.abs() + self.sigmoid_bias
        return logits  # use BCEWithLogitsLoss

    def context_sensitive_encoding(self, seqs):
        seqs1 = [self.encoding(s) for s in seqs]
        seq1 = pad_sequence(seqs1, batch_first=True)
        size1 = torch.tensor([t.shape[0] for t in seqs1], dtype=torch.int)
        emb1 = self.embedding(seq1)
        packed_seq1 = pack_padded_sequence(emb1, size1, batch_first=True, enforce_sorted=False)
        enc1, _ = self.lstm(packed_seq1)
        padded1, _ = pad_packed_sequence(enc1, batch_first=True)
        enc1 = self.dense(padded1)
        enc1 = self.skip_weight * enc1 + (1 - self.skip_weight) * emb1
        return enc1

    def score_pair(self, mb):
        logits = self(mb)
        return self.loss(logits, mb['labels'])

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bce_logits_loss(logits, labels) / logits.shape[0]

    @staticmethod
    def prob(logits: torch.Tensor):
        return torch.sigmoid(logits)

    @torch.no_grad()
    def query_templates(self, query, templates):
        mb = {'seq1': [query] * len(templates), 'seq2': templates}
        scores = self(mb)
        return scores.cpu().numpy()

    @torch.no_grad()
    def query_templates(self, query, templates):
        mb = {'seq1': [query] * len(templates), 'seq2': templates}
        scores = self(mb)
        return scores.cpu().numpy()


def init_model(args: argparse.Namespace):
    model = EmbeddingLstmModel(embedding_size=64,
                               d_input=64,
                               d_lstm_out=64,
                               num_lstm_layers=3,
                               d_dense_out=64,
                               n_cat=1)
    checkpoint = torch.load(args.model_weights, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)
    return model.to(args.device)

@torch.no_grad()
def eval_single_query(model, query, templates, mb_size):
    scores_tmp = []
    for i in range(0, len(templates), mb_size):
        log.debug(f"Minibatch {i}:{i + mb_size} out of {len(templates)}")
        scores = model.query_templates(query, templates[i:i + mb_size])
        scores_tmp.append(scores.reshape(-1))
        gc.collect()
    # global scores
    scores = np.concatenate(scores_tmp)
    return scores.argmax()


def main(args):
    model = init_model(args)
    queries = list(SeqIO.parse(args.query, 'fasta'))
    template_records = list(SeqIO.parse(args.templates, 'fasta'))
    templates = [sr.seq for sr in template_records]
    folds = []
    for q in queries:
        log.debug(q.id)
        tx = eval_single_query(model, q.seq, templates, args.minibatch_size)
        folds.append(template_records[tx].description.split(' ')[1])
    out_df = pd.DataFrame({'query': [q.id for q in queries], 'fold': folds})
    if args.out_file:
        out_df.to_csv(args.out_file, index=False)
    else:
        out_df.to_csv(sys.stdout, index=False)

if __name__ == '__main__':
    args=setup_args()
    log.setLevel(args.log_level)
    setup_logging(args.log_file, args.log_level)
    main(args)
