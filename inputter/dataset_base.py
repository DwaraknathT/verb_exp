# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab

from itertools import chain
import torchtext
from onmt.inputters.inputter import get_fields


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example

def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    return GATDataset.get_fields_gat("gat", n_src_features, n_tgt_features)

def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


class DatasetBase(torchtext.data.Dataset):
    """[summary]
    
    Arguments:
        torchtext {[type]} -- [description]
    """
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(DatsetBase, self).__reduce_ex__()

    def load_fields(self, vocab_dict):
        """ Load fields from vocab.pt, and set the `fields` attribute.

        Args:
            vocab_dict (dict): a dict of loaded vocab from vocab.pt file.
        """

        fields = load_fields_from_vocab(vocab_dict.items(), self.data_type)
        self.fields = dict([(k, f) for (k, f) in fields.items()
                           if k in self.examples[0].__dict__])

    def extract_text_features(tokens):
        """
        Args:
            tokens: A list of tokens, where each token consists of a word,
                optionally followed by u"￨"-delimited features.
        Returns:
            A sequence of words, a sequence of features, and num of features.
        """
        if not tokens:
            return [], [], -1

        split_tokens = [token.split(u"￨") for token in tokens]
        split_tokens = [token for token in split_tokens if token[0]]
        token_size = len(split_tokens[0])

        assert all(len(token) == token_size for token in split_tokens), \
            "all words must have the same number of features"
        words_and_features = list(zip(*split_tokens))
        words = words_and_features[0]
        features = words_and_features[1:]

        return words, features, token_size - 1

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        """
        Args:
            seq: an iterator.

        Returns:
            the first thing returned by calling next() on the iterator
            and an iterator created by re-chaining that value to the beginning
            of the iterator.
        """
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
        """
        Args:
            data: the data to be set as the value of the attributes of
                the to-be-created `Example`, associating with respective
                `Field` objects with same key.
            fields: a dict of `torchtext.data.Field` objects. The keys
                are attributes of the to-be-created `Example`.

        Returns:
            the created `Example` object.
        """
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex
