# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain 
import io
import codecs 
import sys 

import torch 
import torchtext 

from onmt.utils.misc import aeq 

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class datasetBase(torchtext.data.Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(datasetBase, self).__reduce_ex__()

    def load_fields(self, vocab_dict):
        """ Load fields from vocab.pt, and set the `fields` attribute.

        Args:
            vocab_dict (dict): a dict of loaded vocab from vocab.pt file.
        """
        fields = load_fields_from_vocab(vocab_dict.items(), self.data_type)
        self.fields = dict([(k, f) for (k, f) in fields.items()
                           if k in self.examples[0].__dict__])