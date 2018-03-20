# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午1:29
# @Author  : zhanzecheng
# @File    : word2vec_helper.py
# @Software: PyCharm
"""
import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

import re
import unicodedata
import random
import shutil
# import subprocesss
from contextlib import contextmanager
import numpy as np
import numbers
from six import string_types, u
import collections
import warnings

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode

try:
    from smart_open import smart_open
except ImportError:
    print("smart_open library not found; falling back to local-filesystem-only")


    def make_closing(base, **attrs):
        if not hasattr(base, '__enter__'):
            attrs['__enter__'] = lambda self: self
        if not hasattr(base, '__exit__'):
            attrs['__exit__'] = lambda self, type, value, traceback: self.close()
        return type('Closing' + base.__name__, (base, object), attrs)


    def smart_open(fname, mode='rb'):
        _, ext = os.path.splitext(fname)
        if ext == '.bz2':
            from bz2 import BX2File
            return make_closing(BZ2File)(fname, mode)
        if ext == '.gz':
            from gzip import GzipFile
            return make_closing(GzipFile)(fname, mode)
        return open(fname, mode)

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


def get_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a np.random.RandomState instance' % seed)


def tokenize(
        text,
        lowercase=False,
        deacc=False,
        encoding='utf8',
        errors="strict",
        to_lower=False,
        lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    """
    Convert a document into a list of tokens.
    This lowercases, tokenizes, de-accents (optional). -- the output are final
    tokens = unicode strings, that won't be processed any further.
    """
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


from numpy import dot
from numpy.linalg import norm

cosine = lambda a, b: dot(a, b) / (norm(a) * norm(b))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def call_on_class_only(*args, **kwargs):
    """Raise exception when load methods are called on instance"""
    raise AttributeError('This method should be called on a class object.')


def is_digit(obj):
    '''
    Check if an object is Number
    '''
    return isinstance(obj, (numbers.Integral, numbers.Complex, numbers.Real))


def is_zhs(str):
    '''
    Check if str is Chinese Word
    '''
    for i in str:
        if not is_zh(i):
            return False
    return True


def is_zh(ch):
    """return True if ch is Chinese character.
    full-width puncts/latins are not counted in.
    """
    x = ord(ch)
    # CJK Radicals Supplement and Kangxi radicals
    if 0x2e80 <= x <= 0x2fef:
        return True
    # CJK Unified Ideographs Extension A
    elif 0x3400 <= x <= 0x4dbf:
        return True
    # CJK Unified Ideographs
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif 0xf900 <= x <= 0xfad9:
        return True
    # CJK Unified Ideographs Extension B
    elif 0x20000 <= x <= 0x2a6df:
        return True
    else:
        return False


def is_punct(ch):
    x = ord(ch)
    # in no-formal literals, space is used as punctuation sometimes.
    if x < 127 and ascii.ispunct(x):
        return True
    # General Punctuation
    elif 0x2000 <= x <= 0x206f:
        return True
    # CJK Symbols and Punctuation
    elif 0x3000 <= x <= 0x303f:
        return True
    # Halfwidth and Fullwidth Forms
    elif 0xff00 <= x <= 0xffef:
        return True
    # CJK Compatibility Forms
    elif 0xfe30 <= x <= 0xfe4f:
        return True
    else:
        return False