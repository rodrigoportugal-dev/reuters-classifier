"""Text tokenization.

Two tokenizers are available so their effect can be measured:

* ``basic``: the one used in the original submission, ``text.lower().split()``.
  Punctuation stays attached to words, so "oil", "oil." and "oil," are three
  different features.
* ``clean`` (default): keeps only alphabetic tokens of 2+ letters and removes
  English stop words plus the corpus signature "reuter" that ends every article.
"""

import re

TOKEN_RE = re.compile(r"[a-z]+")

# Standard English stop-word list (short version) plus Reuters-specific noise.
STOP_WORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that
the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you
your yours yourself yourselves said says say reuter reuters
""".split())


def tokenize_basic(text):
    return text.lower().split()


def tokenize_clean(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOP_WORDS]


TOKENIZERS = {"basic": tokenize_basic, "clean": tokenize_clean}


def get_tokenizer(name):
    try:
        return TOKENIZERS[name]
    except KeyError:
        raise ValueError(f"Unknown tokenizer '{name}'. Use one of: {', '.join(TOKENIZERS)}")
