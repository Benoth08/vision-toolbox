"""Decoupage train / val / test et controle des fuites."""

# rend ces fonctions accessibles directement par "from vtb.split import ..." (raccourci vers vtb.split.splits)
from vtb.split.splits import assign_folds, check_no_leakage, load_split, make_split, save_split, summarize_split

# __all__ : liste officielle des noms exportes par ce paquet (ce que "from vtb.split import *" importe)
__all__ = ["assign_folds", "check_no_leakage", "load_split", "make_split", "save_split", "summarize_split"]
