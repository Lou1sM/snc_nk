import numpy as np
from dl_utils.label_funcs import get_num_labels


def cond_ent_for_alignment(x,y):
    num_classes = get_num_labels(y)
    dividers_idx = np.arange(0,len(x),len(x)/num_classes).astype(int)
    bin_dividers = np.sort(x)[dividers_idx]
    bin_vals = sum([x<bd for bd in bin_dividers])
    total_ent = 0
    for bv in np.unique(bin_vals):
        bv_mask = bin_vals==bv
        gts_for_this_val = y[bv_mask]
        new_ent = np_ent(gts_for_this_val)
        total_ent += new_ent*bv_mask.sum()/len(x)
    return total_ent

def combo_acc(cs1,cs2):
    return np.logical_and(cs1,cs2).mean()

def round_maybe_list(x,round_factor=100):
    if isinstance(x,float):
        return round(round_factor*x,2)
    elif isinstance(x,list):
        return [round(round_factor*item,2) for item in x]
    else: return x

def normalize(x):
    return (x-x.min()) / (x.max() - x.min())

def load_trans_dict(trans_dict_fpath):
    return dict([p for p in np.load(trans_dict_fpath)])

def np_ent(p):
    counts = np.bincount(p)
    return stats.entropy(counts)
