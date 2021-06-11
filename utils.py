from fastai.vision.all import *
from torchvision.models import *

def _grand_grand_parent_idxs(items, name):
    def _inner(items, name): return mask2idxs(Path(o).parent.parent.parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]


def GrandGrandParentSplitter(train_name='train', valid_name='valid'):
    "Split `items` from the grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _grand_grand_parent_idxs(o, train_name),_grand_grand_parent_idxs(o, valid_name)
    return _inner