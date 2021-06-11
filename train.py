from fastai.vision.all import *
from fastcore.script import *
from torchvision.models import *
from fasterai.sparse.all import *
from utils import *


def get_dls(size,bs, dataset, device):

    if dataset=='CIFAR': path = URLs.CIFAR ; splitter = GrandparentSplitter(valid_name='test')
    elif dataset == 'CIFAR_100': path = URLs.CIFAR_100 ; splitter=GrandGrandParentSplitter(valid_name='test')
    elif dataset == 'CALTECH_101': path = URLs.CALTECH_101 ; splitter=RandomSplitter()
    source = untar_data(path)
    blocks=(ImageBlock, CategoryBlock)
    item_tfms=Resize(size)

    stats = imagenet_stats if dataset == 'CALTECH_101' else cifar_stats
    batch_tfms = [Normalize.from_stats(*stats), *aug_transforms()] 

    dblock = DataBlock(blocks=blocks,
            get_items = get_image_files,
            splitter=splitter,     
            get_y=parent_label,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms)
    
    return dblock.dataloaders(source, bs=bs, device=device)

@call_parse
def main(
    cuda:  Param("Which device to use", str)='cuda:2',
    lr:    Param("Learning rate", float)=1e-3,
    size:  Param("Size", int)=224,
    epochs:Param("Number of epochs", int)=25,
    bs:    Param("Batch size", int)=32,
    arch:  Param("Architecture", str)='resnet18',
    dataset : Param("Dataset", str)='CIFAR',
    runs:  Param("Number of times to repeat training", int)=3, 
    schedule: Param("Pruning Schedule", str)='one_shot', 
    start_epoch: Param('Starting epoch', int)=0

):

    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(cuda[-1]))

    sched = globals()[schedule]

    sparsities = [80, 90, 95]

    for sp in sparsities: 

        for run in range(runs):

            m = globals()[arch]
            dls = get_dls(size, bs, dataset, device)

            print(f'Run: {run}')
            learn = Learner(dls, m(num_classes=dls.c), metrics=accuracy)

            sp_cb = SparsifyCallback(sp, 'weight', 'local', large_final, sched, start_epoch=start_epoch)

            learn.fit_one_cycle(epochs, lr, cbs=[sp_cb])

            del learn
            del dls