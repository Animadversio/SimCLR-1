import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from os.path import join
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--ckpt_every_n_epocs', default=100, type=int,
                    help='Log every n epocs')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


parser.add_argument('--log_root', default="/scratch1/fs1/crponce/simclr_runs", \
    type=str, help='root folder to put logs')
parser.add_argument('--run_label', default="", \
    type=str, help='folder prefix to identify runs')


parser.add_argument('--crop', action='store_true', default=False, help='Enable crop')
parser.add_argument('--disable_blur', action='store_true', default=False,  # blur == True
    help='Do Deperministic Gaussian blur augmentation ')


parser.add_argument('--magnif', action='store_true', default=False,
    help='Do random magnif augmentation')
parser.add_argument('--sampling_bdr', default=16,
    type=int, help='border width for sampling the fixation point on the image')
parser.add_argument('--fov_size', default=20,
    type=float, help='Scaling coefficent for kernel of foveation blur')
parser.add_argument('--K', default=20,
    type=float, help='border width for sampling the fixation point on the image')
parser.add_argument('--cover_ratio', default=(0.05, 0.7),
    type=float, nargs="+", help='Range of fovea area as a ratio of the whole image size.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.blur = not args.disable_blur
    print(args)

    # from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
    # dataset = ContrastiveLearningDataset(args.data)
    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    from data_aug.dataset_w_salmap import Contrastive_STL10_w_salmap, Contrastive_STL10_w_CortMagnif
    from data_aug.saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density, RandomResizedCrop
    from data_aug.visualize_aug_dataset import visualize_augmented_dataset

    train_dataset = Contrastive_STL10_w_CortMagnif(dataset_dir=args.data,
            split="unlabeled", crop=args.crop)
    train_dataset.transform = train_dataset.get_simclr_magnif_transform(96, blur=args.blur, magnif=args.magnif, crop=args.crop,
                                    bdr=args.sampling_bdr, fov=args.fov_size, K=args.K, cover_ratio=args.cover_ratio)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args) # args carry the global config variables here.
        mtg = visualize_augmented_dataset(train_dataset)
        mtg.save(join(simclr.writer.log_dir, "sample_data_augs.png"))  # print sample data augmentations
        simclr.train(train_loader)


if __name__ == "__main__":
    main()





