
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets, transforms, utils
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pylab as plt
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator

class STL10_w_salmap(Dataset):
    """ Return STL image with saliency maps """

    def __init__(self, dataset_dir=r"/scratch1/fs1/crponce/Datasets", transform=None, split="unlabeled"):
        """
        Args:
            dataset_dir (string): Directory with all the images. E:\Datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: "unlabeled"
        """
        self.dataset = datasets.STL10(dataset_dir, split=split, download=True,
                                 transform=None,)
        self.salmaps = np.load(join(dataset_dir, "stl10_unlabeled_salmaps_salicon.npy")) # stl10_unlabeled_saliency.npy
        assert len(self.dataset) == self.salmaps.shape[0]
        # transforms.Compose([transforms.ToTensor(),
        #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        self.root_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx) # img is PIL.Image, label is xxxx 
        salmap = self.salmaps[idx, :, :, :].astype('float') # numpy.ndarray
        if self.transform:
            img = self.transform(img)
        salmap_tsr = F.interpolate(torch.tensor(salmap).unsqueeze(0),[224,224]).float()
        return (img, salmap), label  # labels can be dropped.


from .saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density
class Contrastive_STL10_w_salmap(Dataset):
    """ Return Crops of STL10 images with saliency maps """

    @staticmethod
    def get_simclr_post_crop_transform(size, s=1, blur=True):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),] +
                                            ([GaussianBlur(kernel_size=int(0.1 * size)),]  if  blur  else  []) +
                                              [transforms.ToTensor()])
        # transforms.Compose([transforms.ToTensor(),
        #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                                          std=(0.2023, 0.1994, 0.2010))])
        return data_transforms

    def __init__(self, dataset_dir=r"/scratch1/fs1/crponce/Datasets", \
        density_cropper=RandomResizedCrop_with_Density((96, 96),), \
        transform_post_crop=None, split="unlabeled", n_views=2):
        """
        Args:
            dataset_dir (string): Directory with all the images. E:\Datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: "unlabeled"
        """
        self.dataset = datasets.STL10(dataset_dir, split=split, download=True,
                                 transform=None,)
        self.salmaps = np.load(join(dataset_dir, "stl10_unlabeled_salmaps_salicon.npy")) # stl10_unlabeled_saliency.npy
        assert len(self.dataset) == self.salmaps.shape[0]
        self.root_dir = dataset_dir
        self.density_cropper = density_cropper
        if transform_post_crop is not None:
            self.transform = transform_post_crop
        else:
            self.transform = self.get_simclr_post_crop_transform(96, s=1, blur=True)
        self.n_views = n_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx) # img is PIL.Image, label is xxxx 
        salmap = self.salmaps[idx, :, :, :].astype('float') # numpy.ndarray
        salmap_tsr = F.interpolate(torch.tensor(salmap).unsqueeze(0), [96, 96]).float()
        sal_crops = [self.density_cropper(img, salmap_tsr) for i in range(n_views)]

        if self.transform:
            imgs = [self.transform(cropview) for cropview in sal_crops]
            return imgs
        else:
            return sal_crops

    


def visualize_samples(saldataset):
    figh, axs = plt.subplots(2, 10, figsize=(14, 3.5))
    for i in range(10):
        idx = np.random.randint(1E5)
        (img, salmap) , _ = saldataset[idx]
        axs[0, i].imshow(img.permute([1,2,0]))
        axs[0, i].axis("off")
        axs[1, i].imshow(salmap[0])
        axs[1, i].axis("off")
    figh.savefig("/scratch1/fs1/crponce/Datasets/example%03d.png"%np.random.randint(1E3))
# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                     root_dir='data/faces/')
#
# fig = plt.figure()
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break
#%%
# import matplotlib.pylab as plt
# img, salmap = STL10_sal[89930]
# fig, axs = plt.subplots(1, 2, figsize=[8, 4.5])
# axs[0].imshow(img[0])
# axs[1].imshow(salmap[0, 0, :, :])
# plt.show()


