
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pylab as plt

class STL10_w_salmap(Dataset):
    """Face Landmarks dataset."""

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
        img, label = self.dataset.__getitem__(idx)
        salmap = self.salmaps[idx, :, :, :].astype('float')
        if self.transform:
            sample = self.transform(img)
        return (img, salmap), label  # labels can be dropped.
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        #
        # img_name = join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        # return sample

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


