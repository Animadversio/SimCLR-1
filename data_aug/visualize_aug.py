#%%
import numpy as np
import matplotlib.pylab as plt
from data_aug.dataset_w_salmap import Contrastive_STL10_w_salmap
from data_aug.saliency_random_cropper import RandomResizedCrop_with_Density, RandomCrop_with_Density
crop_temperature = 1.5
pad_img = True
dataset_dir = r"E:\Datasets"
cropper = RandomResizedCrop_with_Density(96, temperature=crop_temperature, pad_if_needed=pad_img)
train_dataset = Contrastive_STL10_w_salmap(dataset_dir=dataset_dir,
           density_cropper=cropper, split="unlabeled")  # imgv1, imgv2 =  saldataset[10]
#%%
def visualize_samples(train_dataset, idxs=None):
    imgs, _ = train_dataset[1]
    figh, axs = plt.subplots(len(imgs), 10, figsize=(15, len(imgs) * 1.6))
    idx_col = [] if idxs is None else idxs
    for i in range(10):
        if idxs is None:
            idx = np.random.randint(1E5)
            idx_col.append(idx)
        else:
            idx = idxs[i]
        imgs , _ = train_dataset[idx]
        for j in range(len(imgs)):
            axs[j, i].imshow(imgs[j].permute([1,2,0]))
            axs[j, i].axis("off")

    figh.show()
    return figh, idx_col
#%%
idxs = [96659, 54019, 88327, 81148, 98469, 77493, 131, 58202, 66666, 65017]
#%%
# crop_temperature = 1.5
# pad_img = True
cropper = RandomResizedCrop_with_Density(96, \
        temperature=crop_temperature, pad_if_needed=pad_img)
cropper.pad_if_needed = False
cropper.temperature = 15
train_dataset.n_views = 7
train_dataset.density_cropper = cropper
_, idxs = visualize_samples(train_dataset, idxs)
#%%

# figh.savefig("/scratch1/fs1/crponce/Datasets/example%03d.png"%np.random.randint(1E3))
#%%
from torchvision.transforms import RandomResizedCrop
rndcropper = RandomResizedCrop(96,)
bsl_cropper = lambda img, salmap: rndcropper(img)
train_dataset.density_cropper = bsl_cropper
_, idxs = visualize_samples(train_dataset, idxs)

