from io import BytesIO
import win32clipboard
def send_to_clipboard(image):
    """https://stackoverflow.com/questions/34322132/copy-image-to-clipboard"""
    output = BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

from data_aug.foveation import randomFoveated, FoveateAt
from scipy.misc import face
import torch
from torchvision import datasets
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from torch.nn.functional import interpolate

#%%
facetsr = torch.tensor(face()/255.0).float().permute([2,0,1]).unsqueeze(0)
facetsr_rsz = interpolate(facetsr, [192, 256])
#%% test different kerW parameters
views = randomFoveated(facetsr_rsz, 9, bdr=16, kerW_coef=0.01)
ToPILImage()(make_grid(views.squeeze(1),nrow=3)).show()
#%%

stldata = datasets.STL10(r"E:\Datasets", transform=ToTensor(), split="train")#unlabeled
#%%
imgtsr, _ = stldata[np.random.randint(5000)]
views = randomFoveated(imgtsr.unsqueeze(0), 9, bdr=12, kerW_coef=0.06, spacing=0.2, fov_area_rng=(0.01, 0.5))
mtg = ToPILImage()(make_grid(views, nrow=3))
send_to_clipboard(mtg)
mtg.show()

#%%

