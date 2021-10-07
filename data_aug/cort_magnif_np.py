
from .aug_utils import send_to_clipboard
from scipy.misc import face
import torch
from torchvision import datasets

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from torch.nn.functional import interpolate
#%%
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from skimage.transform import rescale
from scipy.misc import face
from scipy.interpolate import griddata
img = rescale(face(), (0.25, 0.25, 1))
#%%
from scipy.stats import norm
def img_cortical_magnif_general(img, pnt, grid_func, demo=True):
    H, W, _ = img.shape
    XX_intp, YY_intp = grid_func(img, pnt)
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    img_cm = np.zeros_like(img.astype(np.float32))
    for ci in range(3):
        imgval = img[:, :, ci]
        img_interp = griddata((grid_y.flatten(), grid_x.flatten()), imgval.flatten(), \
                              (YY_intp.flatten(), XX_intp.flatten()))
        img_cm[:, :, ci] = img_interp.reshape([H, W])

    if demo:
        # % Visualize the Manified plot.
        figh, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].imshow(img_cm)
        axs[0].axis("off")
        axs[1].imshow(img)
        axs[1].axis("off")
        axs[1].scatter([pX], [pY], c='r', s=16, alpha=0.5)
        axs[2].scatter(XX_intp[::2, ::2].flatten(), YY_intp[::2, ::2].flatten(), c="r", s=0.25, alpha=0.2)
        axs[2].set_xlim([0, W])
        axs[2].set_ylim([0, H])
        axs[2].invert_yaxis()
        plt.show()
    return img_cm

def linear_separable_gridfun(img, pnt):
    H, W, _ = img.shape
    Hhalf, Whalf = H // 2, W // 2
    Hsum = Hhalf * (Hhalf + 1) / 2
    Wsum = Whalf * (Whalf + 1) / 2
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * np.arange(Whalf, 0, -1))
    Right_ticks = np.cumsum(RightDelta * np.arange(1, Whalf + 1, 1)) + pX
    Up_ticks = np.cumsum(UpDelta * np.arange(Hhalf, 0, -1))
    Down_ticks = np.cumsum(DownDelta * np.arange(1, Hhalf + 1, 1)) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp


def normal_gridfun(img, pnt):
    H, W, _ = img.shape
    Hhalf, Whalf = H // 2, W // 2
    Hdensity = norm.pdf(np.linspace(0, 2.25, Hhalf))
    Wdensity = norm.pdf(np.linspace(0, 2.25, Whalf))
    H_delta = (1 / Hdensity)
    W_delta = (1 / Wdensity)
    Hsum = H_delta.sum()
    Wsum = W_delta.sum()
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * W_delta[::-1])
    Right_ticks = np.cumsum(RightDelta * W_delta[::]) + pX
    Up_ticks = np.cumsum(UpDelta * H_delta[::-1])
    Down_ticks = np.cumsum(DownDelta * H_delta[::]) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp
#%%
img_cm = img_cortical_magnif_general(img, (80, 120), linear_separable_gridfun, demo=True)
#%%
img_cm = img_cortical_magnif_general(img, (80, 120), normal_gridfun, demo=True)
#%%
def radial_isotrop_gridfun(img, pnt, fov=20, K=20, M=20):
    H, W, _ = img.shape
    Hhalf, Whalf = H // 2, W // 2
    pY, pX = pnt
    maxdist = np.sqrt(max(H - pY, pY)**2 + max(W - pX, pX)**2)  # in pixel
    grid_y, grid_x = np.mgrid[-Hhalf+0.5:Hhalf+0.5, -Whalf+0.5:Whalf+0.5]
    ecc2 = grid_y**2 + grid_x**2 # R2
    ecc = np.sqrt(ecc2)
    # RadDistTfm = lambda R, R2 : (R < fov) * R + (R > fov) * (R**2 - fov**2 + fov)
    RadDistTfm = lambda R: (R < fov) * R + \
        (R > fov) * ((R + K) ** 2 / 2 / (fov + K) + fov - (fov + K) / 2)
    # fov = 10
    # M = 30
    # K = 30
    ecc_tfm = RadDistTfm(ecc, )
    coef = maxdist / ecc_tfm.max()
    XX_intp = pX + coef * ecc_tfm * (grid_x / ecc)
    YY_intp = pY + coef * ecc_tfm * (grid_y / ecc)
    return XX_intp, YY_intp

img_cm = img_cortical_magnif_general(img, (90, 120), radial_isotrop_gridfun, demo=True)

#%%
pnt = (80, 120)

H, W, _ = img.shape
Hhalf, Whalf = H // 2, W // 2
Hdensity = norm.pdf(np.linspace(0, 3, Hhalf))
Wdensity = norm.pdf(np.linspace(0, 3, Whalf))
H_delta = (1 / Hdensity)
W_delta = (1 / Wdensity)
Hsum = H_delta.sum()
Wsum = W_delta.sum()
pY, pX = pnt
UpDelta = pY / Hsum
LeftDelta = pX / Wsum
DownDelta = (H - pY) / Hsum
RightDelta = (W - pX) / Wsum
Left_ticks = np.cumsum(LeftDelta * W_delta[::-1])
Right_ticks = np.cumsum(RightDelta * W_delta[::]) + pX
Up_ticks = np.cumsum(UpDelta * H_delta[::-1])
Down_ticks = np.cumsum(DownDelta * H_delta[::]) + pY
X_ticks = np.hstack((Left_ticks, Right_ticks))
Y_ticks = np.hstack((Up_ticks, Down_ticks))
XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
#%
grid_y, grid_x = np.mgrid[0:H, 0:W]
img_cm = np.zeros_like(img.astype(np.float32))
for ci in range(3):
    imgval = img[:, :, ci]
    img_interp = griddata((grid_y.flatten(), grid_x.flatten()), imgval.flatten(), \
                        (YY_intp.flatten(), XX_intp.flatten()))
    img_cm[:, :, ci] = img_interp.reshape([H, W])
#%% Visualize the Manified plot.
figh, axs = plt.subplots(3, 1, figsize=(6, 12 ))
axs[0].imshow(img_cm)
axs[0].axis("off")
axs[1].imshow(img)
axs[1].axis("off")
axs[1].scatter([pX], [pY], c='r')
axs[2].scatter(XX_intp[::2,::2].flatten(), YY_intp[::2,::2].flatten(), c="r", s=0.25, alpha=0.2)
axs[2].set_xlim([0, W])
axs[2].set_ylim([0, H])
axs[2].invert_yaxis()
plt.show()
#%%
plt.scatter(XX_intp.flatten(), YY_intp.flatten(), c="r", s=1, alpha=0.1)
plt.xlim([0, W])
plt.ylim([0, H])
plt.show()
#%%
grid_x, grid_y = np.mgrid[0:H, 0:W]
img_cm = np.zeros_like(img.astype(np.float32))
img_interp = griddata((grid_x.flatten(), grid_y.flatten()), img.reshape([-1, 3]), \
                          (XX_intp.flatten(), YY_intp.flatten()))
img_cm = img_interp.reshape(img.shape)
#%%
RadDistTfm = lambda R: (R < fov) * R / M + \
        (R > fov) * ((R+K)**2 / 2 / (fov + K) + fov - (fov + K)/2) / M
fov = 10
M = 30
K = 30
xtick = np.linspace(0,100,1000)
plt.plot(xtick, RadDistTfm(xtick, ))
plt.show()
