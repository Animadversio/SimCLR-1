import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from torchvision.transforms import functional as TF
def _setup_size(size, error_msg):
  if isinstance(size, numbers.Number):
    return int(size), int(size)

  if isinstance(size, Sequence) and len(size) == 1:
    return size[0], size[0]

  if len(size) != 2:
    raise ValueError(error_msg)

  return size

def unravel_indices(
  indices: torch.LongTensor,
  shape: Tuple[int, ...],
) -> torch.LongTensor:
  r"""Converts flat indices into unraveled coordinates in a target shape.
  Args:
    indices: A tensor of (flat) indices, (*, N).
    shape: The targeted shape, (D,).
  Returns:
    The unraveled coordinates, (*, N, D).
  """
  coord = []
  for dim in reversed(shape):
    coord.append(indices % dim)
    indices = indices // dim
  coord = torch.stack(coord[::-1], dim=-1)
  return coord

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, pad=0):
  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

  mean = (kernel_size - 1)/2.
  variance = sigma**2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

  gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, groups=channels, bias=False, padding=pad)

  gaussian_filter.weight.data = gaussian_kernel
  gaussian_filter.weight.requires_grad = False
  
  return gaussian_filter

class RandomCrop_with_Density(torch.nn.Module):
  """Crop the given image at a random location determined by a density map. 
  If the image is torch Tensor, it is expected
  to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
  but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (h, w), a square crop (size, size) is
      made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    padding (int or sequence, optional): Optional padding on each border
      of the image. Default is None. If a single int is provided this
      is used to pad all borders. If sequence of length 2 is provided this is the padding
      on left/right and top/bottom respectively. If a sequence of length 4 is provided
      this is the padding for the left, top, right and bottom borders respectively.

      .. note::
        In torchscript mode padding as single int is not supported, use a sequence of
        length 1: ``[padding, ]``.
    pad_if_needed (boolean): It will pad the image if smaller than the
      desired size to avoid raising an exception. Since cropping is done
      after padding, the padding seems to be done at a random offset.
    fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
      length 3, it is used to fill R, G, B channels respectively.
      This value is only used when the padding_mode is constant.
      Only number is supported for torch Tensor.
      Only int or str or tuple value is supported for PIL Image.
    padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
      Default is constant.

      - constant: pads with a constant value, this value is specified with fill

      - edge: pads with the last value at the edge of the image.
        If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

      - reflect: pads with reflection of image without repeating the last value on the edge.
        For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
        will result in [3, 2, 1, 2, 3, 4, 3, 2]

      - symmetric: pads with reflection of image repeating the last value on the edge.
        For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
        will result in [2, 1, 1, 2, 3, 4, 4, 3]
  """
  def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", temperature=4, 
               device="cuda", avgpool=True, density_sigma=31,):
    super().__init__()

    self.size = tuple(_setup_size(
      size, error_msg="Please provide only two dimensions (h, w) for size."
    ))
    self.padding = padding
    self.pad_if_needed = pad_if_needed
    self.fill = fill
    self.padding_mode = padding_mode
    self.temperature = temperature
    if avgpool:
      self.salmapPooling = nn.AvgPool2d(self.size, stride=1, padding=0) # note the pooling of salmap can use a smaller window than outputsize
    else:
      self.salmapPooling = get_gaussian_kernel(self.size[0], sigma=density_sigma, channels=1).to(device)
    self.device = device

  @staticmethod
  def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
      img (PIL Image or Tensor): Image to be cropped.
      output_size (tuple): Expected output size of the crop.

    Returns:
      tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = TF._get_image_size(img)
    th, tw = output_size

    if h + 1 < th or w + 1 < tw:
      raise ValueError(
        "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
      )

    if w == tw and h == th:
      return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1, )).item()
    j = torch.randint(0, w - tw + 1, size=(1, )).item()
    return i, j, th, tw

  def sample_crops(self, img: Tensor, output_size: Tuple[int, int], salmap):
    # density: 4d tensor with [1,1,H,W]
    th, tw = output_size
    w, h = TF._get_image_size(img)

    densitymap = torch.exp((salmap.to(self.device) - torch.logsumexp(salmap.to(self.device), (0,1,2,3), keepdim=True)) / self.temperature)
    densitymap_pad = TF.pad(densitymap, self.padding, padding_mode='constant', fill=0)
    centermap = self.salmapPooling(densitymap_pad)
    flat_idx = torch.multinomial(centermap.flatten(), 1, replacement=True).cpu()
    coord = unravel_indices(flat_idx, centermap[0, 0, :, :].shape)
    i, j = coord[0,0], coord[0,1]
    return i, j, th, tw

  def forward(self, img, density=None):
    """
    Args:
      img (PIL Image or Tensor): Image to be cropped.
      density (Tensor): same size as unpadded image. density to sample the fixation signal.  

    Returns:
      PIL Image or Tensor: Cropped image.
    """
    if self.padding is not None:
      img = TF.pad(img, self.padding, self.fill, self.padding_mode)

    width, height = TF._get_image_size(img)
    # pad the width if needed
    if self.pad_if_needed and width < self.size[1]:
      padding = [self.size[1] - width, 0]
      img = TF.pad(img, padding, self.fill, self.padding_mode)
    # pad the height if needed
    if self.pad_if_needed and height < self.size[0]:
      padding = [0, self.size[0] - height]
      img = TF.pad(img, padding, self.fill, self.padding_mode)

    if density is None:
      i, j, h, w = self.get_params(img, self.size)
    else:
      i, j, h, w = self.sample_crops(img, self.size, density)

    return TF.crop(img, i, j, h, w)


  def __repr__(self):
    return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)