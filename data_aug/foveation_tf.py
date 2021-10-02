import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import math
import scipy

def showimg(image,figsize=[8,8]):
  if len(image.shape)==4:
    for i in range(image.shape[0]):
      figh,ax = showimg(image[i], figsize=figsize)
  else:
    figh,ax = plt.subplots(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
  return figh,ax


pi = tf.constant(math.pi)
@tf.function
def cosfunc(x):
  """The cosine square smoothing function"""
  Lower = tf.square(tf.cos(pi*(x + 1/4)));
  Upper = 1 - tf.square(tf.cos(pi*(x - 3/4)));
  # print(tf.logical_and((x <= -1/4), (x > -3/4)).dtype)
  fval = tf.where(tf.logical_and((x <= -1/4), (x >-3/4)), Lower, tf.zeros(1)) + \
      tf.where(tf.logical_and((x >= 1/4), (x <= 3/4)), Upper, tf.zeros(1)) + \
      tf.where(tf.logical_and((x < 1/4), (x > -1/4)), tf.ones(1), tf.zeros(1))
  return fval

@tf.function
def rbf(ecc, N, spacing, e_o=1.0):
  """ Number N radial basis function
  ecc: eccentricities, tf array.  
  N: numbering of basis function, starting from 0. 
  spacing: log scale spacing of ring radius (deg), scalar.
  e_o: radius of 0 string, scalar. 
  """
  spacing = tf.convert_to_tensor(spacing, dtype="float32")
  e_o = tf.convert_to_tensor(e_o, dtype="float32")
  preinput = tf.divide(tf.math.log(ecc) - (tf.math.log(e_o) + (N + 1) * spacing), spacing)
  ecc_basis = cosfunc(preinput);
  return ecc_basis

@tf.function
def fov_rbf(ecc, spacing, e_o=1.0):
  """Initial radial basis function
  """
  spacing = tf.convert_to_tensor(spacing,dtype="float32")
  e_o = tf.convert_to_tensor(e_o,dtype="float32")
  preinput = tf.divide(tf.math.log(ecc) - tf.math.log(e_o), spacing)
  preinput = tf.clip_by_value(preinput, tf.zeros(1), tf.ones(1)) # only clip 0 is enough.
  ecc_basis = cosfunc(preinput);
  return ecc_basis

# these seems to be hard to be form as tf.function, 
def FoveateAt(img, pnt:tuple, kerW_coef=0.04, e_o=1, N_e=5, spacing=0.5, demo=False):
  """Apply foveation transform at (x,y) coordinate `pnt` to `img`

  Parameters: 
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  H, W = img.shape[0], img.shape[1] # if this is fixed then these two steps could be saved
  XX, YY = tf.meshgrid(tf.range(W),tf.range(H))
  deg_per_pix = 20/math.sqrt(H**2+W**2);
  # pixel coordinate of fixation point.
  xid, yid = pnt
  D2fov = tf.sqrt(tf.cast(tf.square(XX - xid) + tf.square(YY - yid), 'float32'))
  D2fov_deg = D2fov * deg_per_pix
  # maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]) # maximal deviation at 4 corner
  # maxecc = tf.reduce_max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  # maxecc = max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  maxecc = math.sqrt(max(xid, W-xid)**2 + max(yid, H-yid)**2) * deg_per_pix #max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]])
  # e_r = maxecc; # 15
  if N_e is None:
    N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing).astype("int32")
  rbf_basis = fov_rbf(D2fov_deg, spacing, e_o)
  finalimg = tf.expand_dims(rbf_basis,-1)*img
  for N in range(N_e):
    rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
    mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
    kerW = kerW_coef * mean_dev / deg_per_pix
    kerSz = int(kerW * 3)
    img_gsft = tfa.image.gaussian_filter2d(img, filter_shape=(kerSz, kerSz), sigma=kerW, padding='REFLECT')
    finalimg = finalimg + tf.expand_dims(rbf_basis,-1)*img_gsft
  
  if demo: # Comment out this part when really run. 
    figh,ax = plt.subplots(figsize=[10,10])
    plt.imshow(finalimg)
    plt.axis("off")
    plt.show()
    figh,ax = plt.subplots(figsize=[10,10])
    plt.imshow(finalimg)
    plt.axis("off")
    vis_belts(ax, img, pnt, kerW_coef, e_o, N_e, spacing)
    figh.show()
  return finalimg 


def randomFoveated(img, pntN:int, kerW_coef=0.04, e_o=1, N_e=None, spacing=0.5, bdr=32):
  """Randomly apply `pntN` foveation transform to `img`

  Parameters: 
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  H, W = img.shape[0], img.shape[1] # if this is fixed then these two steps could be saved
  XX, YY = tf.meshgrid(tf.range(W),tf.range(H))
  deg_per_pix = 20/math.sqrt(H**2+W**2);
  finimg_list = []
  xids = tf.random.uniform(shape=[pntN,], minval=bdr, maxval=W-bdr, dtype=tf.int32)
  yids = tf.random.uniform(shape=[pntN,], minval=bdr, maxval=H-bdr, dtype=tf.int32)
  for it in range(pntN):
    xid, yid = xids[it], yids[it] # pixel coordinate of fixation point.
    D2fov = tf.sqrt(tf.cast(tf.square(XX - xid) + tf.square(YY - yid), 'float32'))
    D2fov_deg = D2fov * deg_per_pix
    maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]) # maximal deviation at 4 corner
    e_r = maxecc; # 15
    if N_e is None:
      N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing+1).astype("int32")
    # spacing = tf.convert_to_tensor((math.log(e_r) - math.log(e_o)) / N_e);
    # spacing = tf.convert_to_tensor(spacing, dtype="float32");
    rbf_basis = fov_rbf(D2fov_deg,spacing,e_o)
    finalimg = tf.expand_dims(rbf_basis, -1)*img
    for N in range(N_e):
      rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
      mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
      kerW = kerW_coef * mean_dev / deg_per_pix
      kerSz = int(kerW * 3)
      img_gsft = tfa.image.gaussian_filter2d(img, filter_shape=(kerSz, kerSz), sigma=kerW, padding='REFLECT')
      finalimg = finalimg + tf.expand_dims(rbf_basis, -1)*img_gsft
    finimg_list.append(finalimg)
  finimgs = tf.stack(finimg_list)
  return finimgs

def vis_belts(ax, img, pnt, kerW_coef=0.04, e_o=1, N_e=None, spacing=0.5):
  """A visualization helper for parameter tuning purpose.
    It plot out the masking belts for the computation, with the flat region and the smoothing region.

  """
  if ax is None: ax = plt.gca()
  H, W = img.shape[0], img.shape[1]
  deg_per_pix = 20/math.sqrt(H**2+W**2);
  # pixel coordinate of fixation point.
  xid, yid = pnt
  if N_e is None:
    maxecc = math.sqrt(max(xid, H-xid)**2 + max(yid,W-yid)**2) * deg_per_pix
    N_e = np.ceil((np.log(maxecc)-np.log(e_o))/spacing).astype("int32")
    
  print("radius of belt center:",)
  for N in range(N_e):
    radius = math.exp(math.log(e_o) + (N+1) * spacing) / deg_per_pix
    inner_smooth_rad = math.exp(math.log(e_o) + (N+1-1/4) * spacing) / deg_per_pix
    inner_smooth_rad2 = math.exp(math.log(e_o) + (N+1-3/4) * spacing) / deg_per_pix
    outer_smooth_rad = math.exp(math.log(e_o) + (N+1+1/4) * spacing) / deg_per_pix
    outer_smooth_rad2 = math.exp(math.log(e_o) + (N+1+3/4) * spacing) / deg_per_pix
    circle1 = plt.Circle((xid, yid), inner_smooth_rad, color='r', linestyle=":", fill=False, clip_on=False)
    circle12 = plt.Circle((xid, yid), inner_smooth_rad2, color='r', linestyle=":", fill=False, clip_on=False)
    circle3 = plt.Circle((xid, yid), outer_smooth_rad, color='r', linestyle=":", fill=False, clip_on=False)
    circle32 = plt.Circle((xid, yid), outer_smooth_rad2, color='r', linestyle=":", fill=False, clip_on=False)
    circle2 = plt.Circle((xid, yid), radius, color='k', linestyle="-.", fill=False, clip_on=False)
    ax.plot(xid,yid,'ro')
    ax.add_patch(circle1)
    ax.add_patch(circle12)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle32)
