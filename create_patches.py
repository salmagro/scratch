import os 
import torch
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



tf_2_tensor = transforms.ToTensor()
tf_2_PIL = transforms.ToPILImage()

kh, kw = 256, 256 # kernel size
dh, dw = 256, 256 # stride dize

img_channels = 3
gt_channels = 1

number_images_row = 12431//256
number_images_col = 4136//256

def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

def visualize(patches):
    """Imshow for Tensor."""    
    fig = plt.figure(figsize=(number_images_row, number_images_col))
    for i in range(number_images_col):
        for j in range(number_images_row):
            inp = tf_2_PIL(patches[0][i][j])
            inp = np.array(inp)
            ax = fig.add_subplot(number_images_row, number_images_col, ((i*number_images_col)+j)+1, xticks=[], yticks=[])
            plt.imshow(inp)

def resize_tensor(input_tensor):
    _, x, y = input_tensor.shape
    rows = 16 # x // kh
    cols = 48 # y // kw

    out_tensor = input_tensor.unsqueeze(0)
    out_tensor = torch.nn.functional.interpolate(out_tensor,size=(rows * kh, cols * kw), mode='bilinear')
    
    return out_tensor.squeeze(0)



def create_patches_for_predict(source_dir, complete_save_dir, plotting=False, verbose=True):
  """
  args:
        source_dir = Path for reading original images
        complete_save_dir = Path for saving patches
        plotting = Flag to plot intermiadte results
        verbose = Flag for printing shapes
  """

  name_images = sorted(list_files(source_dir, "tif"))

  it = iter(name_images)

  iteration = 0
  print(name_images)

  for i in it:
    image_t_before = tf_2_tensor(Image.open(i))
    image_t_before = resize_tensor(image_t_before)
    print("image_t_before.shape =", image_t_before.shape) if verbose else None

    image_t_after = tf_2_tensor(Image.open(next(it)))
    image_t_after = resize_tensor(image_t_after)
    print("image_t_after.shape =", image_t_after.shape) if verbose else None
    
    # Creating patches
    patches_before = image_t_before.data.unfold(0, img_channels, img_channels).unfold(1, kh, dh).unfold(2, kw, dw)
    patches_after = image_t_after.data.unfold(0, img_channels, img_channels).unfold(1, kh, dh).unfold(2, kw, dw)
    
    if plotting:
      visualize(patches_before)
      visualize(patches_after)
    
    # itering over patches to save patches
    for j in range(number_images_col):
      for k in range(number_images_row):
        name_save_before = (complete_save_dir +"/predict_patch_before/" +  str(iteration +1) + '_' + str(j) + '_' + str(k) + '.tif')
        p = patches_before[0][j][k]
        save_image(p, name_save_before)
        # print("Patch before size",p.shape) if verbose else None
        
        name_save_after = (name_save_before).replace("predict_patch_before", "predict_patch_after")
        m = patches_after[0][j][k]
        save_image(m, name_save_after)
        # print("Patch after size",m.shape) if verbose else None

    print('file saved =', i)

    iteration += 1
    print("itearation =", iteration)

    if iteration == 1:
      break

"""
Due the limitation of memory, patches should be created in subset at a time, 
eg. 1, 2 or 3
Each batch will contain 6000 images.
"""


path_dir = '/home/sebastian/workspace/ETH/data/2021.03.22_mosaic'

save_dir = '/home/sebastian/workspace/ETH/data/2021.03.22_mosaic/patches'

create_patches_for_predict(path_dir, save_dir)