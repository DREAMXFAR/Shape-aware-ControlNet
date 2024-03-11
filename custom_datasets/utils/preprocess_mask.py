"""
refer to https://github.com/PITI-Synthesis/PITI

run: pip install blobfile
"""
import os 
import numpy as np
from PIL import Image
import blobfile as bf
from tqdm import tqdm


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
    

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] =  r
            cmap[i, 1] =  g
            cmap[i, 2] =  b
     
    return cmap


class Colorize(object):
    def __init__(self, n=182):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1])) 
     
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image ) 
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def preprocess_mask(split):
    colorizer = Colorize(182)

    COCOSTUFF_ROOT = r"/dat03/xuanwenjie/datasets/COCO/stuff_anno"
    file_path = os.path.join(COCOSTUFF_ROOT, r"{}2017".format(split))

    SAVE_ROOT = os.path.join(COCOSTUFF_ROOT, r"colored_{}2017".format(split))
    if not os.path.exists(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
    
    all_files = _list_image_files_recursively(file_path)

    pbar = tqdm(all_files)
    for name in pbar:
        img = Image.open(name).convert('L')
        img_a = np.array(img)
        img_c = np.transpose(colorizer(img_a) , (1,2,0))
        #print(img_c.astype(np.uint8) )
        img_c = Image.fromarray(img_c.astype(np.uint8))

        save_path = name.replace('{}2017'.format(split), 'colored_{}2017'.format(split))
        img_c.save(save_path)
        pbar.set_description(save_path)
        # print('==> ', save_path)
        
        ### debug use 
        # break


if __name__ == '__main__':
    ### 1. process val
    split = 'val'
    preprocess_mask(split)
    
    ### processs train
    # split = 'train'
    # preprocess_mask(split)
