""" this module subclass the pytorch Dataset object, in order to produce a stream
 of data that will be fed to the shadow mask detector network,
 that takes as input  an image with shadow, and outputs a shadow mask

 The object streams pairs of image, that are preprocessed before to be fed to the network,

"""
from torch.utils.data import Dataset
import os,glob
from PIL import Image
import utils,torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import torch

try:
    from .preprocess_module import custom_transforms
except:
    from preprocess_module import custom_transforms

class ShadowRemovalDataSet(Dataset):
    def __init__(self,path_data,joint_transform=None,inpt_img_transform=None,out_img_transform=None,truncate=None):
        """
            takes as input a folder containing 2 subfolders (at least)
            one that contains the input image with shadow, with suffix "_A"
            and another one that contains shadow masks with suffix "_B"
        """
        self.path_data = path_data
        self.path_imgs_with_data = self.get_paths_images()

        if truncate is not None:
            if isinstance(truncate,int):
                self.path_imgs_with_data = self.path_imgs_with_data[:truncate]

        self.inpt_img_shadow_mask = self.get_pairs_inpt_img_shadow_mask()

        self.joint_transform = joint_transform
        self.inpt_img_transform = inpt_img_transform
        self.out_img_transform = out_img_transform

    def get_pairs_inpt_img_shadow_mask(self):
        """return a list , that outputs pair of inputs image, and shadow mask"""
        pairs = [(el,el.replace("_A","_B")) for el in self.path_imgs_with_data]
        return pairs
    def get_paths_images(self):
        """of all the images with shadow used as input to the network , before preprocessing"""
        basename = os.path.basename(self.path_data)
        path_img_with_shadows = os.path.join(self.path_data,basename+"_A")
        path_imgs_with_data = glob.glob(os.path.join(path_img_with_shadows,"*"))
        return path_imgs_with_data
    def __getitem__(self, index):
        """
            returns the input image (with shadow), and the shadow mask,whose paths are at  index
            "index" of the list sekf.inpt_img_shadow_mask
        """
        path_inpt,path_shdw_mask = self.inpt_img_shadow_mask[index]
        inpt_image = Image.open(path_inpt)#.convert("RGB")
        shadow_mask = Image.open(path_shdw_mask)

        if self.joint_transform is not None:
            inpt_image, shadow_mask = self.joint_transform(inpt_image, shadow_mask)
        if self.inpt_img_transform is not None:
            inpt_image = self.inpt_img_transform(inpt_image)
        if self.out_img_transform is not None:
            shadow_mask = self.out_img_transform(shadow_mask)
        return inpt_image,shadow_mask

    def __len__(self):
        return len(self.inpt_img_shadow_mask)

    def get_first_elements(self,nb):
        iterator = iter(self)
        els = [next(iterator) for el in range(nb)]
        return els
    #TODO : test if appyling tensor transform on set of pil imaages is faster than appyling tensor
    # transform at the end

    def sample(self,with_path=False):
        index = np.random.randint(len(self))
        res = self.__getitem__(index)
        if with_path:
            return res,(index,self.path_imgs_with_data[index])
        else:
            return res

    def collate_fn(self,list_pair_imgs):
        """
            get list of pairs of imgs, and transform it to two batches, that
            will be used to feed the neural network, the pairs of  image, must
            be from the pairs returned by the __getitem__ object.

        """
        batch_inpt_images = []
        batch_sdw_masks = []

        for inpt_image,shadow_mask in list_pair_imgs:
            batch_inpt_images.append(inpt_image)
            batch_sdw_masks.append(shadow_mask)
        batch_inpt_images = torch.stack(batch_inpt_images)
        batch_sdw_masks = torch.stack(batch_sdw_masks)

        batch_inpt_images = batch_inpt_images.to(device)
        batch_sdw_masks = batch_sdw_masks.to(device)

        return batch_inpt_images,batch_sdw_masks



if __name__ == '__main__':

    import time
    start = time.time()
    pathTrainingData = "../Data/ISTD_Dataset/train"
    pathTestingData = "../Data/ISTD_Dataset/test"

    dtset = ShadowRemovalDataSet(
            pathTrainingData,
    joint_transform=custom_transforms.joint_transform,
    inpt_img_transform = custom_transforms.inpt_img_transform,
    out_img_transform = custom_transforms.out_img_transform
            )

    dtset = ShadowRemovalDataSet(pathTrainingData)
    els = dtset.get_first_elements(10)

    list_pair_imgs = els

    # res = dtset.collate_fn(list_pair_imgs)
    stop = time.time()

    print(stop-start)