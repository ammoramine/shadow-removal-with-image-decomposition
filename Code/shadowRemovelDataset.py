""" this module subclass the pytorch Dataset object, in order to produce a stream
 of data that will be fed to the shadow mask detector network,
 that takes as input  an image with shadow, and outputs a shadow mask

 The object streams pairs of image, that are preprocessed before to be fed to the network,

"""
from torch.utils.data import Dataset
import os,glob
from PIL import Image
import numpy as np
import torch

try:
    from .preprocess_module import transforms
except:
    from preprocess_module import transforms

class ShadowRemovalDataSet(Dataset):
    def __init__(self,path_data):
        self.path_data = path_data
        self.path_imgs_with_data = self.get_paths_images()

        self.inpt_img_shadow_mask = self.get_pairs_inpt_img_shadow_mask()

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
        image_inpt = Image.open(path_inpt)
        shdw_mask = Image.open(path_shdw_mask)
        return image_inpt,shdw_mask

    def __len__(self):
        return len(self.inpt_img_shadow_mask)

    def preprocess_inpt(self,inpt):
        """
            preprocess the input image, in order to be processed by
            the pretrained shadow mask detector network
        """
        out = transforms.preprocessor_shdw_mask_net(inpt)
        return out

    def collate_fn(self,list_pair_imgs):
        """
            get list of pairs of imgs, and transform it to two batches, that
            will be used to feed the neural network, the pairs of  image, must
            be from the pairs returned by the __getitem__ object.

        """
        batch_inpt_images = []
        batch_sdw_masks = []

        for inpt_image,shadow_mask in list_pair_imgs:
            inpt_image,shadow_mask = transforms.joint_transform(inpt_image, shadow_mask)
            batch_inpt_images.append(inpt_image)
            batch_sdw_masks.append(shadow_mask)


        batch_inpt_images = torch.Tensor([np.array(el) for el in batch_inpt_images])
        batch_inpt_images = torch.moveaxis(batch_inpt_images,3,1)
        batch_sdw_masks = torch.Tensor([np.array(el) for el in batch_sdw_masks])
        return batch_inpt_images,batch_sdw_masks

if __name__ == '__main__':

    import time
    start = time.time()
    pathTrainingData = "../Data/ISTD_Dataset/train"
    pathTestingData = "../Data/ISTD_Dataset/test"

    dtset = ShadowRemovalDataSet(pathTrainingData)


    els = []
    for i, el in enumerate(dtset):
        els.append(el)
        if i == 10:
            break

    list_pair_imgs = els

    res = dtset.collate_fn(list_pair_imgs)
    stop = time.time()

    print(stop-start)