# import albumentations.augmentations.functional as F
import numpy as np
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class StepcropAlbu(ImageOnlyTransform):
    """Rotate the input by 90 degrees zero or more times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, *args, **kwargs):
        super(StepcropAlbu, self).__init__(*args, **kwargs)

    def apply(self, img, **params):
        return self.DataAugmentation3(img)

    def DataAugmentation3(self,image):
        n = 8
        im_list = []
        iv_list = []
        patch_initial = np.array([0,0])
        patch_scale = 1/n #find 5 patch on diagonal
        smaller_dim = np.min(image.shape[0:2])
        #print(smaller_dim)
        image = cv2.resize(image,((smaller_dim,smaller_dim)))
        patch_size = int(patch_scale * smaller_dim)
        #print(patch_size)
        for i in range(n):
            patch_x = patch_initial[0]
            patch_y = patch_initial[1]
            patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
            #print(patch_image.shape)
            #patch_image = zoomin(patch_image,3)
            #print(patch_image.shape)
            x2 = smaller_dim - patch_x
            patch_image2 = image[x2-patch_size:x2,patch_y:patch_y+patch_size]
            #patch_image2 = zoomin(patch_image2,3)
            patch_initial = np.array([patch_x+patch_size,patch_y+patch_size])
            iv_list.append(patch_image)
            im_list.append(patch_image2)
        im_list = im_list[1:n]
        im_h = cv2.vconcat(iv_list)
        #print(im_h.shape)
        width = patch_size*(n-1)
        #print(width)
        image = cv2.resize(image,(width,width))
        im_v=cv2.hconcat(im_list)
        #print(im_v.shape)
        im_v = cv2.vconcat([image,im_v])
        #print(im_v.shape)
        img = cv2.hconcat([im_v,im_h])
        return img
