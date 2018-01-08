import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm
from glob import glob


# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell. 
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they 
                          make things slower because the feature dimenionality 
                          increases and more importantly the step size of the 
                          classifier decreases at test time.
    RET:
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################

    random.seed()
    im_paths = glob(os.path.join(non_face_scn_path, '*jpg'))

    N = len(im_paths)
    D = int((feature_params['template_size']/feature_params['hog_cell_size'])**2 * 31)
    window = feature_params['template_size']
    features_neg = np.zeros([num_samples, D])
    im_pyramid = []

    # image pyramid
    print('Get negative images pyramid...')
    for idx in tqdm(range(N)):
        im = imread(im_paths[idx], as_grey=True)
        for (_, resized) in enumerate(pyramid_gaussian(im, downscale=feature_params['scale'])):
            if resized.shape[0] < window or resized.shape[1] < window:
                break
            im_pyramid.append(resized)

    # crop image features
    print('Get negative features...')
    for idx in tqdm(range(num_samples)):
        select = random.randint(0, len(im_pyramid)-1)
        im = im_pyramid[select].copy()
        h, w = im.shape
        crop_h = random.randint(0, h - window)
        crop_w = random.randint(0, w - window)
        features_neg[idx, :] = np.reshape(hog(im[crop_h:crop_h+window, crop_w:crop_w+window], feature_params['hog_cell_size']), D)
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg

