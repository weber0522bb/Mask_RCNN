import numpy as np
import os
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from cyvlfeat.hog import hog
from tqdm import tqdm
import sys
sys.path.append("..")

from proj4_2.code.non_max_supr_bbox import non_max_supr_bbox

def run_detector(img, model, feature_params):
    """
    FUNC: This function returns detections on all of the images in a given path.
        You will want to use non-maximum suppression on your detections or your
        performance will be poor (the evaluation counts a duplicate detection as
        wrong). The non-maximum suppression is done on a per-image basis. The
        starter code includes a call to a provided non-max suppression function.
    ARG:
        - test_scn_path: a string; This directory contains images which may or
                        may not have faces in them. This function should work for
                        the MIT+CMU test set but also for any other images.
        - model: the linear classifier model
        - feature_params: a dict; 'template_size': the number of pixels spanned
                        by each train / test template (probably 36).
                        'hog_cell_size': the number of pixels in each HoG cell
                        (default 6).
                        Template size should be evenly divisible by hog_cell_size.
                        Smaller HoG cell sizes tend to work better, but they make
                        things slower because the feature dimensionality increases
                        and more importantly the step size of the classifier
                        decreases at test time.
        - MORE...  You can add additional arguments for advanced work like multi-
                   scale, pixel shift and so on.
                   
    RET:
        - bboxes: (N, 4) ndarray; N is the number of non-overlapping detections, bboxes[i,:] is
                        [x_min, y_min, x_max, y_max] for detection i.
        - confidences: (N, 1) ndarray; confidences[i, :] is the real valued confidence
                        of detection i.
        - image_ids: (N, 1) ndarray;  image_ids[i, :] is the image file name for detection i.
    """
    # The placeholder version of this code will return random bounding boxes in
    # each test image. It will even do non-maximum suppression on the random
    # bounding boxes to give you an example of how to call the function.

    # Your actual code should convert each test image to HoG feature space with
    # a _single_ call to vl_hog for each scale. Then step over the HoG cells,
    # taking groups of cells that are the same size as your learned template,
    # and classifying them. If the classification is above some confidence,
    # keep the detection and then pass all the detections for an image to
    # non-maximum suppression. For your initial debugging, you can operate only
    # at a single scale and you can skip calling non-maximum suppression.

    #test_images = os.listdir(test_scn_path)

    # initialize these as empty and incrementally expand them.
    bboxes = np.zeros([0, 4])
    confidences = np.zeros([0, 1])
    #image_ids = np.zeros([0, 1])
    THRESH = 0.8

    cell_size = feature_params['hog_cell_size']
    cell_num = int(feature_params['template_size'] / feature_params['hog_cell_size'])  # cell number of each template
    D = int((feature_params['template_size'] / feature_params['hog_cell_size']) ** 2 * 31)
    window = feature_params['template_size']
    print('Testing...')
    for i in tqdm(range(1)):

        #########################################
        ##          you code here              ##
        #########################################
        cur_image_ids = np.zeros([0, 1])
        cur_bboxes = np.zeros([0, 4])
        cur_confidences = np.zeros([0, 1])
        ori_img = np.array(color.rgb2gray(img))

        # image pyramid
        for (sc_idx, im_pyramid) in enumerate(pyramid_gaussian(ori_img, downscale=feature_params['scale'])):
            if im_pyramid.shape[0] < window or im_pyramid.shape[1] < window:
                break
            scale =feature_params['scale']**sc_idx

            feats = hog(im_pyramid, cell_size)

            # sliding window
            feats_num = (feats.shape[0]-cell_num+1) * (feats.shape[1]-cell_num+1)
            #print(feats.shape)
            #print(feats_num)
            crop_feats = np.zeros([feats_num, cell_num, cell_num, 31])
            idx = 0
            for y in range(0, feats.shape[0]-cell_num+1):
                for x in range(0, feats.shape[1]-cell_num+1):
                    crop_feats[idx, :, :, :] = feats[y:y+cell_num, x:x+cell_num, :].copy()
                    idx = idx + 1
            crop_feats = np.reshape(crop_feats, [feats_num, D])

            conf = model.predict_proba(crop_feats)
            conf = conf[:, 1]

            conf_idx = np.argwhere(conf >= THRESH)
            if len(conf_idx) != 0:
                py_conf = conf[conf_idx]
                num_box = len(conf_idx)

                py_bboxes = np.zeros([num_box, 4])
                for idx in range(num_box):
                    #cur_image_ids = np.concatenate([cur_image_ids, np.reshape(test_images[i],[1,1])], axis=0)
                    window_num = feats.shape[1]-cell_num+1
                    xstep = conf_idx[idx] % window_num
                    ystep = conf_idx[idx] // window_num

                    py_bboxes[idx, 1] = ystep * cell_size * scale
                    py_bboxes[idx, 0] = xstep * cell_size * scale
                    py_bboxes[idx, 3] = py_bboxes[idx, 1] + window * scale
                    py_bboxes[idx, 2] = py_bboxes[idx, 0] + window * scale

                cur_bboxes = np.concatenate([cur_bboxes, py_bboxes], axis=0)
                cur_confidences = np.concatenate([cur_confidences, py_conf], axis=0)
        a= 20
        print(a,"fuc")
        is_maximum = non_max_supr_bbox(cur_bboxes, cur_confidences, ori_img.shape)

        cur_bboxes = cur_bboxes[is_maximum[:, 0], :]
        cur_confidences = cur_confidences[is_maximum[:, 0], :]
        #cur_image_ids = cur_image_ids[is_maximum[:, 0]]

        bboxes = np.concatenate([bboxes, cur_bboxes], axis=0)
        confidences = np.concatenate([confidences, cur_confidences], axis=0)
        #image_ids = np.concatenate([image_ids, cur_image_ids], axis=0)
        #########################################
        ##          you code here              ##
        #########################################

        # non_max_supr_bbox can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You don't need to modify
        # anything in non_max_supr_bbox, but you can.


    return bboxes, confidences

