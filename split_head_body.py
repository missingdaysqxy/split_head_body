import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import time
import argparse
import numpy as np
from PIL import Image

# FaceModel must be imported before COCODemo, otherwise
# there will be a serious conflict leading to kernel restart
from face_fcn8 import FaceModel
from maskrcnn_benchmark.config import cfg
from rcnn import COCODemo as PersonModel

# Set GPU limit for Keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=tfconfig)
KTF.set_session(session)


def get_persons(image, person_model):
    """return a list of [offset, person_img, person_mask]"""
    predictions = person_model.compute_prediction(image)
    predictions = person_model.select_top_predictions(predictions)
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    bboxes = predictions.bbox
    # print("%d object(s) detected." % len(labels))
    persons = []  # [offset, person_img, person_mask]
    for mask, label, bbox in zip(masks, labels, bboxes):
        if label == 1:
            mask = mask[0, :, :, np.newaxis]
            _, thresh = cv2.threshold(mask, mask.mean(), 255, cv2.THRESH_BINARY)
            x1, y1, x2, y2 = bbox.numpy().astype(np.int32)
            person_img = image[y1:y2 + 1, x1:x2 + 1]
            person_mask = thresh[y1:y2 + 1, x1:x2 + 1]
            persons.append([(x1, y1), person_img, person_mask])
    return persons


def get_face_mask(img, face_model):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = im.shape[0], im.shape[1]
    im = cv2.resize(im, (500, 500))
    im = np.array(im, dtype=np.float32)
    im = im[:, :, ::-1]
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = im[np.newaxis, :]
    # im = im.transpose((2,0,1))
    mask = face_model.predict([im])[0]
    mask = cv2.resize(mask, (w, h)).argmax(axis=-1).astype(np.uint8)
    _, mask = cv2.threshold(mask[:, :, np.newaxis], mask.mean(), 255, cv2.THRESH_BINARY)
    return mask


def morph_open(mask, kernel_size=5):
    """Removing noise by morphological opening operation (erosion followed by dilation)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def bbox(mask):
    """Find the bounding box of a mask"""
    axis0 = np.any(mask, axis=0)
    axis1 = np.any(mask, axis=1)
    x, u = np.where(axis0)[0][[0, -1]]
    y, v = np.where(axis1)[0][[0, -1]]
    return (x, y), (u, v)


def get_person_masks(image, person_model, face_model, noisy_filter_size):
    """
    Get a list of [offset, head_mask, body_mask]).
    masks, i.e. head_mask and body_mask, are both 2-axis array with the same size.
    offset is the position of masks left-top points in image.
    """
    # get persons
    persons = get_persons(image, person_model)
    # print("Stored %d person(s) into list" % len(humans))

    # get head mask
    for elem in persons:
        offset, person_img, person_mask = elem
        h, w = person_mask.shape
        in_img = person_img[:w]
        face_mask = get_face_mask(in_img, face_model)
        if h > w:
            face_mask = np.concatenate([face_mask, np.zeros((h - w, w), np.uint8)])
        elem.append(face_mask)

    # Now persons' elements are [offset, cropped_img, body_mask, face_mask]
    ret_masks = []
    for offset, person_img, person_mask, face_mask in persons:
        denoised_mask = morph_open(face_mask, noisy_filter_size)
        if not denoised_mask.any():
            continue
        (x, y), (u, v) = bbox(denoised_mask)
        h, w = denoised_mask.shape
        left_y = np.where(denoised_mask[:, x])[0][0]
        right_y = np.where(denoised_mask[:, u])[0][0]
        mid_x = (x + u) // 2
        # Take the intersection for Head
        head_mask = denoised_mask.copy()
        head_mask[:left_y, :mid_x] = 255
        head_mask[:right_y, mid_x:] = 255
        head_mask = person_mask & head_mask
        # Take the difference set for Body
        diff_mask = person_mask & (~head_mask)
        body_mask = morph_open(diff_mask, int(1.3 * noisy_filter_size))
        ret_masks.append([offset, head_mask, body_mask])
    return ret_masks


def apply_masks(image, masks, mask_alpha, body_color, head_color):
    """
    Apply masks in image.
    You can overwrite this function for other usages with masks
    """
    mask_layer = np.zeros_like(image)
    for offset, head_mask, body_mask in masks:
        x, y = offset
        h, w = body_mask.shape
        blank = np.zeros((h, w, image.shape[2]), dtype=np.uint8)
        blank[:] = body_color
        m_body = cv2.bitwise_or(blank, blank, mask=body_mask)
        mask_layer[y:y + h, x:x + w] = m_body
        blank[:] = head_color
        m_head = cv2.bitwise_or(blank, blank, mask=head_mask)
        mask_layer[y:y + h, x:x + w] += m_head
    canvas = cv2.addWeighted(mask_layer, mask_alpha, image, 1 - mask_alpha, 0)
    # canvas = cv2.scaleAdd(mask_layer, mask_alpha, image)
    return canvas


def main(args):
    # Parameters
    rcnn_cfg = args.config_file
    face_weight = args.face_weight
    noisy_filter_size = args.noisy_filter_size
    line_color = [255, 255, 0]
    line_thickness = 5
    body_color = [255, 20, 127]
    head_color = [0, 23, 232]
    mask_alpha = args.mask_alpha

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    # body_model must be instantiated before face_model, otherwise there may be a serious conflict leading to kernel restart
    person_model = PersonModel(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    face_model = FaceModel()
    face_model.load_weights(args.face_weight)

    flist = os.listdir(args.input)
    os.makedirs(args.output, exist_ok=True)
    for name in flist:
        if os.path.splitext(name)[1].lower() not in ['.jpg', '.png', '.bmp']:
            continue
        start_time = time.time()
        print("Process with ", name, end=" ")
        image_path = os.path.join(args.input, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = get_person_masks(image, person_model, face_model, noisy_filter_size)
        canvas = apply_masks(image, masks, mask_alpha, body_color, head_color)
        save_path = os.path.join(args.output, name)
        cv2.imwrite(save_path, canvas)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        # cv2.imshow("Masked Image", canvas)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get masks of head and body for persons in a picture.")
    parser.add_argument("input", type=str, help="Directory of input picture(s)")
    parser.add_argument("-o", "--output", type=str,
                        default="./output", help="Output directory of masked pictures")
    parser.add_argument(
        "--config_file",
        default="./rcnn_configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml",
        help="Path of maskrcnn config file",
    )
    parser.add_argument(
        "--face_weight",
        default="./Keras_FCN8s_face_seg_YuvalNirkin.h5",
        help="Path of face_fcn8 weight file",
    )
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.55,
        help="Alpha value of masks"
    )
    parser.add_argument(
        "--noisy_filter_size",
        type=int,
        default=20,
        help="Size of the morphological kernel, the larger the value, the greater the range of filters",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min_image_size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show_mask_heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks_per_dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    # parser.add_argument(
    #     "opts",
    #     help="Modify model config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )

    args = parser.parse_args()

    main(args)
