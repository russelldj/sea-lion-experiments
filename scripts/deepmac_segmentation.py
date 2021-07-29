import glob
import io
import logging
import os
import random
import warnings
import pdb

os.environ["KWIMAGE_DISABLE_C_EXTENSIONS"] = "1"
import kwcoco
import kwimage
import imageio
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
#from object_detection.utils import colab_utils
from object_detection.utils import ops
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
from six import BytesIO
from skimage import color
from skimage import transform
from skimage import util
from skimage.color import rgb_colors
import tensorflow as tf

COLORS = ([rgb_colors.cyan, rgb_colors.orange, rgb_colors.pink,
           rgb_colors.purple, rgb_colors.limegreen , rgb_colors.crimson] +
          [(color) for (name, color) in color.color_dict.items()])
random.shuffle(COLORS)

logging.disable(logging.WARNING)


def read_image(path):
  """Read an image and optionally resize it for better plotting."""
  with tf.io.gfile.GFile(path, 'rb') as f:
    img = Image.open(f)
    return np.array(img, dtype=np.uint8)


def resize_for_display(image, max_height=600):
  height, width, _ = image.shape
  width = int(width * max_height / height)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return util.img_as_ubyte(transform.resize(image, (height, width)))


def get_mask_prediction_function(model):
  """Get single image mask prediction function using a model."""

  @tf.function
  def predict_masks(image, boxes):
    height, width, _ = image.shape.as_list()
    batch = image[tf.newaxis]
    boxes = boxes[tf.newaxis]

    detections = model(batch, boxes)
    masks = detections['detection_masks']

    return ops.reframe_box_masks_to_image_masks(masks[0], boxes[0],
                                                height, width)

  return predict_masks


def plot_image_annotations(image, boxes, masks, darken_image=0.5):
  fig, ax = plt.subplots(figsize=(16, 12))
  ax.set_axis_off()
  image = (image * darken_image).astype(np.uint8)
  ax.imshow(image)

  height, width, _ = image.shape

  num_colors = len(COLORS)
  color_index = 0

  for box, mask in zip(boxes, masks):
    ymin, xmin, ymax, xmax = box
    ymin *= height
    ymax *= height
    xmin *= width
    xmax *= width

    color = COLORS[color_index]
    color = np.array(color)
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2.5, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    mask = (mask > 0.5).astype(np.float32)
    color_image = np.ones_like(image) * color[np.newaxis, np.newaxis, :]
    color_and_mask = np.concatenate(
        [color_image, mask[:, :, np.newaxis]], axis=2)

    ax.imshow(color_and_mask, alpha=0.5)

    color_index = (color_index + 1) % num_colors

  return ax

def infer(prediction_function, annotation_path, output_filename, chunk_size=20):
    coco_dset = kwcoco.CocoDataset.coerce(annotation_path)

    for img in coco_dset.index.imgs.values():
        gid = img['id']
        aids = coco_dset.index.gid_to_aids[gid]
        # TODO make this actually generalizable
        image = read_image(os.path.join(coco_dset.bundle_dpath, img["file_name"]))
        tensor_image = tf.convert_to_tensor(image)
        #image = read_image("/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION/2016/images/20160623_SSLC0073_C.jpg")
        im_height, im_width, _ = image.shape

        locs = []
        aids_list = []
        for aid in aids:
            aids_list.append(aid)
            ann = coco_dset.index.anns[aid]
            x, y, w, h = ann["bbox"]
            normed_x, normed_w = [p / im_width for p in (x, w)]
            normed_y, normed_h = [p / im_height for p in (y, h)]
            locs.append([normed_y, normed_x,  normed_y + normed_h, normed_x + normed_w])
        boxes = np.asarray(locs)
        boxes = np.clip(boxes, 0, 1)

        aids_index = 0
        for i in range(0, boxes.shape[0], chunk_size):
            # Only so many boxes can be processed at ones
            subsetted_boxes = boxes[i:i+chunk_size]
            #import xdev
            #xdev.embed()
            tensor_box = tf.convert_to_tensor(subsetted_boxes, dtype=tf.float32)
            masks = prediction_function(tensor_image, tensor_box)

            for mask in masks:
                c_mask = (mask.numpy() > 0.5).astype(np.uint8)
                kwimage_mask = kwimage.Mask.coerce(c_mask)
                poly = kwimage_mask.to_multi_polygon()
                #poly = kwimage.structs.MultiPolygon([])

                ann = coco_dset.index.anns[aids_list[aids_index]]
                # Check whether this is relative to the box or the world
                ann["segmentation"] = poly.to_coco(style="new")
                aids_index += 1
        # Remove this
    coco_dset.dump(output_filename)

model_path = 'models/research/object_detection/test_data/deepmac_1024x1024_coco17/saved_model'
print('Loading SavedModel')
model = tf.keras.models.load_model(model_path)
prediction_function = get_mask_prediction_function(model)
print("Prediction function generated")

FOLDER = "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION"
files = glob.glob(os.path.join(FOLDER, "*"))
files = sorted(filter(os.path.isdir, files))
print(files)
for f in files:
    basefolder = os.path.split(f)[-1]
    pattern = f"sealions_{basefolder}_v[-1-9].kwcoco.json"
    json_file = os.path.join(f, pattern)
    print(json_file)

    annotation_path = glob.glob(json_file)[-1]

    save_path = os.path.basename(annotation_path)
    save_path = os.path.join("output", save_path)
    infer(prediction_function, annotation_path, output_filename=save_path, chunk_size=20)
