import pdb
import pandas as pd
import skimage.io as io
import skimage.transform as transform
import os
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow as tf
#import glob
#import os
#import numpy as np
#from PolygonModel import PolygonModel
#from EvalNet import EvalNet
#from GGNNPolyModel import GGNNPolygonModel
#import utils
#import skimage.io as io
#import tqdm
#import json

_BATCH_SIZE = 1
_FIRST_TOP_K = 5

def rect_to_box(xyxy):
    lx, ty, rx, by = xyxy
    center = (np.mean([lx, rx]), np.mean([ty, by]))
    longer_half = max( rx - lx, by - ty ) / 2.0
    square_xyxy = [int(round(x)) for x in [ center[0] - longer_half, center[1] - longer_half,
                    center[0] + longer_half, center[1] + longer_half ] ]
    return square_xyxy


class PolygonRefiner():
    def __init__(self):
        self.last_image = None
        self.last_image_name = None
        self.index = 0
        self.output_folder = "/home/local/KHQ/david.russell/dev/polyrnn-pp/imgs"
        # TODO add a bunch of selfs as needed
        return
        self.evalGraph = tf.Graph()
        self.polyGraph = tf.Graph()
        # Evaluator Network
        tf.logging.info("Building EvalNet...")
        with self.evalGraph.as_default():
            with tf.variable_scope("discriminator_network"):
                evaluator = EvalNet(_BATCH_SIZE)
                evaluator.build_graph()
            saver = tf.train.Saver()

            # Start session
            evalSess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True
            ), graph=self.evalGraph)
            saver.restore(evalSess, FLAGS.EvalNet_checkpoint)

        # PolygonRNN++
        tf.logging.info("Building PolygonRNN++ ...")
        self.model = PolygonModel(FLAGS.PolyRNN_metagraph, polyGraph)

        self.model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))

        polySess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=polyGraph)

        self.model.saver.restore(polySess, FLAGS.PolyRNN_checkpoint)

        #if FLAGS.Use_ggnn:
        #    ggnnGraph = tf.Graph()
        #    tf.logging.info("Building GGNN ...")
        #    ggnnModel = GGNNPolygonModel(FLAGS.GGNN_metagraph, ggnnGraph)
        #    ggnnSess = tf.Session(config=tf.ConfigProto(
        #        allow_soft_placement=True
        #    ), graph=ggnnGraph)

        #    ggnnModel.saver.restore(ggnnSess, FLAGS.GGNN_checkpoint)

    def refine(self, image_file, corners):
        # Go through each of the detection and chip out the image
        # Then pass it through the inference network
        # Grab the mask and update the detection, a la the burnout thing
        if self.last_image_name == image_file:
            image_np = self.last_image
        else:
            image_np = io.imread(image_file)
            self.last_image_name = image_file
            self.last_image = image_np.copy()
        # Creating the graphs
        lx, ty, rx, by = rect_to_box(corners)
        image_np = image_np[ty:by, lx:rx]
        if image_np.size == 0:
            return
        image_np = transform.resize(image_np, (224, 224))
        output_file = os.path.join(self.output_folder, "{}_{:06d}.png".format(os.path.basename(image_file), self.index))
        io.imsave(output_file, image_np)
        self.index += 1
        return

        image_np = np.expand_dims(image_np, axis=0)
        preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

        # sort predictions based on the eval score and pick the best
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)[0]

        if FLAGS.Use_ggnn:
            polys = np.copy(preds['polys'][0])
            feature_indexs, poly, mask = utils.preprocess_ggnn_input(polys)
            preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
            output = {'polys': preds['polys'], 'polys_ggnn': preds_gnn['polys_ggnn']}
        else:
            output = {'polys': preds['polys']}

        # dumping to json files
        save_to_json(crop_path, output)

    def process_file(self, annotation_file, image_basename, output_folder):
        df = pd.read_csv(annotation_file, names=range(16), skiprows=2)
        corners = df.iloc[:, 3:7]
        filenames = df.iloc[:, 1]
        print(corners)
        for (corner, filename) in zip(corners.iterrows(), filenames):
            thing = self.refine(os.path.join(image_basename, filename),
                            corner[1])

PolygonRefiner().process_file(
    "/home/local/KHQ/david.russell/experiments/NOAA/sealion_pixel/output/utility_add_segmentations_watershed_2015.csv",
    "/home/local/KHQ/david.russell/experiments/NOAA/sealion_pixel/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION/2015/images",
    "")