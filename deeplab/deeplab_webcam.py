import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import get_dataset_colormap


sys.path.append(os.path.abspath(".."))
from Spout_for_Python.Library.Spout import Spout

_MODEL_URLS = {
    'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}

_TARBALL_NAME = 'deeplab_model.tar.gz'
model_url = _MODEL_URLS['xception_coco_voctrainaug']

model_dir = "model"
tf.io.gfile.makedirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
if not os.path.isfile(download_path):
    print('downloading model to %s, this might take a while...' % download_path)
    urllib.request.urlretrieve(model_url, download_path)
    print('download completed!')

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

model = DeepLabModel(download_path)

## Webcam demo
cap = cv2.VideoCapture(0)

# Next line may need adjusting depending on webcam resolution
final = np.zeros((1, 384, 1026, 3))

spout = Spout(silent = False)
spout.createSender('output')

while True:
    ret, frame = cap.read()

    # From cv2 to PIL
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    # Run model
    resized_im, seg_map = model.run(pil_im)

    # Adjust color of mask
    seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)


    cv2.imshow("segmentation", seg_image)

    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    ret2, img_otsu = cv2.threshold(seg_image, 0, 1, cv2.THRESH_OTSU)
    img_otsu = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2RGB)
    mask_img = Image.fromarray(img_otsu).resize((384, 513))
    mask = np.array(mask_img)

    im = pil_im.resize((384, 513))
    im = np.array(im)
    sendImg = im * mask

    spout.check()
    spout.send(sendImg)
    """
    if cv2.waitKey(16) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    """
