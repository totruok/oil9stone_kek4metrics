import tempfile

import tensorflow as tf
import tensorflow.contrib.slim as slim

from rude_carnie.yolodetect import PersonDetectorYOLOTiny
from rude_carnie.guess import AGE_LIST, GENDER_LIST, RESIZE_FINAL, classify_one_multi_crop
from rude_carnie.utils import ImageCoder
from rude_carnie.model import get_checkpoint, inception_v3


class AgeGenderDetector:
    def __init__(self, work_dir, yolo_path, age_path, gender_path):
        self.work_dir = work_dir
        self.face_detect = PersonDetectorYOLOTiny(str(yolo_path), tgtdir=self.work_dir)

        checkpoint_paths = {'age': age_path, 'gender': gender_path}
        self.label_lists = {'age': AGE_LIST, 'gender': GENDER_LIST}
        self.image_inputs = {}
        self.softmax_outputs = {}


        config = tf.ConfigProto(allow_soft_placement=True)

        self.input = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])

        self.session = tf.Session(config=config)

        print("[KEKLOG] Loading models...")

        for model_name in ('age', 'gender'):
            with tf.variable_scope(model_name):
                self.softmax_outputs[model_name] = tf.nn.softmax(inception_v3(len(self.label_lists[model_name]), self.input, 1, False))
                slim.assign_from_checkpoint_fn(
                    model_path=tf.train.latest_checkpoint(checkpoint_paths[model_name]),
                    var_list={k.name[k.name.find('/')+1:k.name.find(':')]: k for k in tf.contrib.framework.get_variables_to_restore(include=[model_name])}
                )(self.session)

        print("[KEKLOG] Models loaded")


    def run(self, image):
        face_files, rectangles = self.face_detect.run_img(image)
        print(face_files, rectangles)

        result = {}

        for class_type in ('age', 'gender'):
            for image_file in face_files:
                result[class_type] = classify_one_multi_crop(
                    self.session,
                    self.label_lists[class_type],
                    self.softmax_outputs[class_type],
                    ImageCoder(),
                    self.input,
                    image_file,
                    writer=None
                )

        return result


if __name__ == '__main__':
    from pathlib import Path
    import skimage.io
    age_gender_detector = AgeGenderDetector(
        work_dir=tempfile.mkdtemp(prefix='age-gender-'),
        yolo_path=Path('../models/YOLO_tiny.ckpt').expanduser(),
        age_path=Path('../models/age').expanduser(),
        gender_path=Path('../models/gender').expanduser()
    )

    image = skimage.io.imread(Path('../images/grisha.jpg').expanduser())
    print(age_gender_detector.run(image))
