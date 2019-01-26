import tempfile

import tensorflow as tf

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
        self.sessions = {class_type: tf.Session(config=config) for class_type in ('age', 'gender')}

        for class_type in ('age', 'gender'):
            with self.sessions[class_type]:
                self.image_inputs[class_type] = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits = inception_v3(len(self.label_lists[class_type]), self.image_inputs[class_type], 1, False)

                model_checkpoint_path, global_step = get_checkpoint(
                    checkpoint_paths[class_type], None, 'checkpoint')

                saver = tf.train.Saver()
                saver.restore(self.sessions[class_type], model_checkpoint_path)

                self.softmax_outputs[class_type] = tf.nn.softmax(logits)

    def run(self, image):
        face_files, rectangles = self.face_detect.run_img(image)
        print(face_files, rectangles)

        result = {}

        for class_type in ('age', 'gender'):
            with self.sessions[class_type]:
                for image_file in face_files:
                    result[class_type] = classify_one_multi_crop(
                        self.sessions[class_type],
                        self.label_lists[class_type],
                        self.softmax_outputs[class_type],
                        ImageCoder(),
                        self.image_inputs[class_type],
                        image_file,
                        writer=None
                    )

        return result


if __name__ == '__main__':
    from pathlib import Path
    import skimage.io
    age_gender_detector = AgeGenderDetector(
        work_dir=tempfile.mkdtemp(prefix='age-gender-'),
        yolo_path=Path('~/path/to/YOLO_tiny.ckpt').expanduser(),
        age_path=Path('~/path/to/inception-age-22801/').expanduser(),
        gender_path=Path('~/path/to/inception-gender-21936/').expanduser()
    )

    image = skimage.io.imread(Path('~/path/to/somepic.jpg').expanduser())
    print(age_gender_detector.run(image))
