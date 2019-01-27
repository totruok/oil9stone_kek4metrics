import base64
import logging
import tempfile
from pathlib import Path

import skimage.io
from flask import Flask, request, redirect, url_for, abort, render_template, send_from_directory

from age_gender import AgeGenderDetector

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)-15s] %(levelname)s: %(message)s'
)

app = Flask(__name__)
storage_dir = Path(tempfile.mkdtemp(prefix='photokek-'))
logging.debug('Data directory is at {storage_dir}'.format(storage_dir=storage_dir))
age_gender_detector = AgeGenderDetector(
    storage_dir,
    yolo_path=Path('../models/YOLO_tiny.ckpt'),
    age_path=Path('../models/age'),
    gender_path=Path('../models/gender')
)


def process(image):
    age_gender = age_gender_detector.run(image)
    logging.debug('Computed age & gender: {}'.format(age_gender))
    # Some magic happens here
    return image, 'funny text'


def hash_image(image):
    return str(hash(image.data.tobytes()) % 0xFFFFFFFFFFFFFFFF)


@app.route('/result', methods=['GET'])
def result():
    key = request.args['key']
    result_dir = storage_dir / key

    if not result_dir.exists():
        logging.warning('Key {key} not found in {storage_dir}'.format(
            key=key,
            storage_dir=storage_dir
        ))
        abort(404)

    logging.info('Serving key {key} from {storage_dir}'.format(
        key=key, storage_dir=storage_dir))
    with open(result_dir / 'image.png', 'rb') as f:
        im_b64 = base64.b64encode(f.read()).decode('ascii')
    with open(result_dir / 'text.txt', 'r') as f:
        im_text = f.read()

    return '''
    <!doctype html>
    <title>Result</title>
    <h1>Result</h1>
    <img src="data:image/png;base64,{im_b64}" width="500">
    <p>{im_text}</p>
    '''.format(im_b64=im_b64, im_text=im_text)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect('root')
    in_file = request.files['image']
    logging.info(
        'Received file: filename={filename!r}, mimetype={mimetype!r}'.format(
            filename=in_file.filename,
            mimetype=in_file.mimetype
        )
    )
    in_image = skimage.io.imread(in_file.stream)
    logging.debug(
        'Image shape is {shape}'.format(shape=in_image.shape)
    )
    key = hash_image(in_image)
    out_dir = storage_dir / key
    if out_dir.exists():
        logging.info('Processed version of {filename!r} is cached in {out_dir}'.format(
            filename=in_file.filename,
            out_dir=out_dir
        ))
    else:
        logging.info('Processing {filename!r}, key={key}'.format(
            filename=in_file.filename,
            key=key
        ))
        out_dir.mkdir(parents=True)
        out_image, out_text = process(in_image)
        logging.info('Saving processed {filename!r} to {out_dir}'.format(
            filename=in_file.filename,
            out_dir=out_dir
        ))
        skimage.io.imsave(out_dir / 'image.png', out_image)
        with (out_dir / 'text.txt').open('w') as f:
            f.write(out_text)
    return redirect(url_for('result', key=key))


@app.route('/')
def root():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()#debug=True)
