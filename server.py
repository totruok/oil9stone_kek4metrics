import base64
import logging
import tempfile
from pathlib import Path

import skimage.io
from flask import Flask, request, redirect, url_for, abort

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)-15s] %(levelname)s: %(message)s'
)

app = Flask(__name__)
storage_dir = Path(tempfile.mkdtemp(prefix='photokek-'))
logging.debug(f'Data directory is at {storage_dir}')


def process(image):
    # Some magic happens here
    return image, 'funny text'


def hash_image(image):
    return str(hash(image.data.tobytes()) % 0xFFFFFFFFFFFFFFFF)


@app.route('/result', methods=['GET'])
def result():
    key = request.args['key']
    result_dir = storage_dir / key

    if not result_dir.exists():
        logging.warning(f'Key {key} not found in {storage_dir}')
        abort(404)

    logging.info(f'Serving key {key} from {storage_dir}')
    with open(result_dir / 'image.png', 'rb') as f:
        im_b64 = base64.b64encode(f.read()).decode('ascii')
    with open(result_dir / 'text.txt', 'r') as f:
        im_text = f.read()

    return f'''
    <!doctype html>
    <title>Result</title>
    <h1>Result</h1>
    <img src="data:image/png;base64,{im_b64}" width="500">
    <p>{im_text}</p>
    '''


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect('root')
    in_file = request.files['image']
    logging.info(
        f'Received file: filename={in_file.filename!r}, mimetype={in_file.mimetype!r}'
    )
    in_image = skimage.io.imread(in_file.stream)
    logging.debug(
        f'Image shape is {in_image.shape}'
    )
    key = hash_image(in_image)
    out_dir = storage_dir / key
    if out_dir.exists():
        logging.info(f'Processed version of {in_file.filename!r} is cached in {out_dir}')
    else:
        logging.info(f'Processing {in_file.filename!r}, key={key}')
        out_dir.mkdir(parents=True)
        out_image, out_text = process(in_image)
        logging.info(f'Saving processed {in_file.filename!r} to {out_dir}')
        skimage.io.imsave(out_dir / 'image.png', out_image)
        with (out_dir / 'text.txt').open('w') as f:
            f.write(out_text)
    return redirect(url_for('result', key=key))


@app.route('/')
def root():
    return '''
    <!doctype html>
    <title>Process new image</title>
    <h1>Process new image</h1>
    <form method=post enctype=multipart/form-data action=upload>
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
