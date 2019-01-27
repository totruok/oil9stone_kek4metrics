import io
import json
import logging
import tempfile
import time
from pathlib import Path

import requests
import skimage.io
from flask import Flask, request, redirect, url_for, abort, render_template, send_from_directory

from age_gender import AgeGenderDetector
from rang import sort_questions

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
AUTH = ("kekfacer", "whatsyourmooddude")
with open('templates.json', 'r') as fp:
    questions = json.load(fp)


def send_file(image):
    url = "http://home.totruok.ru:40936/upload"

    png = io.BytesIO()
    skimage.io.imsave(png, image, format_str='png')
    png.seek(0)

    files = {'image': png}
    return requests.post(
        url,
        files=files,
        auth=AUTH
    ).json()["access_key"]


def send_meme_job(access_key, meme_id):
    url = "http://home.totruok.ru:40936/push?kind=swap&priority=0"
    params = {
        "face": access_key,
        "meme": meme_id
    }

    return requests.post(
        url,
        data={"params": json.dumps(params)},
        auth=AUTH
    ).json()['jobid']


def wait_for_jobs(job_ids):
    url = "http://home.totruok.ru:40936/get_result"
    meme_ids = ['pending'] * len(job_ids)
    while any((meme_id == 'pending') for meme_id in meme_ids):
        for i in range(len(job_ids)):
            if meme_ids[i] is None:
                data = requests.get(
                    url,
                    params={"jobid": job_ids[i]},
                    auth=AUTH
                ).json()
                if 'result' in data:
                    meme_ids[i] = data['result']
                elif 'error' in data:
                    meme_ids[i] = None
        time.sleep(1)
    return meme_ids
    # pic is at http://home.totruok.ru:40925/{meme_id}


def process(image):
    age_gender = age_gender_detector.run(image)
    age_range, age_prob = age_gender['age']
    gender, gender_prob = age_gender['gender']
    logging.debug('Computed age: {} (p={}), gender: {} (p={})'.format(age_range, age_prob, gender, gender_prob))

    # qs = sort_questions(age_range, gender, questions)
    # q = qs[0]
    # caption = q['name']

    return image, 'funny text'


def hash_image(image):
    return str(hash(image.data.tobytes()) % 0xFFFFFFFFFFFFFFFF)


@app.route('/retrieve/<key>', methods=['GET'])
def retrieve(key):
    result_dir = storage_dir / key

    if not result_dir.exists():
        logging.warning('Key {key} not found in {storage_dir}'.format(
            key=key,
            storage_dir=storage_dir
        ))
        abort(404)

    logging.info('Serving key {key} from {storage_dir}'.format(
        key=key, storage_dir=storage_dir))

    return send_from_directory(result_dir, 'image.png')


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

    with open(result_dir / 'text.txt', 'r') as f:
        im_text = f.read()

    return render_template(
        'template.html',
        pic=key,
        description=im_text,
        pic2='',
        pic3='',
        pic4='',
        pic5='',
    )


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
    app.run()  # debug=True)
