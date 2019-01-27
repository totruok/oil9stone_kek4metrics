import io
import json
import logging
import random
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
with open('templates.json', 'r') as fp:
    questions = json.load(fp)
AUTH = ("kekfacer", "whatsyourmooddude")
age_gender_detector = AgeGenderDetector(
    storage_dir,
    yolo_path=Path('../models/YOLO_tiny.ckpt'),
    age_path=Path('../models/age'),
    gender_path=Path('../models/gender')
)


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


def send_meme_job(access_key, meme_id, face_swap_method):
    url = "http://home.totruok.ru:40936/push?kind=swap&priority=0"
    params = {
        "face": access_key,
        "face_swap_method": face_swap_method,
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
            if meme_ids[i] == 'pending':
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


def process(image):
    age_gender = age_gender_detector.run(image)
    age_range, age_prob = age_gender['age']
    gender, gender_prob = age_gender['gender']
    logging.debug('Computed age: {} (p={}), gender: {} (p={})'.format(age_range, age_prob, gender, gender_prob))

    image_key = send_file(image)

    qs = sort_questions(age_range, gender, questions)

    random.seed(int(hash_image(image), base=16))
    t_indices = [random.randint(0, len(q['templates']) - 1) for q in qs]

    job_ids = [
        send_meme_job(
            image_key,
            q['templates'][t_idx]['access_key_picture'],
            q['swap_to_use']
        )
        for q, t_idx in zip(qs, t_indices)
    ]
    logging.debug('Sent job_ids: {}, polling memes'.format(job_ids))
    meme_ids = wait_for_jobs(job_ids)
    logging.debug('Retrieved meme_ids: {}'.format(meme_ids))

    return {
        'age': {
            'value': age_range,
            'prob': float(age_prob)
        },
        'gender': {
            'value': gender,
            'prob': float(gender_prob)
        },
        'templates': [
            {
                'meme_id': meme_id,
                'name': q['name'],
                'text_on_picture': q['templates'][t_idx]['text_on_picture'],
                'text': q['templates'][t_idx]['text']
            }
            for meme_id, t_idx, q
            in zip(meme_ids, t_indices, qs)
            if meme_id is not None
        ]
    }


def hash_image(image):
    return '{:016x}'.format(hash(image.data.tobytes()) % 0xFFFFFFFFFFFFFFFF)


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

    with (result_dir / 'data.json').open('r') as fp:
        data = json.load(fp)

    age_from, age_to = data['age']['value'][1:-1].split(', ')

    return render_template(
        'result.html',
        templates=data['templates'],
        gender={'M': 'мужчина', 'F': 'женщина'}[data['gender']['value']],
        gender_prob='{:.1f}%'.format(data['gender']['prob'] * 100),
        age_from=age_from,
        age_to=age_to,
        age_prob='{:.1f}%'.format(data['age']['prob'] * 100),
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
        data = process(in_image)
        logging.info('Saving processed {filename!r} to {out_dir}'.format(
            filename=in_file.filename,
            out_dir=out_dir
        ))
        skimage.io.imsave(out_dir / 'image.png', in_image)
        with (out_dir / 'data.json').open('w') as fp:
            json.dump(data, fp)
    return redirect(url_for('result', key=key))


@app.route('/')
def root():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()  # debug=True)
