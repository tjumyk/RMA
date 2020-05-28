from flask import Flask
from flask import request
from flask import jsonify
import os
import json
import re
import phrase_extraction
import logging_config

log = logging_config.get_logger()
# print(dirname)
# with open('./config/config.json') as f:
#     pass

app = Flask(__name__)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/hello/<user_name>')
def hello_world(user_name):
    return 'Hello World! %s' % user_name


@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        return request.data


@app.route('/api/getCo/query/<query_id>', methods=['POST'])
def get_tok_phr_co(query_id):
    if request.method == 'POST':
        result = phrase_extraction.phrase_extraction_web_co_redis(query_id)
        log.info('clauses: %s' % result)
        return jsonify(result)


@app.route('/api/getFq/query/<query_id>', methods=['POST'])
def get_tok_phr_fq(query_id):
    if request.method == 'POST':
        result = phrase_extraction.phrase_extraction_web_fq_redis(query_id)
        log.info('clauses: %s' % result)
        return jsonify(result)


@app.route('/api/getIDF/query/<query_id>', methods=['POST'])
def get_tok_phr_idf(query_id):
    if request.method == 'POST':
        result = phrase_extraction.phrase_extraction_web_idf_redis(query_id)
        log.info('clauses: %s' % result)
        return jsonify(result)


@app.route('/api/getETR/query/<query_id>', methods=['POST'])
def get_tok_phr_etr(query_id):
    if request.method == 'POST':
        result = phrase_extraction.phrase_extraction_web_idf_entropy_redis(query_id)
        log.info('clauses: %s' % result)
        return jsonify(result)


if __name__ == '__main__':
    # debug = True
    # handler = RotatingFileHandler('foo.log')
    # handler.setLevel(logging.INFO)
    # app.logger.addHandler(handler)
    # log = nerd.logging.getLogger()
    app.run(host='0.0.0.0', port=8086, threaded=True)
    #app.run()
