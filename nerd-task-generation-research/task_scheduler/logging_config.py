""" Logging Configuration """
from __future__ import absolute_import
import logging
import os


def setup(filename='app.log'):
    fmt = '%(asctime)s|%(levelname)s|%(module)s|%(lineno)d|%(message)s'
    # ln = "app.log"
    logging.basicConfig(format=fmt)
    log = logging.getLogger('nerd')
    # filepath = '/Users/dzs/PycharmProjects/NERD-OPENIE/experiment/log'
    # handler = logging.FileHandler(os.path.join(filepath, filename))
    # log.addHandler(handler)
    log.setLevel(logging.INFO)


def getLogger(filename='app.log'):
    setup(filename)
    return logging.getLogger('nerd')





def get_output_file(entity, mention):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(os.path.dirname(dir_path), 'static', 'matching-detail', str(mention) + "-" + str(entity) +'.html')
    file_content = "<script language='JavaScript'> function setVisibility(id, visibility) {document.getElementById(id).style.display = visibility;}</script> "
    file_content += "<h1>" + str(mention) + "<h1>"
    file_content += "<table border='1'><tr>"
    file_content += "<th>Entity</th>" + "<th>Mention Keyphrases</th>" + "<th>Entity Keyphrases</th>" + "<th>Similarity</th>" + "</tr>"
    with open(file_path, 'w') as f:
        f.write(file_content)
    return file_path
