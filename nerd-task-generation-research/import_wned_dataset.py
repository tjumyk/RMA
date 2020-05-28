import argparse
import json
import os

import psycopg2
from lxml import etree


def import_data_set(root_folder: str, name: str, _db_conn, table_name: str = None):
    if table_name is None:
        table_name = name

    data_folder = os.path.join(root_folder, name)
    if not os.path.exists(data_folder):
        raise IOError('data folder not found')

    annotation_path = os.path.join(data_folder, name + '.xml')
    if not os.path.exists(annotation_path):
        raise IOError('annotation file not found')
    doc_folder = os.path.join(data_folder, 'RawText')
    if not os.path.exists(doc_folder):
        raise IOError('raw text folder not found')

    with open(annotation_path) as f_annotation:
        tree = etree.parse(f_annotation)

    with _db_conn.cursor() as cursor:
        print('Creating table')
        sql = '''create table %s(
                    id          serial primary key,
                    doc_id      varchar(255),
                    surfaceform varchar(255),
                    length      varchar(255),
                    begoffset   varchar(255),
                    annotation  varchar(255) default 'none',
                    type        varchar(255) default 'test',
                    doc_text    text,
                    wiki_url    text
                )''' % table_name
        cursor.execute(sql)

        for doc in tree.getroot():
            doc_name = doc.get('docName')
            print('Importing document: %s' % doc_name)

            with open(os.path.join(doc_folder, doc_name)) as f_doc:
                doc_content = f_doc.read()

            for annotation in doc:
                mention, entity, offset, length, wiki_url = None, None, None, None, None
                for child in annotation:
                    if child.tag == 'mention':
                        mention = child.text
                    elif child.tag == 'wikiName':
                        if child.text:  # can be an empty tag if no annotation
                            entity = child.text.replace(' ', '_')
                    elif child.tag == 'offset':
                        offset = int(child.text)
                    elif child.tag == 'length':
                        length = int(child.text)
                if not entity:
                    print('[Warning] No annotation for mention %s' % mention)
                    entity = 'none'  # enforce default value and keep the same format as the existing tables
                wiki_url = 'https://en.wikipedia.org/wiki/' + entity

                sql = 'insert into %s (doc_id, surfaceform, length, begoffset, annotation, doc_text, wiki_url) ' \
                      'values %s' % (table_name, '(%s, %s, %s, %s, %s, %s, %s)')
                cursor.execute(sql, (doc_name, mention, length, offset, entity, doc_content, wiki_url))

        db_conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import WNED dataset')
    parser.add_argument('root_folder', help='Root folder of the WNED datasets')
    parser.add_argument('name', help='Name of the dataset to import')
    parser.add_argument('table_name', help='Name of the database table')
    args = parser.parse_args()

    with open('config/config.json') as f_config:
        config = json.load(f_config)
    db_config = config['db']
    db_conn = psycopg2.connect('postgres://%s:%s@%s:%s/%s' % (db_config['user'], db_config['password'],
                                                              db_config['host'], db_config['port'],
                                                              db_config['database']))
    import_data_set(args.root_folder, args.name, db_conn, args.table_name)
