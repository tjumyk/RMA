import argparse

from process import conn


def copy_list(from_key, to_key):
    data = conn.lrange(from_key, 0, -1)
    return conn.rpush(to_key, *data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy a feature from one key to another')
    parser.add_argument('data_source', help='Data source')
    parser.add_argument('data_type', help='Data type')
    parser.add_argument('from_key', help='From which key to copy')
    parser.add_argument('to_key', help='To which key to copy')
    args = parser.parse_args()

    num_records = copy_list('result:::::%s:::::%s:::::%s' % (args.from_key, args.data_source, args.data_type),
                            'result:::::%s:::::%s:::::%s' % (args.to_key, args.data_source, args.data_type))
    print('%d records copied' % num_records)
