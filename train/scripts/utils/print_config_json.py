import sys
import json

IGNORE_KEYS = ['gpu', 'keras_memory', 'use_multiprocessing', 'num_workers', 'max_queue_size']

if __name__ == '__main__':
    conf_dict = json.load(open(sys.argv[1], 'r'))[sys.argv[2]]

    for k in IGNORE_KEYS:
        if k in conf_dict:
            del conf_dict[k]

    print(json.dumps(conf_dict, sort_keys=True))
