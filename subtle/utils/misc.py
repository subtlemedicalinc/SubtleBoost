import hashlib

def get_timestamp_hash(n=6):
    return hashlib.sha1(str(time.time()).encode()).hexdigest()[:n]

def dict_merge(dct, merge_dct, add_keys=True):
    dct = dct.copy()
    if not add_keys:
        merge_dct = {
            k: merge_dct[k]
            for k in set(dct).intersection(set(merge_dct))
        }

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dct[k] = dict_merge(dct[k], merge_dct[k], add_keys=add_keys)
        else:
            dct[k] = merge_dct[k]

    return dct

def print_progress_bar(iteration, total, fhandle, prefix='', suffix='', decimals=0, length=30, fill='█', print_end='\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('{} |{}| {}% {}\r'.format(prefix, bar, percent, suffix), file=fhandle, end=print_end)
    # print('{}\r\r'.format(suffix), file=fhandle)

    if iteration == total:
        print('', file=fhandle)
