import os
import argparse
import tempfile
import shutil

import boto3

usage_str = 'usage: %(prog)s [options]'
description_str = 'Upload DICOMs to S3'

def s3_upload_zip(dirpath, bucket_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Creating DICOM zip file...')
        zname = dirpath.split('/')[-1]
        fpath_zip = os.path.join(tmpdir, zname)

        shutil.make_archive(
            fpath_zip, 'zip',
            root_dir='/'.join(dirpath.split('/')[:-1]),
            base_dir=dirpath.split('/')[-1]
        )

        print('Uploading to S3 bucket...')
        s3_client = boto3.client('s3')
        s3_client.upload_file('{}.zip'.format(fpath_zip), bucket_name, '{}.zip'.format(zname))
        print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dicom_path', action='store', dest='dicom_path', type=str, help='Path to DICOM folder', default=None)
    parser.add_argument('--bucket_name', action='store', dest='bucket_name', type=str, help='Name of the S3 bucket', default='subtlegad')

    args = parser.parse_args()

    s3_upload_zip(args.dicom_path, args.bucket_name)
