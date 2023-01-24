from time import time
import os
import boto3
import subprocess


class S3:
    def __init__(self, logger):
        self.logger = logger
        self.s3_client = boto3.client('s3')

    def download_files_in_shell(self, s3_path, local_path):
        cmd = f'aws s3 cp {s3_path} {local_path} --rec'
        self.logger.info(f'downloading from s3. cmd: {cmd}')
        subprocess.run(cmd.split())

    def download_files(self, local_key, s3_bucket, s3_key=None, keys=None):
        start_time = time()

        if not (s3_key or keys):
            raise ValueError('either s3_key or keys must be specified')

        if s3_key:
            self.logger.info('listing files in: {bucket}/{key}'.format(bucket=s3_bucket, key=s3_key))
            keys = []
            next_token = ''
            base_kwargs = {'Bucket': s3_bucket, 'Prefix': s3_key}
            while next_token is not None:
                kwargs = base_kwargs.copy()
                if next_token != '':
                    kwargs.update({'ContinuationToken': next_token})
                results = self.s3_client.list_objects_v2(**kwargs)
                contents = results.get('Contents')
                for i in contents:
                    if i.get('Size') > 0:
                        k = i.get('Key')
                        keys.append(k)
                next_token = results.get('NextContinuationToken')
        self.logger.info('total number of files: {}'.format(len(keys)))

        self.logger.info('starting to download files')
        total_size = 0
        for k in keys:
            local_file_path = os.path.join(local_key, k.split('/')[-1])
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            self.logger.info('calculating size for key: s3://{}/{}'.format(s3_bucket, k))
            k_size = self.s3_client.head_object(Bucket=s3_bucket, Key=k)['ContentLength'] / 1024 / 1024 / 1024
            total_size += k_size
            self.logger.info('key size (GB): {}, downloading to {}'.format(k_size, local_file_path))
            self.s3_client.download_file(s3_bucket, k, local_file_path)
        self.logger.info('finished downloading files')

        total_time = (time() - start_time) / 60 / 60
        self.logger.info('total listing and download time: {}'.format(total_time))
        self.logger.info('total data size: {}'.format(total_size))

    def download_into_single_file(self, s3_bucket, s3_key, local_key):
        start_time = time()

        self.logger.info('starting to download files from: {bucket}/{key}'.format(bucket=s3_bucket, key=s3_key))
        s3_object_keys = [d.get('Key') for d in self.list_objects(Bucket=s3_bucket, Prefix=s3_key)]
        with open(local_key, 'ab') as data:
            for key in s3_object_keys:
                self.logger.info(key)
                self.s3_client.download_fileobj(s3_bucket, key, data)
        self.logger.info('finished downloading files')

        total_size = os.path.getsize(local_key) / 1024 / 1024 / 1024
        total_time = (time() - start_time) / 60 / 60
        self.logger.info('total listing and download time: {}'.format(total_time))
        self.logger.info('total data size: {}'.format(total_size))

        return total_size

    def upload_files(self, s3_bucket, files, s3_key):
        self.logger.info('starting to upload files to: {}/{}'.format(s3_bucket, s3_key))
        for file in files:
            self.logger.info('uploading file: {file}'.format(file=file))
            full_key = os.path.join(s3_key, file.split("/")[-1])
            self.logger.debug('uploading {} to s3://{}/{}'.format(file, s3_bucket, full_key))
            self.s3_client.upload_file(file, s3_bucket, full_key)
        self.logger.info('finished uploading')

    def upload_object(self, s3_bucket, s3_key, obj):
        self.logger.info('Start uploading object to: s3://{bucket}/{key}'.format(bucket=s3_bucket, key=s3_key))
        self.s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=obj)
        self.logger.info('Finished uploading')

    def list_objects(self, **base_kwargs):
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = self.s3_client.list_objects_v2(**list_kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')
