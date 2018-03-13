import os
from urllib.request import urlopen
from zipfile import ZipFile

def download_datasets(args):
    model_net10_url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    model_net40_url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'

    net10_data_destination = args.ModelNet10_dir
    net40_data_destination = args.ModelNet40_dir
    print('[*] Downloading and unzipping datasets.')

    unzip_files(model_net10_url, net10_data_destination)
    unzip_files(model_net40_url, net40_data_destination)


def unzip_files(url, destination):
    zip_resp = urlopen(url)
    temp_zip = open('/tmp/tempfile.zip', 'wb')
    temp_zip.write(zip_resp.read())
    temp_zip.close()
    zf = ZipFile('/tmp/tempfile.zip')
    zf.extractall(path=destination)
    zf.close()