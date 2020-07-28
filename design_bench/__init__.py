from design_bench.registration import registry, register, make, spec
from design_bench.task import Task
import numpy as np
import requests
import os


def maybe_download(fid,
                   destination):
    """If a file does not already exist then download it from
    google drive using a custom GET request

    Args:

    fid: str
        the file id specified by google which is the 'X' in the url:
        https://drive.google.com/file/d/X/view?usp=sharing
    destination: str
        the destination for the file on this device; if the file
        already exists it will not be downloaded again
    """

    if not os.path.exists(destination):
        session = requests.Session()
        response = session.get(
            "https://docs.google.com/uc?export=download",
            params={'id': fid}, stream=True)
        token = get_confirm_token(response)
        if token is not None:
            response = session.get(
                "https://docs.google.com/uc?export=download",
                params={'id': fid, 'confirm': token}, stream=True)
        save_response_content(
            response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value


def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data')


register(
    'GP1D-v0',
    'design_bench.tasks.gp:GP1DTask',
    kwargs=dict(
        dataset_size=100,
        upper_bound=(4.0,),
        lower_bound=(-4.0,),
        noise=0.2))
register(
    'GP2D-v0',
    'design_bench.tasks.gp:GP2DTask',
    kwargs=dict(
        dataset_size=100,
        upper_bound=(0.0, 15.0),
        lower_bound=(-5.0, 10.0)))


register(
    'GFP-v0',
    'design_bench.tasks.gfp:GFPTask')
maybe_download('1UO8L3uOp141m2v5dVlpGZ4tZ42XIJ4Vq',
               os.path.join(DATA_DIR, 'gfp_gt_evals.npy'))
maybe_download('1DeOoYQs5GEis3jIYsbGxuemjtsBiUSJm',
               os.path.join(DATA_DIR, 'gfp_gpy.npy'))
maybe_download('10xMOWXZjGOKLokO4jP6ya29-ZD2tb46X',
               os.path.join(DATA_DIR, 'gfp_gpX.npy'))
maybe_download('18EvOK25vmPvRGNbviv1Oep2CPXt3UrLt',
               os.path.join(DATA_DIR, 'gfp_gpparams.npy'))
maybe_download('1ySC8Rkfye6JfRKqoDS_KAXqUQTKtrbvZ',
               os.path.join(DATA_DIR, 'gfp_gpKinv.npy'))
maybe_download('1tRvY0W4ygoPxytdhAWZuwSQmvNj2QEtK',
               os.path.join(DATA_DIR, 'gfp_gpK.npy'))
maybe_download('1_jcPkQ-M1FRhkEONoE57WEbp_Rivkho2',
               os.path.join(DATA_DIR, 'gfp_data.csv'))


register(
    'HopperController-v0',
    'design_bench.tasks.controller:ControllerTask',
    kwargs=dict(
        obs_dim=11,
        action_dim=3,
        hidden_dim=64,
        env_name='Hopper-v2',
        x_file='data/hopper_controller_X.txt',
        y_file='data/hopper_controller_y.txt'))
maybe_download('18yuyw8xSQa7ydIQN-_ajHPZXh1VQbT6o',
               os.path.join(DATA_DIR, 'hopper_controller_X.txt'))
maybe_download('1nX0LObb4OWJQfcBbQxq2rdqiujUOF79y',
               os.path.join(DATA_DIR, 'hopper_controller_y.txt'))
