from design_bench.registration import registry, register, make, spec
from design_bench.task import Task
import requests
import zipfile
import os


def maybe_download(fid,
                   destination,
                   unzip=True):
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
        save_response(response, destination)
        if destination.endswith('.zip') and unzip:
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(destination))


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value


def save_response(response, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data')


register(
    'Quadratic-v0',
    'design_bench.tasks.quadratic:QuadraticTask',
    kwargs=dict(
        global_optimum=(-1.0, 4.0, 2.0, -3.0, 5.0, 1.0,),
        oracle_noise_std=0.2,
        dataset_size=100,
        percentile=80))
register(
    'GFP-v0',
    'design_bench.tasks.gfp:GFPTask')
register(
    'Superconductor-v0',
    'design_bench.tasks.superconductor:SuperconductorTask')

register(
    'MoleculeActivity688150-v0',
    'design_bench.tasks.molecule_activity_v0:MoleculeActivityV0Task',
    kwargs=dict(target_assay=688150,
                split_percentile=80))
register(
    'MoleculeActivity600886-v0',
    'design_bench.tasks.molecule_activity_v0:MoleculeActivityV0Task',
    kwargs=dict(target_assay=600886,
                split_percentile=80))
register(
    'MoleculeActivity600885-v0',
    'design_bench.tasks.molecule_activity_v0:MoleculeActivityV0Task',
    kwargs=dict(target_assay=600885,
                split_percentile=80))
register(
    'MoleculeActivity688537-v0',
    'design_bench.tasks.molecule_activity_v0:MoleculeActivityV0Task',
    kwargs=dict(target_assay=688537,
                split_percentile=80))
register(
    'MoleculeActivity688597-v0',
    'design_bench.tasks.molecule_activity_v0:MoleculeActivityV0Task',
    kwargs=dict(target_assay=688597,
                split_percentile=80))

register(
    'MoleculeActivity-v1',
    'design_bench.tasks.molecule_activity:MoleculeActivityTask')
register(
    'HopperController-v0',
    'design_bench.tasks.controller_v0:ControllerV0Task',
    kwargs=dict(
        obs_dim=11,
        action_dim=3,
        hidden_dim=64,
        env_name='Hopper-v2',
        x_file='hopper_controller_v0_X.npy',
        y_file='hopper_controller_v0_y.npy'))
register(
    'HopperController-v1',
    'design_bench.tasks.controller_v1:ControllerV1Task',
    kwargs=dict(
        obs_dim=11,
        action_dim=3,
        hidden_dim=256,
        env_name='Hopper-v2',
        x_file='hopper_controller_v1_X.npy',
        y_file='hopper_controller_v1_y.npy'))
register(
    'AntMorphology-v0',
    'design_bench.tasks.morphology_v0:MorphologyV0Task',
    kwargs=dict(
        num_parallel=1,
        num_gpus=1,
        n_envs=4,
        max_episode_steps=500,
        total_timesteps=1000000,
        domain='ant'))
register(
    'DogMorphology-v0',
    'design_bench.tasks.morphology_v0:MorphologyV0Task',
    kwargs=dict(
        num_parallel=1,
        num_gpus=1,
        n_envs=4,
        max_episode_steps=500,
        total_timesteps=1000000,
        domain='dog'))
