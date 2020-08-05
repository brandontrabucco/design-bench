from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from docker import types
import numpy as np
import os
import requests
import shutil
import docker


BASHRC = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin;" \
         "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin;" \
         "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin;" \
         "export PATH=$HOME/anaconda3/bin:$PATH;" \
         "source /root/anaconda3/bin/activate"


class MorphologyV0Task(Task):

    def __init__(self,
                 num_parallel=1,
                 num_gpus=1,
                 n_envs=4,
                 max_episode_steps=500,
                 total_timesteps=1000000,
                 domain='ant'):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        num_parallel: int
            the number of parallel trials in the docker container
        num_gpus: int
            the number of gpus to use in this docker container
        n_envs: int
            the number of parallel sampling environments for PPO
        max_episode_steps: int
            the maximum length of a episode when collecting samples
        total_timesteps: int
            the maximum number of samples to collect from the environment
        domain: str
            the particular morphology domain such as 'ant' or 'dog'
        """

        self.num_parallel = num_parallel
        self.num_gpus = num_gpus
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self.total_timesteps = total_timesteps
        self.domain = domain
        self.client = docker.from_env()

        # locate an existing docker image for the trainer
        found = False
        for image in self.client.images.list():
            found = found or 'morphing-datasets:latest' in image.tags

        if not found:

            # download the dockerfile
            url = 'https://raw.githubusercontent.com/' \
                  'brandontrabucco/morphing-datasets/master/dockerfile'
            r = requests.get(url)
            with open(os.path.join(DATA_DIR, 'dockerfile'), 'wb') as f:
                f.write(r.content)

            # copy the mujoco key into the data folder
            shutil.copyfile(os.path.expanduser(
                "~/.mujoco/mjkey.txt"), os.path.join(DATA_DIR, 'mjkey.txt'))

            # build the docker image manually
            self.client.build.build(path=DATA_DIR, tag='morphing-datasets')

        maybe_download('1ueVwGoKJarXLuOZkNP1dH0vt5Im9BbE-',
                       os.path.join(DATA_DIR, 'centered_ant_X.txt'))
        maybe_download('1Ml_S6Qq2rVsq6PBW7GLeI7JfTO9n7hIX',
                       os.path.join(DATA_DIR, 'centered_ant_y.txt'))
        maybe_download('1Hs8S7TWmw0Z5ZMu2vTVBPQwvZaXtpXNJ',
                       os.path.join(DATA_DIR, 'centered_dog_X.txt'))
        maybe_download('1U8svyRKGLerGVymGIRnG-FdAJ3NF_AWr',
                       os.path.join(DATA_DIR, 'centered_dog_y.txt'))

        x = np.loadtxt(os.path.join(DATA_DIR, f'centered_{domain}_X.txt'))
        y = np.loadtxt(os.path.join(DATA_DIR, f'centered_{domain}_y.txt'))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        self.x = x
        self.y = y

    def score(self,
              x: np.ndarray) -> np.ndarray:
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        # save the morphology in a shared folder
        np.save(os.path.join(DATA_DIR, 'morphology.npy'), x)

        # train policies using the designs
        stdout = self.client.containers.run(
            'morphing-datasets:latest',
            f"/bin/bash -c '{BASHRC}; "
            "conda run -n morphing-datasets "
            "python /root/morphing-datasets/evaluate_designs.py "
            "--local-dir /data "
            "--designs-file /data/morphology.npy "
            f"--num-parallel {self.num_parallel} "
            f"--num-gpus {self.num_gpus} "
            f"--n-envs {self.n_envs} "
            f"--max-episode-steps {self.max_episode_steps} "
            f"--total-timesteps {self.total_timesteps} "
            f"--domain {self.domain} '",
            mounts=[types.Mount('/data', DATA_DIR, type='bind')],
            auto_remove=True,
            stdout=True,
            stderr=True)

        # load the scores evaluated using docker
        return np.load(os.path.join(
            DATA_DIR, 'score.npy')).astype(np.float32).reshape([-1, 1])
