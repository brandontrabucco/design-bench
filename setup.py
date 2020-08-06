from setuptools import find_packages
from setuptools import setup


PACKAGES = ['numpy',
            'pandas',
            'gym[mujoco]',
            'requests',
            'scikit-learn',
            'docker']


setup(name='design-bench',
      description='Design Benchmarks for Model-Based Optimization',
      license='MIT',
      version='0.1',
      zip_safe=True,
      include_package_data=True,
      packages=find_packages(),
      install_requires=PACKAGES)
