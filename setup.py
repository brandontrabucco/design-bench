from setuptools import find_packages
from setuptools import setup


F = 'README.md'
with open(F, 'r') as readme:
    LONG_DESCRIPTION = readme.read()


INSTALL_REQUIRES = ['numpy', 'pandas', 'requests', 'scikit-learn', 'tape_proteins']
EXTRA_REQUIRES = ['gym[mujoco]', 'morphing-agents']


CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8']


setup(
    name='design-bench',
    packages=find_packages(include=['design_bench', 'design_bench.*']),
    version='1.4',
    license='MIT',
    description='Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Brandon Trabucco',
    author_email='brandon@btrabucco.com',
    url='https://github.com/brandontrabucco/design-bench',
    download_url='https://github.com/brandontrabucco/design-bench/archive/v1_4.tar.gz',
    keywords=['Offline', 'Benchmark', 'Model-Based Optimization'],
    install_requires=INSTALL_REQUIRES,
    extras_require={'all': EXTRA_REQUIRES},
    classifiers=CLASSIFIERS)
