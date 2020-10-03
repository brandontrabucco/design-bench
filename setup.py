from setuptools import setup


INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'gym[mujoco]',
    'requests',
    'scikit-learn',
    'morphing-agents']


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
    packages=['design_bench'],
    version='1.0',
    license='MIT',
    description='Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization',
    author='Brandon Trabucco',
    author_email='brandon@btrabucco.com',
    url='https://github.com/brandontrabucco/design-bench',
    download_url='https://github.com/brandontrabucco/design-bench/archive/v1.tar.gz',
    keywords=['Offline', 'Benchmark', 'Model-Based Optimization'],
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS)
