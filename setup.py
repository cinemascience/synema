from setuptools import setup

setup(
    name='synema',
    version='0.1',
    packages=['synema', 'synema.encoders', 'synema.models',
              'synema.renderers', 'synema.samplers'],
    url='https://github.com/cinemascience/synema',
    license='BSD',
    author='Li-Ta Lo',
    author_email='ollie@lanl.gov',
    description='View Synthesis for Cinema',
    install_requires=[
        'flax==0.10.4',
        'h5py==3.11.0',
        'jax>=0.5.2',
        'jaxtyping==0.2.33',
        'matplotlib==3.8.4',
        'numpy==1.26.4',
        'optax==0.2.3',
        'setuptools==69.0.3',
        'tqdm>=4.66.4'
    ]
)
