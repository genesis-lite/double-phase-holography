from setuptools import setup, find_packages

setup(
    name='double_phase_hologram',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'Pillow',
    ],
)
