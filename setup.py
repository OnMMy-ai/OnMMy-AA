from setuptools import setup, find_packages

setup(
    name='onmmy_ai',
    version='0.1.0',
    description='ONMMY AI Project - Advanced AI models and tools',
    author='ONMMY Team',
    author_email='contact@onmmy.ai',
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'pyyaml',
        'Pillow',
        'numpy',
    ],
    python_requires='>=3.7',
)
