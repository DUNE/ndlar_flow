import setuptools

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('VERSION', 'r') as fh:
    version = fh.read().strip()

setuptools.setup(name='module0_flow',
                 version=version,
                 description='An h5flow-based analysis framework for Module0 data',
                 long_description=long_description,
                 long_description_content_type='text/x-rst',
                 author='Peter Madigan',
                 author_email='pmadigan@berkeley.edu',
                 # package_dir='module0_flow',
                 packages=[p for p in setuptools.find_packages(where='.') if 'module0_flow' in p],
                 python_requires='>=3.7',
                 install_requires=[
                     'h5py>=2.10',
                     'pytest',
                     'scipy',
                     'scikit-image',
                     'scikit-learn',
                     'h5flow>=0.1.0'
                 ]
                 )
