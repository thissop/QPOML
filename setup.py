from distutils.core import setup

setup(
    name='QPOML',
    version='0.1dev',
    author='Thaddaeus Kiker',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'pandas', 'matplotlib', 'astropy', 'sklearn']
)