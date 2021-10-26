from setuptools import find_packages, setup

setup(
    name='pymomorphic3',
    packages=find_packages(include=['pymomorphic3']),
    version='0.2.0',
    description='A Python library of functions to perform homomorphic encryption, decryption and multiplication',
    author='Mariano Perez Chaher',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='pymomorphic3_test',
)

#Activate the python venv with "source pymomorphic3_venv/bin/activate"

#Build Library with "python setup.py bdist_wheel" in root folder

# pip uninstall pymomorphic3

#Install library to PATH with "pip install dist/pymomorphic3-0.1.0-py3-none-any.whl"