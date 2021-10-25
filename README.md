# Homomorphic Encryption in Python

This python script was developed as a part of my graduation theses for the University of Groningen.


## Table of contents
<!--ts-->
  * [Performance](#performance)
  * [Code Example](#code-example)
  * [Resources](#resources)
  * [Contributing](#contributing)
<!--te-->


# Performance

The expected performance of the functions within the library with increasing secret key length:

<p align="center">
  <img src="https://github.com/M-P-P-C/pymomorphic3/blob/main/media/encryption_performance.png?raw=true" width="500">
</p>

# Code Example

```python
>>> from pymomorphic3 import pymomorphic_py3 as pymorph3

>>> m = 600

>>> my_key = pymorph3.KEY(p = 10**13 , L = 10**3, r = 10**1 , N = 4)

>>> my_c = my_key.encrypt(m)

>>> decrypted_c = my_key.decrypt(my_c)
```
# Install as a Package

I recommend installing this in a virtual environment rather than your main python installation to avoid any problems.

First download the .whl file in the "dist" folder and run this command in the same directory:

```python
pip install pymomorphic3-0.1.0-py3-none-any.whl
```

# Using Python Script

If you just want to do a quick test without setting up virtual environments or installing anything, download any of the .py files on the "pymomorphic3" folder.

With these files you should be able to call the functions

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


Setting up a virtual environment (venv), git commands and project structure. COMING SOON 

# Resources


