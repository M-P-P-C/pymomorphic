# Homomorphic Encryption in Python

This python library was developed as a part of a master graduation thesis for the University of Groningen. The functions for Homomorphic Encryption are based on the work by Kim J. in [[1]](#1)


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
  <img src="https://github.com/M-P-P-C/pymomorphic/blob/main/media/encryption_performance.png?raw=true" width="500">
</p>

# Code Example

```python
>>> from pymomorphic import pymomorphic_py3 as pymorph3

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

In this section I hope to add a clear step-by-step process for any DTPA lab student to work on the repository.

(This section will eventually go to a COMMITTING.md file to declutter the README)

## Configuring git and Github

(Coming later)

## Setting up your Virtual Environment

To experiment with this library without messing with your Python installation you'll first need a virtual environment to do it in. This section will quickly give you the commands you'll need to set it up, you'll find a lot more information on virtual envionments online.

First I reccomend you make a folder where you can store multiple virtual environments if you need to:

```bash
mkdir "my_python_envs"
```

then cd into the created folder, clone this Github repo, and setup the virtual env:


```bash
cd "my_python_envs" 

#This will download the files from github
git clone -b <branch-name> --single-branch https://<your-username>:<your-private-token>@github.com/M-P-P-C/pymomorphic.git 

#cd into the folder you downloaded from github
cd "pymomorphic" 

#Then set up the Python Virtual Environment into a folder called "pymomorphic3_venv"
python3 -m venv "venv"

#Finally, you can quickly install all required dependencies into your virtual environment with
pip install -r "requirements.txt"
```
Depending on the branch you cloned, your folder should look something like this:

```bash
.
├── media
│   └── encryption_performance.png
├── pymomorphic
│   ├── __init__.py
│   ├── performance_analysis.py
│   └── pymomorphic_py3.py
├── README.md
├── requirements.txt
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share
```

with the folder containing your venv and the pymomorphic repo you are ready to start making changes, commiting them, and sending pull requests.

## Useful git commands

This is a small list of git commands you'll find yourself using often:

```bash
# Return which files have been altered and what is available to commit
git status 

# Staging files for your next commit
git add <file-to-be-commited>

# Commit changes to your local branch
git commit -m "<ypur-message-here>"

# Send files to the github repo (needs some extra configuration, like a private token)
git push 
```

## Project Structure

This repository is designed to provide an easily accessible library for Homomorphic Encryption functions. The code is divided into 3 categories, Python 2, Python 3, and Cython. Each language has its own branch that pull requests should be sent to for each respectve one. Then, within the "main" branch they are all combined for easy distribution.

The "tests" folder uses unittest to contain methods that check the functioning of the code. Ideally, each time a new method is added a new test is added to ensure its functioning.


# Resources


# References

<a id="1">[1]</a> 
Kim J., Shim H., Han K. (2020). 
"*Comprehensive Introduction to Fully Homomorphic Encryption for Dynamic Feedback Controller via LWE-based Cryptosystem*". 
In: Farokhi F. (eds) Privacy in Dynamical Systems Springer, Singapore. https://doi.org/10.1007/978-981-15-0493-8_10
