
import unittest

from pymomorphic import pymomorphic_py2 as pmh

import numpy as np

'''
This file carries our unit testing for the pymomorphic package.

To carry out the tests run this in the command line:

"python3 -m unittest tests/test_pymomorphic_py2.py"

python setup.py bdist_wheel
pip install dist/pymomorphic3-0.1.0-py3-none-any.whl --force-reinstall
'''

#def main():
#    test_mod_hom()

class TestModulus(unittest.TestCase):

    def test_modulus_hom(self):
        
        self.assertEqual(pmh.modulus([1000],[10000]),np.array([1000]))

    def test_modulus_hom_neg(self):
        
        self.assertEqual(pmh.modulus([1000],[10000]),np.array([1000]))

class TestEncMethod(unittest.TestCase):

    def test_encryption_int(self):
        
        #Initialize some variables for testing purposes
        my_p = 10**12
        my_L = 10**3
        my_r = 10**1
        my_N = 5

        m = [20]
        m2 = [600]

        #initialize KEY and HOM_OP
        my_key = pmh.KEY(p = my_p , L = my_L, r = my_r , N = my_N)
    
        self.assertEqual(pmh.modulus([1000],[10000]),np.array([1000]))


class TestEncMultiplicationMethod(unittest.TestCase):

    def test_encrypted_multiplication(self):
        #Initialize some variables for testing purposes
        my_p = 10**12
        my_L = 10**3
        my_r = 10**1
        my_N = 5

        m = [20]
        m2 = [600]

        #initialize KEY and HOM_OP
        my_operator = pmh.HOM_OP(p = my_p, L = my_L, r = my_r , N = my_N)
'''
class TestKeyClass(unittest.TestCase):

    def test_key_init(self):

        hom_key = pmh.KEY(p = 10**13, L = 10**3, r=10**1, N = 50)
'''

if __name__ == '__main__':
    unittest.main()