from pymomorphic3 import myfunctions as pmh
from pymomorphic3 import myfunctions2 as pmh2

import numpy as np

import pytest

'''
python setup.py bdist_wheel
pip install dist/pymomorphic3-0.1.0-py3-none-any.whl --force-reinstall
'''

#def main():
#    test_mod_hom()

class TestModulusMethod:

    def test_modulus_hom(self):
        
        assert pmh.mod_hom(1000,10000)==np.array([1000])

    def test_modulus_hom_neg(self):
        
        assert pmh.mod_hom(1000,10000)==np.array([1000])
        #hom_key = pmh2.KEY(p = 10**13, L = 10**3, r=10**1, N = 50)
        #pmh2.KEY()

        #hom_key.process_test()




if __name__ == '__main__':
    pytest.main()