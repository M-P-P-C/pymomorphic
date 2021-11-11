#!/usr/bin/env python3

"""
``pymomorphic.pymomorphic_py3``
================

This pymomorphic_py3 package provides functions to homomorphically encrypt and
operate on data using Python 3. Refer to pymomorphic_py2 to use with Python2.

This package contains the following functions:

Encryption/Decryption
---------------------

    key_generate
    encrypt
    encrypt2
    enc_2_mat
    log_scaling
    decrypt

Data Manipulation
-----------------

    modulus

ROS publication
---------------

    prep_pub_ros_str
    recvr_pub_ros_str

"""

import sys
import warnings
import os
import time
import timeit

import numpy as np
import random #used to generate random arrays used in the encryption process
import csv
#import rospkg #this pac
import json #used to convert lists to strings, used to send messages through ROS, as it does not support values larger than int64

import math
from operator import add
import secrets

#TODO: add check to ensure input arguments are integers
#TODO: update functions to accept integers instead of just lists, for ease of use.
#TODO: write test functions to ensure everything works when compiling packages


def main():

    #Initialize some variables for testing purposes
    my_p = 10**12
    my_L = 10**3
    my_r = 10**1
    my_N = 5

    #integers to encrypt and multiply
    m = [20]
    m2 = [600]
    
    #initialize KEY and HOM_OP
    my_key = KEY(p = my_p , L = my_L, r = my_r , N = my_N)
    my_operator = HOM_OP(p = my_p, L = my_L, r = my_r , N = my_N)


    my_c = my_key.encrypt(m)

    my_c2 = my_key.encrypt2(m2)


    my_c_mult = my_operator.hom_multiply(my_c,my_c2)

    
    my_c_dec = my_key.decrypt(my_c.tolist())

    my_p_mult = my_key.decrypt([my_c_mult])


    print("\n")
    print("Decrypted Variable 1: " + str(my_c_dec))
    print("Multiplication Result: " + str(my_p_mult))
    print("Expected Result:       " + str(m[0]*m2[0]))

    
    #arrays to encrypt and multiply
    M = [[10, 20, 30], [3, 4, 5]]
    M2 = [100, 100, 100]

    my_C = my_key.enc_2_mat(M)

    my_C2 = my_key.encrypt(M2)

    
    my_C_mult = my_operator.hom_mul_mat(my_C, my_C2)


    my_P_mult = my_key.decrypt(my_C_mult)

    print("\n")
    print("Expected Mult Result: " + str(np.dot(np.array(M),np.array(M2))))
    print("Decrypted Matrix Mult Result: " + str(my_P_mult))


    array_to_publish = my_key.prep_pub_ros_str([[1,2,3,4,5]])
    string_array = my_key.recvr_pub_ros_str(array_to_publish)

    print("\n")
    print("Array to publish in ROS " + array_to_publish)
    print("String array turned into a list " + str(string_array))

    #test log scaling function
    array_to_log_scale = np.array([-0.92506512, 0])
    array_scaled = my_key.log_scaling(array_to_log_scale, 3)

    print("\n")
    print("Array to scale " + str(array_to_log_scale))
    print("String array turned into a list " + str(array_scaled))

    #functions to time any tasks in "process_test" method
    #timeit.timeit(my_key.process_test)
    #timeit.timeit(lambda:my_key.encrypt2([20]), number = 1000)

    #example of z_values obtained with 3 neighboring robots in ROS
    #z_values = np.array([[ 0.00934644, -0.05645715, -0.80737489,  1.23556484], [-0.00632714,  0.67307276, -0.42058253,  1.25996497],[0.01942838,  0.72351229,  0.38469833,  1.2203629 ]])



    #enc_matlab(var.p, var.L, var.q, var.r, var.N, sk, np.zeros(int(math.log10(var.q))*(var.N+1), dtype = int).tolist()) 

    #cx22 = enc1_to_enc2(var.p, var.L, var.q, var.r, var.N, cx) #a function that transforms variables in enc2 form to enc1





def modulus(a, b, neg = False):
    """
    Calculate modulo of two numbers a%b, with the option to return also negative remainders using the flag "neg"

    Parameters
    ----------
    a : list
    b : list
    neg : bool, optional
        flag to determine whether to return negative remainders. 
        if False (default), it will return only positive results

    Examples
    --------
    Use case one:

    >>> mod = modulus([10],[8])
    >>> mod
    [2]

    Use case two, set argument "neg=True", for negative remainders:

    >>> mod = modulus([8],[10], neg=True)
    >>> mod
    [-2]
    """
    
    length = len(a)
    
    y = np.zeros((1,length), dtype = object)

    y = np.mod(a, b)

    if neg is True:
        y = np.where(y >= b/2, y-b, y)

    return y


class KEY:
    """ This class holds all methods needed to encrypt and decrypt data """

    def __init__(self, p : int = 10**13, L : int = 10**3, r : int =10**1, N : int = 50, seed = None):
        """
        Initialize the KEY class to encrypt and decrypt integers.

        Parameters
        ----------
        p : int
            the plaintext space.
        L : int

        r : int
            error to be injected into encrypted values.
        N : int
            key lenght of secret key.
        seed : int, optional
            set any value to return always the same key, useful for debugging or in the case encrption and decryption happen in different scripts.
            if None (default), generated numbers will remain random every time the class is called.

        Returns
        -------

        Examples
        --------
        Initilize the KEY class with:

        >>> my_key = KEY(p = 10**13, L = 10**3, r=10**1, N = 30)

        To encrypt a value after initializing KEY:

        >>> enc_m = my_key.Encrypt(2)

        """

        self.rand_set = random 
        self.rand_set_np = np.random #Initialize numpy's random package to determine if a seed is assigned or not

        #For DEBUG purposes only:
        self.rand_set.seed(seed) #These are used to generate the same key for the encryption and decryption scripts.
        self.rand_set_np.seed(seed)
        
        #Store input variables as class attributes
        self.q = p * L 
        self.p = p
        self.L = L
        self.r = r
        self.N = N

        #convert q to float to use in functions that don't accept large integers
        self.q_float = float(self.q)

        #Warn user if plaintext space is larger than int64, which could cause problems
        if self.q > sys.maxsize: #or maxint
            warnings.warn("plaintext space exceeds int64", Warning) #FIXME: this process needs checking


        self.secret_key = self.key_generate() #generate secret key for encryption.
        self.secret_key_np = np.array(self.secret_key, ndmin=2, dtype = object).T #transpose and store key in NumPy array

        #Initialize a variable used for the method "decryption"
        self.secret_key_np_dec = np.append([[1]], self.secret_key_np, axis=0) #version of the secret key with "1" appended to its start used in the decryption process

        #Initialize three variables used for the method "encryption2"
        self.lq = int(math.log10(self.q))
        self.R = np.kron(np.power(10, np.arange(self.lq, dtype = object)[:, None]), np.eye(self.N+1, dtype = object)) # "[:, None]" transposes the array made by np.arange
        self.mat_zeros_np = np.zeros(self.lq*(self.N+1), dtype = int)

    def plaintext_space_check(self, m):
        """
        Checks the message to be encrypted fits within the set plaintext space p
        
        Parameters
        ----------
        m : int or list?
            message to be encrypted
        """

        if self.p < m:
            warnings.warn("the message to be encrypted does not fit in plaintext space")


    def key_generate(self):
        """
        This function generates a secret key to encrypt and decrypt values
        """

        rand = [self.rand_set.randrange(self.q*self.L) for iter in range(self.N)] #for Python 2 since only Python 3 has "secrets"

        #secretsGenerator = secrets.SystemRandom()
        #rand = [secretsGenerator.randrange(self.q*self.L) for iter in range(self.N)] #More secure but doesn't support seeds

        sk = modulus(rand, self.q*self.L, neg = True)

        return sk

    def encrypt(self,m):
        """
        Encrypts message in plaintext form in enc1, or enc2 defined by the flag 'type' 
        
        Parameters
        ----------
        m : int or list?
            plaintext message to be encrypted

        Returns
        -------
        x : numpy.array, shape (len(m),N)

        Examples
        --------
        To encrypt a value after initializing KEY:

        >>> my_key = KEY(p = 10**3, L = 10**3, r=10**1, N = 5)
        >>> enc_m = my_key.encrypt([2])
        >>> enc_m 
        array([[-389913.0, 172905.0, -83785.0, -158262.0, 315695.0, -456781.0]], dtype=object)
            
        """

        n = len(m)
        
        #A = np.rint(self.rand_set_np.uniform(low = 0, high = self.q_float, size=(n, self.N)), dtype=np.float64)
        A = self.rand_set_np.uniform(low = 0, high = self.q_float, size=(n, self.N))
        A_list = [[int(i) for i in inner] for inner in A]
        A = np.array(A_list, dtype=object)

        m = np.array(m, ndmin=2).T

        temp_rand= self.rand_set_np.randint(low = 1, high = self.r, size = [n,1])

        e = modulus(temp_rand, self.r, neg = True)

        b = np.dot(-A, self.secret_key_np)


        add = np.add(self.L*m,e)

        b = b+add

        A = np.append(b, A, axis=1) #appends the calculated variable "b" to the randomly generated array "A"

        
        ciphertext = modulus(A, self.q, neg=True)


        return ciphertext #[list(map(long,i)) for i in ciphertext] 




    def process_test(self):
        mat_zeros_np = np.zeros(self.lq*(self.N+1), dtype = object)

    def process_test2(self):
        mat_zeros_np = np.zeros(self.lq*(self.N+1)).astype(object)



    def encrypt2(self, m): #This function encrypts the values to be homomorphically multiplied
        """
        Encrypts message in plaintext form using enc2, used to encrypt the multiplicand in c1*c2 
        
        Parameters
        ----------
        m : int or list?
            plaintext message to be encrypted

        Returns
        -------
        x : numpy.array, shape (len(m),N)

        Examples
        --------
        To encrypt a value after initializing KEY:

        >>> my_key = KEY(p = 10**3, L = 10**3, r=10**1, N = 5)
        >>> enc_m = my_key.encrypt2([2])
        >>> enc_m 
        array([[-389913.0, 172905.0, -83785.0, -158262.0, 315695.0, -456781.0]], dtype=object)
            
        """
    
        mat_zeros_enc = self.encrypt(self.mat_zeros_np.tolist())

        hold_1 = np.multiply(m,self.R)
        
        hold = np.add(hold_1, mat_zeros_enc)

        #ciphertext = np.empty((len(mat_zeros_enc[0]), len(mat_zeros_enc)), dtype = object)
        
        ciphertext = modulus(hold, self.q, neg=True)

        return ciphertext.tolist()

    
    def enc_2_mat(self, m): #This function encrypts the values of a matrix to be homomorphically multiplied
        n1 = len(m)
        n2 = len(m[0])

        cA = np.zeros((n1, n2, int(math.log10(self.q)*(self.N+1)), self.N+1)).astype(int).tolist()

        for i in range(n1):
            for j in range(n2):
                cA[i][j] = self.encrypt2([m[i][j]])
                #could just use the insert syntax instead of assigning the whole matrix

        return cA

    def decrypt(self, c):
        '''
        decrypts a homomorphically encrypted value
        
        Parameters
        ----------
        c : int or list?
            ciphetext message to be derypted

        Returns
        -------
        x : numpy.array, shape (len(m),N)

        Examples
        --------
        To encrypt a value after initializing KEY:

        >>> my_key = KEY(p = 10**3, L = 10**3, r=10**1, N = 5)
        >>> enc_m = my_key.encrypt([2])
        >>> dec_m = my_key.decrypt(enc_m)
        >>> dec_m 
        array([[2]], dtype=object)
        '''

        c_np = np.array(c, dtype = object)
        

        dot = np.dot(c_np, self.secret_key_np_dec)

        plain = modulus(dot, self.L*self.p, neg = True).astype(float) #should this be self.q instead?

        plaintext_np = plain/self.L

        if type(plaintext_np) != float:
            plaintext_np = np.around(plaintext_np).astype(int)
        else:
            plaintext_np = round(plaintext_np)
            plaintext_np = int(plaintext_np)


        #print(plaintext_np.tolist())
        return plaintext_np.tolist()


    def output_key_to_csv(self, robot):
        '''
        Outputs generated secret key to a csv file
        
        Parameters
        ----------
        robot : int
            number of Nexus robot the key belongs to (used in output filename)

        Examples
        --------
        To encrypt a value after initializing KEY:

        >>> my_key = KEY(p = 10**3, L = 10**3, r=10**1, N = 5)
        >>> my_key.output_key_to_csv(1)
        '''

        PATH = os.path.dirname(os.path.abspath(__file__)) #rospack.get_path(ros_package)
        FILEPATH = os.path.join(PATH, 'private_key_'+str(robot)+'.csv')

        with open(FILEPATH, "w") as output:
            writer = csv.writer(output, delimiter=',')
            for val in self.sk:
                writer.writerow([val])

    def read_key_from_csv(self, robot):

        sk = []

        PATH = os.path.dirname(os.path.abspath(__file__)) #rospack.get_path(ros_package)
        FILEPATH = os.path.join(PATH, 'private_key_'+str(robot)+'.csv')

        with open(FILEPATH, 'rb') as f: #This section reads the csv file to gather the private key to encrypt
            reader = csv.reader(f, delimiter='\n')
            lis = list(reader)
            for i in lis:
                sk.append(eval(i[0]))

        self.secret_key = sk

    def log_scaling(self, vk, sp_vk):
        """
        This function scales its input (z_values) until all values reach a minimum amount set by the input "scal"

        Parameters
        ----------
        vk : int or list?
            message to be scaled
        sp_vk : int
            desired significant figures

        Returns
        -------
        VK : numpy.array, shape (len(m),N)
            scaled message
        sp_vk : int?
            scaling amount

        Examples
        --------
        To scale a value after initialing KEY:

        >>> my_key = KEY(p = 10**3, L = 10**3, r=10**1, N = 5)
        >>> p = 2.5
        >>> scaled_p = my_key.log_scaling(p, 3)
        >>> scaled_p
        250
        """

        #if type(vk) == float:
        vk = np.array(vk, dtype = float)[np.newaxis]
        #else:
        #    a=1

        sp_vk = np.array(sp_vk)

        #vk = np.array(vk, dtype = object)

        test = np.abs(vk[np.nonzero(vk)]).min()

        flr = np.floor(np.log10(np.abs(test))).astype(int)
        flr = np.full((vk.shape[0], vk.shape[1]), flr, dtype=object)

        tens = np.full((vk.shape[0], vk.shape[1]), 10, dtype=object) #array od 10's 
    
        S_vk = tens**(sp_vk-flr-1) #Shoud scale all numbers by the same?  
    
        VK = (vk*S_vk).astype(int)

        #VK = VK-np.mod(vk, 1) #Remove Decimals
        #VK = [long(i) for i in np.around(VK, decimals =0)]

        return VK, S_vk[0][0]




class HOM_OP:
    """
    This class contains all necessary methods to multiply two homomorphically encrypted ciphertexts
    """

    def __init__(self, p : "int" = 10**13, L = 10**3, r=10**1, N = 50, seed=None):
        """Initialize the HOM_OP class to encrypt and decrypt numbers.

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        Initilize the HOM_OP class with:

        >>> hom_operator = HOM_OP()

        To multiply ciphertexts "c1*c2":

        >>> enc_m = hom_operator.multiply(c1,c2)

        """

        self.q = p * L 
        self.p = p
        self.L = L
        self.r = r
        self.N = N

        self.lq = int(math.log10(self.q)) #used in "decomp" method


    def decomp(self, c1): #function to carry out before multiplying used by the function "hom_mul"


        c1_np = modulus(c1, self.q)

        BBB=np.zeros((c1_np.shape[0],0), dtype = object)

        for i in range(self.lq):

            dum = int(10**(self.lq-1-i))

            Q_np = np.mod(c1_np, dum)
            
            Q_np = np.subtract(c1_np, Q_np)

            BBB = np.append(Q_np//dum, BBB, axis= 1) # "//" (true_divide) is used to ensure the output of the division is integers and not floats

            #c1 = [j - k for j, k in zip(c1,Q)]
            
            c1_np = np.subtract(c1_np,Q_np)

        return BBB


    def hom_multiply(self, c1, c2):
        ''' This function performs the multiplication of two homomorphically encrypted values, c2 must be encrypted using the function "enc2" using NUMPY '''
        
        c1_np = np.array(c1, dtype = object)
        c2_np = np.array(c2, dtype = object)

        c1_np = self.decomp(c1_np)

        #x = np.zeros([1, c2_np.shape[1]], dtype = object)
        
        x = np.dot(c1_np, c2_np) 

        return x.tolist()[0]


    def hom_mul_mat(self, c1, c2):
        ''' This function performs the multiplication of a homomorphically encrypted matrix with a vector, c2 must be encrypted using the function "enc2" '''
        
        n4 = len(c1)
        n3 = len(c1[0])
        #n2 = len(c1[0][0])
        #n1 = len(c1[0][0][0])

        #c1 = decomp(q, c1)

        #Mm=[[0]*(N+1)]*n4

        Mm_np = np.zeros((n4,self.N+1), dtype = object)

        #multiplied = [0]*len(Mm[0])
        #multiplied_np = np.zeros((1,n1), dtype = object)


        dec_c2 = self.decomp(c2)
        
        for i in range(n4):
            for j in range(n3):
                #temp1 = Mm_np[i]
                temp2 = dec_c2[j]
                temp3 = c1[i][j]

                multiplied = np.dot(temp2, temp3)

                added = Mm_np[i] + multiplied

                Mm_np[i] = modulus(added,self.q, neg=True)


        return Mm_np.tolist()



def prep_pub_ros_str(self, c):
    '''Used to convert a list to a string to publish encrypted values in ROS
    
    Parameters
    ----------
    c : list
    
    Returns
    -------
    array_into_string: string
    '''

    array_into_string = json.dumps(c)
    
    return array_into_string

def recvr_pub_ros_str(self, c):
    '''used to convert a string to a list that was previously modified using the method "prep_pub_ros_str()"
    
    Parameters
    ----------
    c : string
    
    Returns
    -------
    pub_list: list
    '''

    pub_list = json.loads(c)
        
    return pub_list




#TODO: Review code below. The remaining code below is a translation of Matlab code provided by Junsoo Kim, that could help speed up
#some processes, but at the moment it is not applicable

###################################################################################################################

def enc_matlab(p, L, q, r, N, secret_key, mat_inp): #function from Matlab code for ITH (not for matrices)
    '''
    This function is to apply LWE encryption to matrices
    '''
    #signal_bound = 2
    #s_G = 2**19
    #L = 2**26

    n_ = N+1
    mod_g = 2**math.ceil(math.log(2*2*L*2**19,2))
    mod_q = q#int(mod_g)

    #d = math.ceil(math.log(mod_q,2)/nu)
    mod_e = r
    #mod_qe = mod_q-math.ceil(mod_e/2)
    #mod_qg = mod_q/mod_g
    mod_q1 = mod_q -1

    #mod_s = 3
    #s= random.randrange(1,mod_s)

    l2 = len(mat_inp) #rows
    l1 = len(mat_inp[0]) if type(mat_inp[0]) is list else 1 #columns

    c = [[0]*l2]*(n_*l1)#[[[0]*l2]*l1*n_ #array of zeros

    from operator import add, sub
    for i in range(l1):
        #temp = zeros( n_, l2, 'uint64');
        temp  = [[0]*l2]*(n_)#[[[0]]*n_]*l2

        #temp(2:n_,:) = uint64(randi(mod_q,n_-1,l2));
        for j in range(1,n_):
            rand_vect = [random.randrange(mod_q) for iter in range(l2)]
            temp[j] = rand_vect

        #temp_e = uint64(randi(mod_e,1,l2))+uint64(mod_qe*ones(1,l2,'uint64'));
        temp_e1 = [random.randrange(mod_e) for iter in range(l2)]
        temp_e2 = [mod_q]*l2
        temp_e =  list( map(add, temp_e1, temp_e2) )

        #temp(1,:) = bitand( mod_q*ones(1,l2,'uint64') - mult(s,temp(2:n_,:))+ uint64(mod_qg*(mod_g+round(m(i,:))))+temp_e ,mod_q1);
        term = [mod_q]*l2
        term2 = mult(q, secret_key, temp[1:])[0]
        dum = mat_inp if type(mat_inp[0]) is int else mat_inp[i]
        term3 = [j+mod_g for j in dum]

        added = list( map(sub, term, term2) )
        added2 = list( map(add, term3, temp_e) )
        added = list( map(add, added, added2) )

        temp[0] = bitand([added],mod_q1)[0]

        #c((i-1)*(n_)+1:i*(n_),:) = temp;
        c[(i)*(n_):(i+1)*(n_)] = temp

    return c

def dec_mat(p, L, q, r, N, secret_key, mat_inp):
    '''
    This function is to apply LWE decryption to matrices
    '''

    signal_bound = 2
    s_G = 2**19
    #L = 2**26
    n=N
    n_ = n+1
    mod_g = 2**math.ceil(math.log(2*signal_bound*L*s_G,2))
    mod_q = int(mod_g)
    nu = 16
    d = math.ceil(math.log(mod_q,2)/nu)
    mod_e = 2**6+1
    mod_qe = mod_q-math.ceil(mod_e/2)
    mod_qg = mod_q/mod_g
    mod_q1 = mod_q -1

    #q = double(mod_q)

    t= [1]+ secret_key
    l = len([mat_inp])/(n+1)
    y = np.zeros((l, 1)).astype(int).tolist()

    for i in range(0,l):
        temp = mult(t,c[(i-1)*(n+1)+1:i*(n+1)])
   
        m = temp - (temp>=q/2) * q
    
        y[i,1] = round(m * (mod_g/q))

        return y

def enc_gains(L, N, secret_key, mat_inp):
    '''
    This function is used to GSW encrypt the gains of the control law for the Infinite Time Horizon computations
    ''' 
    global signal_bound

    mod_s = 3
    #n=1
    n_ = N #n+1
    s_G = 2**19
    L = 2**26
    import math
    mod_g = 2**math.ceil(math.log(2*signal_bound*L*s_G,2))
    mod_q = int(mod_g)
    mod_e = 2**6+1
    nu = 16
    d = math.ceil(math.log(mod_q,2)/nu)
    mod_qe = mod_q-math.ceil(mod_e/2)
    mod_qg = mod_q/mod_g
    mod_q1 = mod_q -1
    nu2 = 2**nu-1

    l1 = len(mat_inp) #rows
    l2 = len(mat_inp[0]) #columns

    #c = [[0]*int(l2*(n_)*d)]*l1*(n_) #array of zeros
    c = [[0]*int(l2*(n_)*d) for i in range(l1*(n_))] #initialize list of lists
    i3 = 0
    #m_temp = double(mod_q)*ones(l1,l2)+mat_inp
    ones_mat = [[mod_q]*l1]*l2
    
    from operator import add
    m_temp = [[]]*l2
    for i in range(l2):
        m_temp[i] = list( map(add, ones_mat[i], mat_inp[i]) )
    
    c_temp = []
    for ii in range(int(d)):
        for j in range(l2):
            for i in range (l1):
                #c( (i-1)*(n_)+1:i*(n_), i3 +(j-1)*(n_)+1 : i3+j*(n_)) =bitand( c( (i-1)*n_+1:i*n_   , i3 +  (j-1)*n_+1 : i3+j*n_) + (m_temp(i,j))*eye(n_,'uint64') + Enc(zeros(1,n_)),mod_q-1);
                #temp1 = c[(ii)*2:(ii+1)*2][i3 +  (j)*2 : i3+(j+1)*2]
                temp1 = [item[i3 + (j)*2 : i3+(j+1)*2] for item in c[(i)*n_:(i+1)*n_]]

                ident = np.identity(2).astype('int').tolist()
                temp2 = [[m_temp[i][j]*kk for kk in jj] for jj in ident] #multiplication with Identity matrix
                temp3 = enc_mat(secret_key, [[0]*n_])  
                added = temp1[:]
                for jj in range(len(temp1)):
                    added[jj] = list(map(add, added[jj], temp2[jj]))
                    added[jj] = list(map(add, added[jj], temp3[jj]))
                
                temp1 = bitand( added,mod_q-1)

                for q in range(len(temp1)):
                    for k in range(len(temp1[0])):
                        c[(q)+n_*i][i3 + (k)+(j*n_)] = temp1[q][k]
        #m_temp = bitand(bitshift(m_temp,nu),mod_q-1) #bitshift is done with >> or <<  e.g. 2>>1
        #bitshift(m_temp,nu)
        m_temp = bitand(bitshift(m_temp,nu),mod_q-1)
        i3 = i3+(n_)*l2

    return c

def splitm(d, nu, nu2, mat_inp):

    #y = zeros(size(c,1)*d, size(c,2), 'uint64')

    l1 = len(mat_inp)
    l2 = len(mat_inp[0]) if type(mat_inp[0]) is list else 1

    y = [[0]*l2 for i in range(l1*d)] #initialize list of lists
    #l1 = size(c,1)
    
    temp = mat_inp[:]

    #nu2=[[nu2]*l2]*l1

    for i in range(d):
        y[(i)*l1:(i+1)*l1] = bitand(temp, nu2)
        temp = bitshift(temp,-nu)
    #for i = 1:d
        #y((i-1)*l1+1:i*l1 ,:) = bitand(temp, nu2)
        #temp = bitshift(temp,-nu)

    return y


def bitshift(mat_A, shift):
    '''
    returns the element-wise "and" of elements between lists. inputs must be list of lists

    E.G. calculating the bitshift() of 20 = 010100 by 1

    010100
    
    101000 =  40
    '''
    #mat_A = [[24214,13213],[24214,13213]] 
    #mat_A = [[206001532940593L]]
    #mat_A = [[24214,13213]]
    #mat_A= [[2],[3],[4]]

    shift = 1

    l1 = len(mat_A)
    l2 = len(mat_A[0]) if type(mat_A[0]) is list else 1

    out = [[0]*l2 for i in range(l1)] #initialize list of lists
    for k in range(l1):
        for i in range(l2):
            
            if shift > 0:
                out[k][i] = mat_A[k][i]<<abs(shift)
            elif shift < 0:
                out[k][i] = mat_A[k][i]>>abs(shift)
    return out

def bitand(mat_A, mat_B):
    '''
    returns the element-wise "and" of elements between lists. inputs must be list of lists

    E.G. calculating the bitand() of 20 = 010100, and 24 = 011000

    010100
    &&&&&&
    011000
    ||||||
    010000  =  16
    '''
    #import pdb
    #pdb.set_trace()

    #mat_A = [[24214,13213],[24214,13213]] #l1 = 2, l2 = 2
    #mat_B = 44#[[44, 44], [44,44]]
    #out = [[4, 12], [4, 12]]

    #mat_A = [206001532940593L] #l1 = 1, l2 = 1
    #mat_B = 281474976710655L
    #out = [[206001532940593L]]

    #mat_A = [[24214,13213]] #l1 = 2, l2 = 1
    #mat_B = 44
    #out = [[4, 12]]

    #mat_A= [[2],[3],[4]] #l1 = 3, l2 = 1
    #mat_B= 6
    #out = [[2], [2], [4]]

    

    l1 = len(mat_A)
    l2 = len(mat_A[0]) if type(mat_A[0]) is list else 1

    if type(mat_A[0]) is list:
        mat_B = [[mat_B]*l2]*l1 
    else:
        mat_B = [mat_B]*l1 


    zzz = [[]]*l1
    if type(mat_A[0]) is list:   
        for i in range(len(mat_A)):
            zzz[i] = zip(mat_A[i],mat_B[i]) 
    else:
        zzz = zip(mat_A,mat_B)
    #zzz = zip(mat_A,mat_B)
    #print(zzz)
    j=0
    out = [[0]*l2 for i in range(l1)] #initialize list of lists
    for i in range(l1):
        for k in range(l2):
            if type(mat_A[0]) is list: #l2>1:
                xxx = zzz[i][k]
            else:
                xxx = zzz[k]
            #zzz = zip(mat_A[i],mat_B[i])

            if type(xxx) is list:
                out[i][k] = int(xxx[0][0]) & int(xxx[0][1]) 
            else:
                out[i][k] = int(xxx[0]) & int(xxx[1]) 
            j=j+1 
            #    for k in range(len(mat_A[0])):
            #        out[i][k] = int(zzz[i][k]) & int(zzz[i][k])        
              
    #if len(mat_A)==1:
    #    out = [k & l for k, l in zip(mat_A, mat_B)]
    #else:
    #    out = [[k & l for k, l in zip(i, j)] for i, j in zip(mat_A, mat_B)]

    return out


def mult(q, c, C):

    l1 = len([c])
    l2 = len(c)
    l3 = len(C[0])

    y = np.zeros((l1, l3)).astype(int).tolist()
    
    for ii in range(0,l3):
        temp = np.zeros((l1, 1)).astype(int).tolist()

        for i in range(0,l2):
            temp2 = bitand([c[i]*C[i][ii]],q)
            temp = bitand( list(map(add, temp[0], temp2[0])), q)

        for sublist in y:
            sublist[ii] = temp[0][0]

    
    return y



if __name__ == '__main__':
    main()