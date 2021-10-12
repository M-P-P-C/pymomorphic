#!/usr/bin/env python3

import csv
import random
#import rospkg
import numpy as np
import os
import math
from operator import add

import time
import json

import pandas as pd

#add check to ensure inputted numbers are (integers) and warning

def main():
    #print "main function called, this function is used when trying to debug this script. It get's executed when running the file directly"
    variables_define(p = 10**13, L = 10**3, r=10**1, N = 50)


    time_enc1 = pd.DataFrame()
    time_enc2 = pd.DataFrame()
    time_mult = pd.DataFrame()
    time_dec = pd.DataFrame()

    
    testt = np.array([-0.92506512, 0])
    log_scaling(testt, 3)

    m = [20]

    m2 = [600]
            

    #mult(281474976710655, [1,3], [[158777085946917,68804555223388,53304918513469], [109973083742059,71648586144056,25010707954525]])
    #enc_matlab(var.p, var.L, var.q, var.r, var.N, sk, np.zeros(int(math.log10(var.q))*(var.N+1), dtype = int).tolist()) 
    #enc_mat(var.p, var.L, var.q, var.r, var.N, sk, [1,2,3]) 
    #dec_mat(var.p, var.L, var.q, var.r, var.N, sk, enc_mat) 

    #start_encmat = time.time()
    #enc_matlab(var.p, var.L, var.q, var.r, var.N, sk, np.zeros(int(math.log10(var.q))*(var.N+1), dtype = int).tolist())
    #enc_matlab(var.p, var.L, var.q, var.r, var.N, sk, np.zeros(10, dtype = int).tolist())
    #end_encmat = time.time()

    dataf = pd.DataFrame()

    for j in range(5):
        for i in range(1,20,10):


            var = variables_import()

            var.N = i

            key_generate(var.q, var.L, var.N, 1)

            sk = key_import(1)

            print("\n")
            #print("Variable 1 to Encrypt: " + str(m[0]))
            #print("Variable 2 to Encrypt: " + str(m2[0]))
            print("Round: " + str(j)+","+str(i))


            start_enc1 = time.time()
            ciphertext = enc_1(var.p, var.L, var.q, var.r, var.N, sk, m) 
            end_enc1 = time.time()

            start_enc2 = time.time()
            ciphertext2 = enc_2(var.p, var.L, var.q, var.r, var.N, sk, m2)
            end_enc2 = time.time()

            start_mult = time.time()
            multiplied = hom_mul(var.q, ciphertext, ciphertext2)
            end_mult = time.time()

            start_dec = time.time()
            decrypted = dec_hom(var.p, var.L, sk, [multiplied])
            end_dec = time.time()

            time_enc1 = time_enc1.append([end_enc1 - start_enc1])
            time_enc2 = time_enc2.append([end_enc2 - start_enc2])
            time_mult = time_mult.append([end_mult - start_mult])
            time_dec = time_dec.append([end_dec - start_dec])

        functions=['Enc1', 'Enc2', 'Mult', 'Dec']
        temp_df = pd.concat([time_enc1,time_enc2,time_mult,time_dec], axis =1)

        temp_df.columns = functions
        dataf = pd.concat([dataf,temp_df], axis = 1)
        time_enc1 = pd.DataFrame()
        time_enc2 = pd.DataFrame()
        time_mult = pd.DataFrame()
        time_dec = pd.DataFrame()

    dirpath = os.getcwd()

    dataf.to_csv(path_or_buf='Ncrypt_200.csv')

    #te=prep_pub_ros_str(multiplied)
    #recvr_pub_ros_str(te)

    #enc_matlab(var.p, var.L, var.q, var.r, var.N, sk, np.zeros(int(math.log10(var.q))*(var.N+1), dtype = int).tolist()) 

    print("\n")
    print("Decrypted Variable 1: " + str(dec_hom(var.p, var.L, sk, [ciphertext])[0][0]))
    print("Multiplication Result: " + str(dec_hom(var.p, var.L, sk, [multiplied])[0][0]))
    print("Expected Result:       " + str(m[0]*m2[0]))
    #print "\n"
    #print "Variable 1 in it's encrypted form: " #Expected = [[-49807360L, 77641302L, -495364491, -338818076, 128598971L, 405696980L]]

    #print ciphertext



    functions=['Enc1', 'Enc2', 'Mult', 'Dec']
    slow_data=[time_enc1,time_enc2,time_mult, time_dec]

    
    print("\n")
    print("Time Enc1 " + str(time_enc1)) #Fast
    print("Time Enc2 " + str(time_enc2)) #Slow at high N
    print("Time Mult " + str(time_mult))
    print("Time Dec " + str(time_dec))

    import seaborn as sns
    import matplotlib.pyplot as plt


    data = pd.DataFrame()
    data["Step"]=functions+functions
    data["time"]=slow_data
    data["mode"]=["old","old","old","old","new","new","new","new"]

    plt.ioff()
    fig = plt.figure(1,(10,10))
    ax =fig.add_subplot(1,1,1)
    ax=sns.barplot(data=data, x="Step", y="time", hue="mode")
    #plt.show()'''
    
    import timeit
    #print timeit.timeit(enc_1_np(var.p, var.L, var.q, var.r, var.N, sk, m), number=100)

    #plt.savefig('times.png')
    #plt.ioff()

    #example of z_values obtained with 3 neighboring robots
    #z_values = np.array([[ 0.00934644, -0.05645715, -0.80737489,  1.23556484], [-0.00632714,  0.67307276, -0.42058253,  1.25996497],[0.01942838,  0.72351229,  0.38469833,  1.2203629 ]])

    #z_values_scal, scal = smart_scaling(z_values, 100)
    
    #print z_values_scal

    #mm = [[237512851739963],[14658622955356],[229124840490966],[205100029880339],[227934635643093],[17846780355862],[55389962772409],[263011659097926],[119115587787204],[241769783421970],[129441262599146],[144502896940715],[226371045488786],[112192969310834],[255391071669921],[8694635013588],[27044624173938],[84810117512235],[335280182744],[93713232175970],[15595059329443],[182451631363946],[133244670283093],[237060086616293]]
    
    #tt = splitm(3, 16, 65535,mm)

    #I need to test all functions for all possible combinations of list types and make sure they work properly and give always the same output

    #m = [[1000,2000],[2000,3000]]
    #res = enc_gains(sk, m)

    #OMFG = hom_mul_mat(res, tt)

    F = [[10, 20, 30], [3, 4, 5]]

    #F = [[1,54]]
    x = [100, 100, 100]

    
    cF = enc_2_mat(var.p, var.L, var.q, var.r, var.N, sk, F)
    
    cx = enc_1(var.p, var.L, var.q, var.r, var.N, sk, x)
    
    #cx22 = enc1_to_enc2(var.p, var.L, var.q, var.r, var.N, cx)
    start_matmult = time.time()
    cFcx = hom_mul_mat(var.q,var.N, cF,cx) #[[1,2],[3,4]]*[1,2] is [[1,2],[3,4]]*[[1],[2]]
    end_matmult = time.time()


    print("\n")
    print("Expected Mult Result: " + str(np.dot(np.array(F),np.array([[100], [100], [100]]))))
    print("Decrypted Matrix Mult Result: " + str(dec_hom(var.p, var.L, sk, cFcx)))

    time_matmult = end_matmult - start_matmult
    print("\n")
    print("Time Matrix Mult " + str(time_matmult)) #Fast

    to_pub = prep_pub_ros(var.q, var.N, [[1,2,3,4,5]])
    to_pub = prep_pub_ros(var.q, var.N,  cF)
    recovered = recvr_pub_ros(var.q, var.N, to_pub, 2, 3)

    



def mod_hom(x, p):
    '''modulus function that works with negative values using numpy'''

    try:
        length = (len(x))
    except:
        length = 1

    y = np.zeros((1,length), dtype = object)

    y = np.mod(x, p)

    y = np.where(y >= p/2, y-p, y)

    return y

    
def mod_hom2(x, p):
        try:
            length = (len(x))
        except:
            length = 1

        y = np.zeros((1,length), dtype = object)

        y = np.mod(x, p)
        
        return y

def variables_define(p = 10**4, L = 10**4, r=10**1, N = 5): 
    '''This function generates a csv file with the variables for homomorphic such that the different ROS nodes can access the same variables'''

    #mod_s = 3

    q = p * L

    #s_G = 2**19 #p

    #mod_e = 2**6+1 #r (should apparently be odd)


    n=1
    n_ = n+1 #This one extends the number as a vector so it's similar to N
    
    nu = 16
    nu2 = 2**nu-1


    signal_bound =4
    
    #s_G=p
    import math
    mod_g = 2**math.ceil(math.log(2*signal_bound*L*p,2))
    mod_q = int(mod_g) #same as q
    d = math.ceil(math.log(mod_q,2)/nu)
    mod_qe = mod_q-math.ceil(r/2)
    mod_qg = mod_q/mod_g
    mod_q1 = mod_q - 1

    PATH = os.path.dirname(os.path.abspath(__file__))
    FILEPATH = os.path.join(PATH, 'variables.csv')

    with open(FILEPATH, "w") as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerow([p])
        writer.writerow([L])
        writer.writerow([q])
        writer.writerow([r])
        writer.writerow([N])
        #writer.writerow([mod_s])
        writer.writerow([signal_bound])
        writer.writerow([nu])
        writer.writerow([nu2])




def variables_import(ros_package = 'nexhom'):
    '''This function imports the variables from the csv file'''

    class variables: #This creates an object to store the variables in
        def __init__(self): 

            #rospack = rospkg.RosPack()
            PATH = os.path.dirname(os.path.abspath(__file__)) #rospack.get_path(ros_package)
            FILEPATH = os.path.join(PATH, 'variables.csv')

            with open(FILEPATH, 'rb') as f: #This section reads the csv file to gather the private key to encrypt
                reader = csv.reader(f, delimiter='\n')
                lis = list(reader)
                self.p=eval(lis[0][0])
                self.L=eval(lis[1][0])
                self.q=eval(lis[2][0])
                self.r=eval(lis[3][0])
                self.N=eval(lis[4][0])

    var = variables()

    return var


def key_generate(q, L, N, robot):
    '''This function generates a secret key to encrypt and saves it into a csv file'''
    
    rand = [random.randrange(q*L) for iter in range(N)]

    sk = mod_hom(rand, q*L)

    PATH = os.path.dirname(os.path.abspath(__file__)) #rospack.get_path(ros_package)
    FILEPATH = os.path.join(PATH, 'private_key_'+str(robot)+'.csv')

    with open(FILEPATH, "w") as output:
        writer = csv.writer(output, delimiter=',')
        for val in sk:
            writer.writerow([val])



def key_import(robot, ros_package = 'nexhom'):
    '''This function imports the key created into a csv file by the "key_generate" function'''
    #rospack = rospkg.RosPack()
    #PATH = rospack.get_path(ros_package)

    sk = []

    PATH = os.path.dirname(os.path.abspath(__file__)) #rospack.get_path(ros_package)
    FILEPATH = os.path.join(PATH, 'private_key_'+str(robot)+'.csv')

    with open(FILEPATH, 'rb') as f: #This section reads the csv file to gather the private key to encrypt
        reader = csv.reader(f, delimiter='\n')
        lis = list(reader)
        for i in lis:
            sk.append(eval(i[0]))

    return sk



def enc_1(p, L, q, r, N, secret_key, m): #This function encrypts the values into a vector (use this to sum homomorphically)
    '''
    This function will encrypt scalars, if fed a list it will encrypt each integer individually, to encrypt matrices refer to enc_mat
    '''
    
    n = len(m)

    q_float = float(q)

    #A = np.random.uniform(low = 0, high = q_float, size=(n, N))
    A = np.random.uniform(low = 0, high = q_float, size=(n, N)).astype(object) #np.around(np.random.uniform(low = 0, high = q_float, size=(n, N))).astype(object)
    
    for i in range(0,len(A)):
        A[i] = np.array(map(long, A[i]), dtype = object) 
    #A = np.array([map(long,i) for i in A], dtype = object)

    #ALISTLONG = [long(i) for i in A.tolist()[0]]
    #A_list = [[long(i) for i in inner] for inner in A]
    #A = np.array(A_list, dtype=object)
    
    #A = np.array(ALISTLONG, dtype=object)[np.newaxis]
    
    m = np.array(m)[:,np.newaxis]

    dum = np.random.randint(low = 1, high = r, size = [n,1])
    
    e = mod_hom(dum, r)

    #b = np.zeros((n,1), dtype = object)

    secret_key_np = np.array(secret_key, dtype = object)[:,np.newaxis]

    b = np.dot(-A, secret_key_np)

    #b = b[np.newaxis]

    add = np.zeros((1, e.shape[0]), dtype = object)
    
    
    add = np.add(L*m,e)

    b = b+add

    A = np.append(b, A, axis =1)

    
    #ciphertext = np.zeros((n,N+1), dtype = object)

    ciphertext = mod_hom(A, q)

    #print(ciphertext)

    return [map(long,i) for i in ciphertext] #ciphertext.tolist()


def enc_2(p, L, q, r, N, secret_key, m): #This function encrypts the values to be homomorphically multiplied
    
    lq = int(math.log10(q))
    
    R = np.kron(np.power(10, np.arange(lq, dtype = object)[:, np.newaxis]), np.eye(N+1, dtype = object))

    mat_zeros_np = np.zeros(lq*(N+1), dtype = object)
    
    #mat_zeros = mat_zeros_np.tolist()

    #start_zeros_np = time.time()
    mat_zeros_enc = enc_1(p, L, q, r, N, secret_key, mat_zeros_np)
    #end_zeros_np = time.time()
    #print "Time for zero encryption NUMPY" + str(end_zeros_np - start_zeros_np) 
    
    #start_zeros = time.time()
    #mat_zeros_enc = enc_1(p, L, q, r, N, secret_key, mat_zeros)
    #end_zeros = time.time()
    #print "Time for zero encryption " + str(end_zeros - start_zeros) 

    #start_zeros1 = time.time()
    #mat_zeros_enc2 = enc_matlab(p, L, q, r, N, secret_key, mat_zeros)
    #end_zeros1 = time.time()
    #print "Time for zero encryption MAtlab" + str(end_zeros1 - start_zeros1) 

    hold_1 = m*R
    
    hold = np.add(hold_1, mat_zeros_enc)

    #ciphertext = np.empty((len(mat_zeros_enc[0]), len(mat_zeros_enc)), dtype = object)
    
    ciphertext = mod_hom(hold, q)

    return ciphertext.tolist()

def enc_2_mat(p, L, q, r, N, secret_key, m): #This function encrypts the values of a matrix to be homomorphically multiplied
    n1 = len(m)
    n2 = len(m[0])

    cA = np.zeros((n1, n2, int(math.log10(q)*(N+1)), N+1)).astype(int).tolist()

    for i in range(n1):
        for j in range(n2):
            cA[i][j] = enc_2(p, L, q, r, N, secret_key, [m[i][j]])
            #could just use the insert syntax instead of assigning the whole matrix

    return cA



def dec_hom(p, L, secret_key, c):
    '''This function decrypts a homomorphically encrypted value using NUMPY'''

    s_np = np.array(secret_key)
    c_np = np.array(c, dtype = object)

    s_np = np.append(1, s_np)[:,np.newaxis] #append "1" to the secret key
    #s_np = s_np.T

    #plaintext_np = np.zeros([1,len(c)], dtype = object)
    
    dot = np.dot(c_np, s_np)

    plain = mod_hom(dot, L*p)

    plaintext_np = plain.astype(float)/L

    if type(plaintext_np) != float:
        plaintext_np = np.around(plaintext_np).astype(int)
    else:
        plaintext_np = round(plaintext_np)
        plaintext_np = int(plaintext_np)


    #print(plaintext_np.tolist())
    return plaintext_np.tolist()


def decomp(q, c1): #function to carry out before multiplying used by the function "hom_mul"
    
    lq_np = int(math.log10(q))

    #c1_np = mod_hom2_np(c1, q)
    #print "ERROR c1 " + str(c1)

    c1_np = np.mod(c1, q)

    BBB=np.zeros((c1_np.shape[0],0), dtype = object)

    for i in range(lq_np):

        dum = 10**(lq_np-1-i)

        Q_np = np.mod(c1_np, dum)
        
        Q_np = np.subtract(c1_np, Q_np)

        BBB = np.append(Q_np/dum, BBB, axis= 1)

        #c1 = [j - k for j, k in zip(c1,Q)]
        
        c1_np = np.subtract(c1_np,Q_np)

    return BBB

def prep_pub_ros(q, N, c):
    '''used to flatten lists to publish in ROS'''

    col = N+1

    c = np.array(c, dtype = np.int64).tolist() #ADDED TO ENSURE VALUES GOING TO ROS ARE UP TO INT64
        
    row = len(c)

    while isinstance(c[0], list):
        c = [item for sublist in c for item in sublist]

    return c

def prep_pub_ros_str(c):
    '''used to convert a list to a string to publish in ROS'''

    string = json.dumps(c)
    
    return string

def recvr_pub_ros_str(c):
    '''used to convert a list to a string to publish in ROS'''

    pub_list = json.loads(c)
        
    return pub_list

def recvr_pub_ros(q, N, c, row=2, col=2):
    '''used to recover flattened lists created with prep_pub_ros to publish in ROS'''

    len_enc2 = int(math.log10(q))*(N+1)

    c_rebuilt=[]
    for i in range(0, len(c), N+1):
        c_rebuilt.append(list(c[i:(N+1)+i]))
    
    if len(c_rebuilt) > len_enc2: #check if it's a matrix to be recovered
        c_rebuilt_mat = [[] for i in range(row)] 
        k = 0
        for i in range(row):
            for j in range(col):
                c_rebuilt_mat[i].append(c_rebuilt[(k)*len_enc2:(k+1)*len_enc2]) 
                k += 1

        c_rebuilt = c_rebuilt_mat #prepare matrix to be outputed
    
    return c_rebuilt



def hom_mul(q, c1, c2):
    ''' This function performs the multiplication of two homomorphically encrypted values, c2 must be encrypted using the function "enc2" using NUMPY '''
    
    c1_np = np.array(c1, dtype = object)
    c2_np = np.array(c2, dtype = object)

    c1_np = decomp(q, c1_np)

    #x = np.zeros([1, c2_np.shape[1]], dtype = object)
    
    x = np.dot(c1_np, c2_np) 

    return x.tolist()[0]


def hom_mul_mat(q, N, c1, c2):
    ''' This function performs the multiplication of a homomorphically encrypted matrix with a vector, c2 must be encrypted using the function "enc2" '''
    
    n4 = len(c1)
    n3 = len(c1[0])
    #n2 = len(c1[0][0])
    #n1 = len(c1[0][0][0])

    #c1 = decomp(q, c1)

    #Mm=[[0]*(N+1)]*n4

    Mm_np = np.zeros((n4,N+1), dtype = object)

    #multiplied = [0]*len(Mm[0])
    #multiplied_np = np.zeros((1,n1), dtype = object)


    dec_c2 = decomp(q, c2)
    
    for i in range(n4):
        for j in range(n3):
            #temp1 = Mm_np[i]
            temp2 = dec_c2[j]
            temp3 = c1[i][j]

            multiplied = np.dot(temp2, temp3)

            added = Mm_np[i] + multiplied

            Mm_np[i] = mod_hom(added,q)


    return Mm_np.tolist()

def smart_scaling(input, scal = 100):
    '''
    This function scales it's input (z_values) until all values reach a minimum amount set by the input "scal"
    @param input: matrix to be scaled an array of arrays: np.array([[1,2],[1,2]])
    @param scal: lowest value desired (usually a power of 10)
    @return: the input scaled up
    '''

    low = abs(input) < 0.0001 #This is to take away values that are too small that would otherwise break this process
            
    input[low] = scal #replace the values that are to small with the scal value to avoid the algorithm trying to scale them up

    values = [] #initialize list to save the scaling values

    for i in range(len(input[0])):
        store = 1 #initialize variable to record the amount scaled up

        while np.any(abs(input[:, i])<=scal):
            
            input[:, i] *= 10

            store *= 10 

        values.append(store)

    input[low] = 1 #replace the undesired small values previously identified by 1

    if min(values[1:3]) == max(values[1:3]):
        pass

    else:
        
        #index_max = values[1:3].index(max(values[1:3]))+1 #middle values must be scaled to the same amount

        high = max(values[1:3])

        low = min(values[1:3])
        
        index_min = values[1:3].index(low)+1

        difference = high/low

        input[:, index_min] *= difference

        values[index_min] = high

    scaling = values[0]*values[1]*values[2]

    return input, scaling

    ##THIS FUNCTION COULD HAVE A PROBLEM WITH THE VARIABLE BEING NAMED "INPUT"


def log_scaling(vk, sp_vk):
    '''
    This function scales it's input (z_values) until all values reach a minimum amount set by the input "scal"
    @param input: matrix to be scaled an array of arrays: np.array([[1,2],[1,2]])
    @param scal: lowest value desired (usually a power of 10)
    @sp_vk = desired significant figures
    '''
    #if type(vk) == float:
    vk = np.array(vk, dtype = np.float)[np.newaxis]
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

    ##THIS FUNCTION COULD HAVE A PROBLEM WITH THE VARIABLE BEING NAMED "INPUT"


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