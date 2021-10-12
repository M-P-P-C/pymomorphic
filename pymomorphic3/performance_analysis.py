#!/usr/bin/env python3

import sys
import warnings

import csv
import random
#import rospkg
import numpy as np
import os
import math
from operator import add

import myfunctions2 as pymorph

import time
import json #used to convert list to string, used to send messages through ROS, as it does not support values larger than int64

import pandas as pd



#add check to ensure inputted numbers are (integers) and warning





def measure_performance_enc(self, test_var = 'N', test_func = ["ENC1", "ENC2", "DEC"], m = [380], range_val=[1,50], iter = 20, step = 5):
    """ 
    Measure perfomance of encryption and decryption functions

    Parameters
    ----------
    test_var = string
        determine which variable to test for. Only allows one of the following inputs : {"p", "L", "r", "N", "m"}
    range_val = list [x,y]
        sets the range of values to test for, with x as the minimum, and y as the maximum
    iter : int
        sets the amount of times to iterate operations
    step : int
        sets the steps between values to be calculated
    """

    #TODO: set this function outside of class so there is no need to modify the self argument

    pymorph.KEY(p = 10**13, L = 10**3, r=10**1, N = 50)

    #store = [self.p, self.L, self.q, self.r, self.N, self.secret_key] 

    valid_var = {'p', 'L', 'r', 'N', 'm'} #dict of valid inputs for argument test_vat
    valid_func = {"ENC1", "ENC2", "DEC"} #dict of valid inputs for argument test_vat

    test_func = [i.upper() for i in test_func] #capitlizes all inputs in test_func to avoid case sensitive issues

    if test_var not in valid_var: #check the input for test_var is valid
        raise ValueError("results: test_var must be one of %r." % valid_var)

    for i in [0,1,2]:
        if test_func[i] not in valid_func: #check the input for test_func is valid
            raise ValueError("results: test_func must be one of %r." % valid_func)

    

    dataf = pd.DataFrame() #initialize dataframe to store values from test.
    time_enc1 = pd.DataFrame()
    time_enc2 = pd.DataFrame()
    time_dec = pd.DataFrame()

    for j in range(iter):
        for i in range(range_val[0],range_val[1],step):
            

            if test_var == "N":
                self.N = i
                self.secret_key = self.key_generate() #generate secret key for encryption.
                self.secret_key_np = np.array(self.secret_key, ndmin=2, dtype = object).T
                self.secret_key_np_dec = np.append([[1]], self.secret_key_np, axis=0)

            #print("\n")
            #print("Round: " + str(j)+","+str(i))

            if "ENC1" in test_func: 
                start_enc1 = time.time()
                ciphertext = self.encrypt(m) 
                end_enc1 = time.time()

            if "ENC2" in test_func: 
                start_enc2 = time.time()
                ciphertext2 = self.enc_2(m)
                end_enc2 = time.time()

            if "DEC" in test_func: 
                start_dec = time.time()
                decrypted = self.decrypt(ciphertext)
                end_dec = time.time()

            time_enc1 = time_enc1.append([end_enc1 - start_enc1], ignore_index=True)
            time_enc2 = time_enc2.append([end_enc2 - start_enc2], ignore_index=True)
            time_dec = time_dec.append([end_dec - start_dec], ignore_index=True)

        functions=['Enc1', 'Enc2', 'Dec']
        temp_df = pd.concat([time_enc1,time_enc2,time_dec], axis =1)

        temp_df.columns = functions
        dataf = pd.concat([dataf,temp_df], axis = 1)
        time_enc1 = pd.DataFrame()
        time_enc2 = pd.DataFrame()
        time_dec = pd.DataFrame()
    
    data_N = dataf

    filter_enc1 = [col for col in data_N if col.startswith('Enc1')]
    filter_enc2 = [col for col in data_N if col.startswith('Enc2')]
    filter_dec = [col for col in data_N if col.startswith('Dec')]

    df_1 = data_N[filter_enc1].transpose().melt()
    df_2 = data_N[filter_enc2].transpose().melt()
    df_4 = data_N[filter_dec].transpose().melt()

    df_1['variable'] = ((df_1['variable']+1)*10)-9
    df_2['variable'] = ((df_2['variable']+1)*10)-9
    df_4['variable'] = ((df_4['variable']+1)*10)-9

    #dataf.to_csv(path_or_buf='Ncrypt_200.csv')

    self.plot_N_fig(df_1, df_2, df_4)

def plot_N_fig(self, df_1, df_2, df_4):


    #plt.grid(color='w', linestyle='--', linewidth=2)

    N_max = 100

    import matplotlib.pyplot as plt
    import seaborn as sns


    #plt.rcParams.update({'text.usetex': True}) #Set Matplotlib to user the Latex Interpeter
    
    sns.set_color_codes()

    sns.set_theme(font_scale = 1.4, style="whitegrid") #Set seaborn text larger and it's main style

    #sns.set_style("whitegrid", {'grid.linestyle': '--', 'text.usetex': True, 'font.family': "serif", 'font.serif': ["Computer Modern Roman"]}) #override some of the style's settings


    fig, ax= plt.subplots(figsize=(8, 5))

    plt.figure(1)

    enc1_plt  = sns.lineplot(x="variable" , y="value" ,data=df_1[["variable", "value"]], label=r'$Enc$')
    enc2_plt = sns.lineplot(x="variable" , y="value" ,data=df_2[["variable", "value"]], label=r'$Enc2$')
    #mult_plt = sns.lineplot(x="variable" , y="value" ,data=df_3[["variable", "value"]], label=r'$Mult$')
    dec_plt = sns.lineplot(x="variable" , y="value" ,data=df_4[["variable", "value"]], label=r'$Dec$')

    y_lim = [0, 0.1]
    x_lim = [0, N_max]

    enc1_plt.get_children()[0].set_color('lightblue')
    enc1_plt.set(ylim = (y_lim[0],y_lim[1]), xlim=(x_lim[0],x_lim[1]))
    #ax.get_xticklabels().set_fontsize(18)

    #ax2.get_children()[0].set_color('lightgreen')
    enc2_plt.set(ylim = (y_lim[0],y_lim[1]), xlim=(x_lim[0],x_lim[1]))
    #mult_plt.set(ylim = (y_lim[0],y_lim[1]), xlim=(x_lim[0],x_lim[1]))
    dec_plt.set(ylim = (y_lim[0],y_lim[1]), xlim=(x_lim[0],x_lim[1]))

    #ax.get_children()[0].set_hatch('//')
    plt.setp(enc1_plt.collections[0], alpha=0.65)
    plt.setp(enc2_plt.collections[0], alpha=0.65)
    #plt.setp(mult_plt.collections[0], alpha=0.65)
    plt.setp(dec_plt.collections[0], alpha=0.65)


    #ax.set_title(r'Analysis of Encrypted Formation From 50 Samples') 
    ax.set_ylabel(r'Time [$s$]')
    ax.set_xlabel(r'$N$') 
    enc1_plt.legend()
    
    #ax.xaxis.get_label().set_fontsize(18)
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


    #PLOT.annotate("Sample Size: 50",xy=(2, 1), xytext=(10.5,1.9))
    #plt.savefig('dist_N40.svg', format='svg')
    #plt.savefig('dist_N40.eps', format='eps')
    
    #plt.savefig('Encryption_Performance.pdf', format='pdf', bbox_inches='tight')

    '''plt.rcParams.update({
    # Use LaTeX to write all text
    'text.usetex': True,
    'font.family': "serif",
    'font.serif': ["Computer Modern Roman"],
    ## Use 10pt font in plots, to match 10pt font in document
    #'axes.labelsize': 10,
    ## Make the legend/label fonts a little smaller
    #'legend.fontsize': 8,
    #'xtick.labelsize': 8,
    #'ytick.labelsize': 8
    })'''

    '''size_font_inside_the_plot = 14
    size_font_subplot_label = 20
    font_size_of_labels = 18
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size_of_labels)'''

    plt.show()

    from seaborn.categorical import _BarPlotter

    #test = _BarPlotter(data=data)


def main():
    
    my_key = pymorph.KEY(p = 10**11, L = 10**3, r=10**1, N = 5)

    import timeit

    #timeit.timeit(xxx.process_test)

    measure_performance_enc('N')
    #print "main function called, this function is used when trying to debug this script. It get's executed when running the file directly"
    variables_define(p = 10**15, L = 10**3, r=10**1, N = 50)


    
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





def plot_N_calc():

    dirpath = os.getcwd()

    data_N = pd.read_csv(dirpath+"\\Ncrypt_200.csv", delimiter=",")
    #data = data_table_init.transpose().values.tolist()

    filter_enc1 = [col for col in data_N if col.startswith('Enc1')]
    filter_enc2 = [col for col in data_N if col.startswith('Enc2')]
    filter_mult = [col for col in data_N if col.startswith('Mult')]
    filter_dec = [col for col in data_N if col.startswith('Dec')]

    df_1 = data_N[filter_enc1].transpose().melt()
    df_2 = data_N[filter_enc2].transpose().melt()
    df_3 = data_N[filter_mult].transpose().melt()
    df_4 = data_N[filter_dec].transpose().melt()

    df_1['variable'] = ((df_1['variable']+1)*10)-9
    df_2['variable'] = ((df_2['variable']+1)*10)-9
    df_3['variable'] = ((df_3['variable']+1)*10)-9
    df_4['variable'] = ((df_4['variable']+1)*10)-9

    return df_1, df_2, df_3, df_4

def dfassembly(path, variables, names):
    '''
    path should lead to the csv file
    variables should be the number of variables that are saved per run (rows per run)
    names should be a list of the variable names, e.g. ['time', 'er1', 'er2']
    '''

    PATH = path
    n = variables
    columns = names

    all_data=pd.DataFrame()
    
    sample_data=pd.read_csv(PATH,usecols=[0], header=None)
    skip=0
    for i in range(int(len(sample_data)/n)):

        temp_df = pd.DataFrame()
        temp_df = pd.read_csv(PATH, header=None, skiprows=skip,nrows=n) #which columns should be accessed
        temp_df = temp_df.T #transpose the DataFrame
        temp_df.columns = names #name the columns
        #temp_df["run"]=i'''
        temp_df.insert(0, "run", i) #add column that specifies which run it is #if it doesn't work uncomment previous line
        #'''
        temp_df.insert(0, "step", range(int(len(temp_df))))
        
        all_data = pd.concat([all_data,temp_df], axis = 1)

        skip += n
    
    return all_data

if __name__ == '__main__':
    main()