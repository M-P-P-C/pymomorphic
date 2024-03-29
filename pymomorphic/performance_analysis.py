#!/usr/bin/env python3

import sys
import os
import warnings

import csv
import random
import math
from operator import add

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pymomorphic_py2 as pymorph2
import pymomorphic_py3 as pymorph3

import time





#add check to ensure inputted numbers are (integers) and warning

def main():
    
    my_key = pymorph3.KEY(p = 10**11, L = 10**3, r=10**1, N = 5)

    import timeit

    #timeit.timeit(xxx.process_test)

    logarithmic_quantizer(0.01, 2)

    measure_performance_enc('N')



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
    


def measure_performance_enc(self, test_var = 'N', test_func = ["ENC1", "ENC2", "DEC"], m = [380], range_val=[1,210], iter = 20, step = 20):
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

                my_key =pymorph3.KEY(p = 10**13, L = 10**3, r=10**1, N = i)

            #print("\n")
            #print("Round: " + str(j)+","+str(i))

            if "ENC1" in test_func: 
                start_enc1 = time.time()
                ciphertext = my_key.encrypt(m) 
                end_enc1 = time.time()

            if "ENC2" in test_func: 
                start_enc2 = time.time()
                ciphertext2 = my_key.encrypt2(m)
                end_enc2 = time.time()

            if "DEC" in test_func: 
                start_dec = time.time()
                decrypted = my_key.decrypt(ciphertext)
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
    
    s = pd.Series(range(range_val[0],range_val[1],step))
    dataf = dataf.set_index([s])
    data_N = dataf

    filter_enc1 = [col for col in data_N if col.startswith('Enc1')]
    filter_enc2 = [col for col in data_N if col.startswith('Enc2')]
    filter_dec = [col for col in data_N if col.startswith('Dec')]

    df_1 = data_N[filter_enc1].transpose().melt()
    df_2 = data_N[filter_enc2].transpose().melt()
    df_4 = data_N[filter_dec].transpose().melt()

    #df_1['variable'] = ((df_1['variable']+1)*range_val[1]/10)-1
    #df_2['variable'] = ((df_2['variable']+1)*range_val[1]/10)-1
    #df_4['variable'] = ((df_4['variable']+1)*range_val[1]/10)-1

    #dataf.to_csv(path_or_buf='Ncrypt_200.csv')

    plot_N_fig(df_1, df_2, df_4)

def plot_N_fig(df_1, df_2, df_4):


    #plt.grid(color='w', linestyle='--', linewidth=2)

    N_max = 210#int(df_1["variable"][3999])+10#100

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

    y_lim = [0, 0.8]
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


def logarithmic_quantizer(value, sig_fig):
    """ 
    Plot logarithmic quantizer equation

    Parameters
    ----------
    value : float 
        value to be scaled up to integer
    sig_fig : int
        desired significant figures to plot for
    """
    
    #TODO: Tidy up code from Matlab

    ## Define Colors for Plots
    str1 = '#6699CC' #Light Blue
    str3 = '#000075' #Dark Blue
    str4 = '#000075' #Green

    ## Setting Up Variables
    sp = [1,3] #Significant Figures
    interv = 0.0001 #Step size for nu
    scale = 2
    nu = np.arange(interv*10**(scale),10**scale, interv) #I set the range of nu to start at a larger point than the step size to avoid funky results
    #nu = -10^2:interv:10^2; % range with negative values

    # Initialized some arrays
    S = np.zeros([np.size(nu), np.size(sp)]); #array for scaling factor S
    nu_hat = np.zeros([np.size(nu), np.size(sp)]); #array for scaled nu
    S_r = np.zeros([np.size(nu), np.size(sp)]); #array for the reciprocal of S


    ## Calculating Logarithmic Scaling Function
    
    for j in range(1,np.size(sp)):
        for i in range(1,np.size(nu)):
            S[i, j] = 10**(sp[j]-np.floor(np.log10(abs(nu[i])))-1)

            nu_hat[i, j] = round(S[i, j]*nu[i])

            S_r[i, j] = S[i, j]**(-1)*nu_hat[i, j]
    
    
    ## Plotting Figures

    fig, ax1 = plt.subplots()

    ax1.semilogx(nu, nu_hat[:, 1],linewidth = 2, color = str4) #PLOT \hat\hat\nu
    ax1.set_ylabel(r'$\bar{\bar{\nu}}_k$', fontsize=13)
    ax1.set_ylim(100, 1000)
    ax1.set_yticks([1,2,3,4,5,6,7,8,9,10])

    ax2 = ax1.twinx()
    ax2.semilogx(nu, nu_hat[:, 0],linewidth = 2, color = str1) #PLOT \hat\hat\nu
    ax2.set_ylabel(r'$\bar{\bar{\nu}}_k$', fontsize=13)
    ax2.set_ylim(1, 10)
    ax2.set_yticks([100,200,300,400,500,600,700,800,900,1000])

    ax1.legend([r'$sp_{\nu_k}=1$', r'$sp_{\nu_k}=3$'], loc = 'upper left')

    fig.show()

    '''
    ############ PLOT FORMATTING STUFF ############
    x1 = [nu(1), nu(end)]%xlim()
    y1 = ylim()
    subplot1.FontSize = 16;
    xlabel('$\nu_k$', 'fontsize', 13);
    xlim(subplot1,x1) 
    xticks([10^(-2), 10^(-1),10^(0), 10^(1), 10^(2)])
    ylim(subplot1,y1) 

    ax = gca;
    ax.YAxis(1).Color = color;
    ax.YAxis(2).Color = color4;
    ##################################################


    subplot2 = subplot(1,2,2);

    loglog(nu,S_r(:, 1),'LineWidth',2,'Color', color) #PLOT S^{-1}*\hat\hat\nu
    hold on
    loglog(nu,S_r(:, 2),'LineWidth',2,'Color', color4) #PLOT S^{-1}*\hat\hat\nu


    legend('$sp_{\nu_k}=1$', '$sp_{\nu_k}=3$',  'Location', 'northwest')

    ########## PLOT FORMATTING STUFF ##########
    subplot2.FontSize = 16; 
    y2 = ylim()
    hold on
    #a = line(xlim(), [1,1], 'LineWidth', 1, 'Color', 'k', 'LineStyle','--');
    #b = line([1,1], ylim(), 'LineWidth', 1, 'Color', 'k', 'LineStyle','--');
    ylabel('$S_{\nu_k}^{-1}\bar{\bar{\nu_k}}$', 'fontsize', 12);
    xlabel('$\nu_k$', 'fontsize', 12);
    ylim(subplot2,y2) 
    xticks([10^(-2), 10^(-1),10^(0), 10^(1), 10^(2)])
    grid on
    ############################################
    '''
    ## Scaling Equations

    sp = 1
    value = 130

    S_v = 10**(sp-np.floor(np.log10(abs(value)))-1)

    scaled_v = np.round(S_v*value)

    div = scaled_v/S_v 

    error = value-div

if __name__ == '__main__':
    main()