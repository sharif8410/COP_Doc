#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:20:26 2022

@author: sharif-al-mahmud
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from copy import deepcopy
rng = np.random.default_rng()
import random


def Update_Res(R,GG,PP):
    for s in range(len(GG)): #Updating Residual within the while loop
        R=R-np.dot(GG[s],PP[s])
    return R




def Length(g_i_j,Y_b,U,e):
    l_i = e+ np.dot(g_i_j,Y_b)/U
    return l_i


# ## H1 Function Modified from Tsao et al. 2020  (-suggested by Dr. Weyers) 

def H1(Q,Y,Pm,U,l_max,e):
    ''' 
    P= [P_min,P_max]
    '''
    if len(Q)!=len(Y):
        raise ValueError('number of sizes and number of fabric consumption does not match')
        
    r_b = deepcopy(Q)
    Y_b = Y
    
    G_a_b=[]
    P_a=[]
    P_min,P_max= Pm
    Beta=len(Y_b)
    K = range(Beta)
    while max(r_b)>0:

        #print(f'\nresidual demand:{r_b}')
        p_i = max(P_min,min(min(r_b),P_max))
        #print('p_i=',p_i)

        g_i_j= list (np.zeros(Beta))   # empty list containg Beta values, Beta=Sizes, Beta
        l_i=0                       #inital length = 0

        K= list(range(Beta))
        random.shuffle(K)  # Random size
        #print(f'shuffled sizes: {1}',K)

        for j in K:
            if r_b[j] >= P_min:

                if l_i <= l_max:
                    temp_g_i_j= g_i_j.copy()
                    temp_g_i_j[j] = math.floor(r_b[j]/p_i)
                    # print (temp_g_i_j)
                    temp_l_i= Length(temp_g_i_j,Y_b,U,e)
                    # print (temp_l_i)

                    if temp_l_i <= l_max:
                        g_i_j=temp_g_i_j
                        l_i= temp_l_i
                        #print ("xx")
                    else:
                        g_i_j[j] = math.floor((l_max-l_i)/(Y_b[j]/U))
                        l_i = Length(g_i_j,Y_b,U,e)
                        #print("yy") 

                else:
                    g_i_j[j] = 0
                #print (g_i_j,l_i,'---')

            elif r_b[j] > 0:
                if l_i <= l_max:
                    temp_g_i_j= g_i_j.copy()
                    temp_g_i_j[j]= math.ceil(r_b[j]/p_i)
                    #print (temp_g_i_j)
                    temp_l_i= Length(temp_g_i_j,Y_b,U,e)
                    #print (temp_l_i)
                    if temp_l_i <= l_max:
                        g_i_j=temp_g_i_j
                        l_i= temp_l_i
                        #print ("xxx")
                    else:
                        g_i_j[j]= math.floor((l_max-l_i)/(Y_b[j]/U))
                        l_i = Length(g_i_j,Y_b,U,e)
                        #print("yyy")
                    #print (g_i_j,l_i)
                else:
                    g_i_j[j]= 0
                #print (g_i_j,l_i,'+++')

            else:
                #print("***")
                g_i_j[j]=0

            #print (g_i_j,l_i,f'=={j}==')
            #print ('--------')
        r_b= r_b-np.dot(g_i_j,p_i)
        G_a_b.append (g_i_j)
        P_a.append(p_i)
        #print ('---------------------------')

    Sol= dict(G=G_a_b,P=P_a)
    return Sol




# ## H3 Function Modified from M&B 2016  (-suggested by Dr. Weyers) 

def H3 (Q,Y,Pm,U,l_max,e):
    '''
    P=[P_min,P_max]
    '''
    if len(Q)!=len(Y):
        raise ValueError('number of sizes and number of fabric consumption does not match')
    
    P_min,P_max=Pm
    r = deepcopy(Q)
    Y_b=Y
    Beta=len(Y_b)
    
    
    
    '''
    Global Variables: 
        Beta= sizes;    l_max= Maximum length, 
        Y_b= fabric consumption rate list; 
        U= Marker efficiency
    
    Another used defined function used:
    def Length(g_i_j,Y_b,U,e):
        return e+np.dot(g_i_j,Y_b)/U
    
    '''
    G=[]
    P=[]
    while max(r)>0:
        
        p_i = max (np.gcd.reduce(r),P_min)  #GCD

        g= list (np.zeros(Beta))   # empty list containg Beta values, Beta=Sizes, betas
        l=0                       #inital length = 0

        K= list(range(Beta))
        random.shuffle(K)  # Random size

        for j in K:
            if r[j] >0:
                if (l_max-l) >= (Y_b[j]/U): # check if there is place to put at least 1 garments of size j

                    temp_g= g.copy()
                    temp_g [j] =  math.ceil(r[j]/p_i)
                    temp_l = Length(temp_g,Y_b,U,e)

                    if temp_l <= l_max:
                        g = temp_g
                        l= temp_l
                    else:
                        g[j] = math.floor((l_max-l)/(Y_b[j]/U))
                        l = Length(g,Y_b,U,e)

                else:
                    g[j]=0
                    
            else:
                g[j]=0

        r= r-np.dot(g,p_i)
        G.append (g)
        P.append(p_i)
    return dict(G=G,P=P)

# Sol_2=H3(Q=Q,Y=Y,Pm=[P_min,P_max],U=u,l_max=L_max,e=E)
# print(Sol_2)



# ## H5 Function - Using Random sizes ( suggested by Dr. Weyers)

def H5(Q,Y,Pm,U,l_max,e):
    '''
    Pm=[P_min,P_max]
    '''
    
    if len(Q)!=len(Y):
        raise ValueError('number of sizes and number of fabric consumption does not match')
    
    P_min,P_max = Pm  #Unpack
    
    i=1
    r = deepcopy(Q)
    Y_b=Y
    Beta= len(Y_b)
    m_max = 2*Beta
    G=[]
    P=[]
    
    while max(r)>0:
        for j in range(Beta):
            if r[j]>0 and r[j]<P_min:
                r[j]=P_min

        #print(f'\nresidual demand:{r}')

        rng = np.random.default_rng()   # Random number Generator object

        p_i = rng.integers(max(P_min,1),min(P_max,max(P_min,max(r))),endpoint=True)
        #print('p_i=',p_i)

        g_i_j= list (np.zeros(Beta))   # empty list containg Beta values, Beta=Sizes, betas
        l_i=0                       #inital length = 0

        K= list(range(Beta))
        random.shuffle(K)  # Random size
        #print(f'shuffled sizes: {K}')

        for j in K:
            temp_g_i_j= g_i_j.copy()
            temp_g_i_j[j]=rng.integers(0,math.floor(r[j]/p_i)+1)
            temp_l_i= Length(temp_g_i_j,Y_b,U,e)
            #print('t-g',temp_g_i_j)
            if temp_l_i <= l_max:
                g_i_j= temp_g_i_j
                l_i= temp_l_i
            elif temp_l_i > l_max:
                g_i_j[j]= rng.integers(0,math.floor((l_max-l_i)/(Y_b[j]/U))+1)
                l_i= Length(g_i_j,Y_b,U,e)
            else:
                g_i_j[j]=0
                l_i= Length(g_i_j,Y_b,U,e)                

            #print (g_i_j,l_i,f'=={j}==')
            #print ('----')

        r= r-np.dot(g_i_j,p_i)
        G.append (g_i_j)
        P.append(p_i)
        #print (f'-----------{i}----------------')

        i+=1
        if i >= m_max and max(r)>0:
            #print ('H3')
            h3=H3(Q=r,Y=Y_b,Pm=[P_min,P_max],U=U,l_max=l_max,e=e)  # Use H3 algorithm to pack all sizes
            g,p = h3.values()    
            #print(g,p)
            G=G+g
            P=P+p
            i=len(P)+1
            break
    return dict(G=G,P=P)

def main():
    print("main function")
    ##Data
    Q_b = [18,172,214,254,227,187,187,79]   # b=beta
    Y_b = [1.14,1.18,1.215,1.225,1.247,1.26,1.275,1.285] # Fabric yield rate per garment of size 𝛽
    U = 0.85 # Marker Utilization Rate
    l_max= 20 #max marker length
    e= .07   # Fabric end allowance  
    P_min,P_max= 10,350 #min and max number of ply in one section
    
    Sol_1=H1(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,l_max=l_max,e=e)
    Sol_2=H3(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,l_max=l_max,e=e)
    Sol_3=H5(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,l_max=l_max,e=e)
    print(Sol_1)
    print(Sol_2)
    print(Sol_3)

if __name__ == "__main__":
    main()
