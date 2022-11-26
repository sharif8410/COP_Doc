#!/usr/bin/env python
# coding: utf-8

# In[1]:


demand = [990,1980,3961,2971,1980]   
d=0  # d% shortage allowance
Y_b = [1.3086,1.3671,1.4183,1.4538,1.5122]           # Fabric yield (consumption rate) rate per garment of size 𝛽

U = 0.85
l_max= 20
e= .07   # Fabric end allowance 
f= 2.90 # Fabric cost 

if len(demand)!=len(Y_b):
    raise ValueError('number of sizes and number of fabric consumption does not match')


# In[2]:


#Input variables (Marker)

M_d = 10                                # Average marker design time (minute)
z = 0.65                                # Printing speed per minute
v = 0.30                                #Standard cost per minute in marker making floor (labor, machine & electricity)


# In[3]:


#Input variables (Cutting Time)

T_G = 30                                  # General Preparation Time
x= .20           # Average spreading speed in minutes after taking account for the idle strokes. 
T_M= 2           # Time for Placement of the marker
t_c= 4.5         # SMV of cutting time per garment pattern
T_S= 5           # preparation time for sticker placement
𝑡_𝑏 = 2.837                               # Standard minute value (SMV) of time takes to bundle.
𝑏 =  15                                   # pieces of garments in one bundle
𝑤 = 0.20                                 # standard cost per minute in cutting floor (labor, machine & electricity)
P_min, P_max= 10,350


# In[4]:


import numpy as np
import math
import pandas as pd
from copy import deepcopy
rng = np.random.default_rng()
import random
import time
import matplotlib.pylab as plt
import plotly.express as px


# In[5]:


def Update_Res(R,GG,PP):
    for s in range(len(GG)): #Updating Residual within the while loop
        R=R-np.dot(GG[s],PP[s])
    return R


# In[6]:


def Length(g_i_j):
    l_i = e+ np.dot(g_i_j,Y_b)/U
    return l_i


# In[7]:


def Shortage_allowance(Q,d=0.01):
    temp=np.dot((1-d),Q)
    return [round(i) for i in temp]
Q_b= Shortage_allowance(demand,d)

# Q_b


# In[8]:


from Heuristics import H1,H3,H5


# In[9]:


# Sol_1 = H5(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
# Sol_1


# ## Objective Function

# In[10]:


def ObjectiveFunction (chromosome):
    
    temp_Chromosome=deepcopy(chromosome)
    G_a_b = temp_Chromosome['G']
    P_a = temp_Chromosome['P']
    Alpha = len(P_a)             # number of Sections
    

    '''                         Fabric Cost                      '''
        
    # Total fabric length = L # Total Fabric Cost = C_F

    l_a=[Length(G_a_b[alpha]) for alpha in range(Alpha) ] #Length function
    L= np.dot(l_a,P_a)       #Multiply then Sum
    C_F = L*f
    #print('Total Fabric Cost = C_F: ',C_F)
    
    
    '''                        Marker Cost                        '''
    
    #Marker Making Cost = C_M

    M_p_a = [(la-e)/z for la in l_a]     # devide each element of a 'l_a' by 'z'
    #M_p_a = Marker Printing time (minute) of section alpha 
 
    '''
    𝑟 = {1 ; 𝑖𝑓 𝑡h𝑒 𝑚𝑎𝑟𝑘𝑒𝑟 𝑖𝑠 𝑏𝑒𝑖𝑛𝑔 𝑢𝑠𝑒𝑑 𝑓𝑜𝑟 𝑡h𝑒 𝑓𝑖𝑟𝑠𝑡 𝑡𝑖𝑚𝑒 
        {0 ; 𝑖𝑓 𝑡h𝑒 𝑚𝑎𝑟𝑘𝑒𝑟 h𝑎𝑠 𝑏𝑒𝑒𝑛 𝑢𝑠𝑒𝑑 𝑏𝑒𝑓𝑜𝑟𝑒
    '''
    r=[]
    for i in range(Alpha):
        temp=0
        j=i-1
        while j>=0:
            if G_a_b[i]== G_a_b[j]:
                temp+=1
                break
            j-=1
        if temp==0:
            r.append(1)
        else:
            r.append(0)

    
    C_M = 0
    for α in range(Alpha):
        if l_a[α]>e:   # this makes sure that section has at least one garments 
            C_M += (M_d*r[α] + M_p_a[α])*v       
    
    # 'if la>e' makes sure that the section contain at least one garments, 
    #  not all G_a_b values are zero
    
    
    '''                          Cutting Cost                            '''
        
    # Cutting Time of one section = T_T  # Total Cutting Cost = C_C
    
    #T_T  =T_G + T_F +T_M+ T_c+T_S +T_B
    
    T_C=[]   #Cutting time for every section
    for alpha in range(Alpha):
        T_C.append(sum(G_a_b[alpha])*t_c)
    
    T_F=[]  # Fab spreading time for each section
    for α in range(Alpha):
        T_F.append(l_a[α]*P_a[α]/x)
    
    T_B=[] #Bundleing time for each section
    for α in range(Alpha):
        T_B.append(math.ceil(P_a[α]/b)*sum(G_a_b[α])*t_b)
    
    
    T_T_T = 0  #Total cutting time
    for α in range(Alpha):
        if l_a[α]>e:   # this makes sure that section has at least one garments
            T_T_T+=T_G+T_F[α]+T_M+T_C[α]+T_S+ T_B[α]
    
    
    C_C = T_T_T*w  #Total cutting cost
    
    
    
    '''                              Total Cost                 '''
    # Total Cost = C_T = C_F + C_M + C_C
    
    return C_F+C_M+C_C


# In[11]:


# ObjectiveFunction(Sol_1)


# ## Fitness Score

# In[12]:


def Fitness(chromosome): 
    
    
    
    t_chromosome=deepcopy(chromosome)
    G_a_b= t_chromosome['G']
    P_a = t_chromosome['P']
    Beta= len(demand)
    
    score= ObjectiveFunction(t_chromosome)
    #print('score:',score)
    fitness_score=score
    
                
    '''       Penalty for shortage production           '''
    R= Update_Res(R=demand,GG=G_a_b,PP=P_a)
    for beta in range(Beta):
        if R[beta]>0:
            s_penalty= R[beta]/sum(demand)
            fitness_score +=score*s_penalty 
    
    
    '''         Penalty for excess production           '''
    r=np.dot(1.02,demand)         # additional 2% allowance
    R= Update_Res(R=r,GG=G_a_b,PP=P_a)
    #print(R)
    for beta in range(Beta):
        if R[beta]<0:
            e_penalty= (-R[beta]/sum(demand))*2   # 2times than s_penalty
            fitness_score +=score*e_penalty   
            

    
    '''       double check if the solution is valid       '''
    res= Update_Res(R=Q_b,GG=G_a_b,PP=P_a)
    if max(res)>0:
        '''solution is unvalid'''
        fitness_score +=10000   #this will eventualy make the solution extinct.
    
    
    return fitness_score

# Fitness(Sol_1)    


# ## Function Initial Population Generation

# In[13]:


def GeneratePopulation(pop_size):
    P_of_S=[]
    for p in range(pop_size):
        rng = np.random.default_rng()
        h=rng.integers(0,3)
        #print('h:',h)
        
        if h==0:
            sol=H1(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
        elif h==1:
            sol=H3(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
        else:
            sol=H5(Q=Q_b,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
            
        P_of_S.append(sol)
    return P_of_S
# Pool_of_Sol= GeneratePopulation(100)
# print(Pool_of_Sol)


# In[14]:


def S_with_F(p_o_s):
    p_o_s_with_f= deepcopy(p_o_s)
    for i in range(len(p_o_s)): 
        if 'F' not in p_o_s[i]:
            p_o_s_with_f[i]['F']=Fitness(p_o_s[i])
    return p_o_s_with_f


# ## PSO

# ### Cleaning section with zeros

# In[15]:


def CleanZeros (Sol):
    Solution=deepcopy(Sol)
    j=0
    while j < len(Solution['G']):
        if max(Solution['G'][j])==0:
            Solution['G'].pop(j)
            Solution['P'].pop(j)
            continue
        j+=1

    #This is to make sure 
    if len(Solution['G'])!=len(Solution['P']):
        raise ValueError('P and G lengths are not same')
    
    return Solution


# In[16]:


# CleanZeros(Sol_1)


# ## Velocity Update (Jarboui et al. 2008)

# Lets assume 1st sol as X, 2nd Sol as Pbest, and 3rd Sol as Gbest

# #### Now we have to calculate Y

# ##### Initial Velocity generator

# In[17]:


def initial_velocity(Range, Sol): #Range is a list
    a,b= Range
    m=len(Sol['G'])
    
    #generate a random uniform array  [-a,b] of the same size of the solutions 
    
    v=(b-a) * np.random.random_sample(m) +a  #http://bit.ly/3To2OWe
    v=v.tolist()
    
    return {'V':v}


# In[18]:


def Get_Y(X,GBest,PBest): #(Jarboui et al., 2008, p. 302)
    y=[]
    lens=[len(i) for i in [X['G'],GBest['G'],PBest['G']]]
    min_len=min(lens)
    
    for i in range(min_len):
        if X['G'][i]==GBest['G'][i] and X['G'][i]==PBest['G'][i]:
            y.append(random.choice([-1,1]))
        elif X['G'][i]==GBest['G'][i]:
            y.append(1)
        elif X['G'][i]==PBest['G'][i]:
            y.append(-1)
        else:
            y.append(0)
        
    return {'Y':y}


# ### Now we have to calculate Velocity

# In[19]:


def New_V(YY,VV,c1=1,c2=1,w=.75): #Parameter setting: (Jarboui et al., 2008, p. 306)
    Y=deepcopy(YY)
    V=deepcopy(VV)
    
    lens=[len(i) for i in [Y['Y'],V['V']]]
    min_len=min(lens)
    
    for i in range(min_len):
        y=Y['Y'][i]
        v=V['V'][i]
        V['V'][i]= w*v+ np.random.rand()*c1*(-1-y)+np.random.rand()*c2*(1-y)
        
    return V


# ### Now we need to calculate λ

# In[20]:


def Get_λ(YY,VV):
    Y=deepcopy(YY)
    V=deepcopy(VV)
    
    lens=[len(i) for i in [Y['Y'],V['V']]]
    min_len=min(lens)
    
    λ=[]
    for i in range(min_len):
        λ.append(Y['Y'][i]+V['V'][i])
    return {'λ':λ}
# λ=Get_λ(Y,V)
# λ


# ### Update X with Eq-10 (Jarboui et al., 2008, p. 303)

# In[21]:


def Perturbation(xg,xp,R,p_rate):
    
    if np.random.rand()<p_rate:
        p1,p2=sorted([xp,min(P_max,max(P_min,max(R)))])
        xp= rng.integers(p1,p2+1)
        if xp<P_min:
            xp=P_min
    for j in range(len(xg)): #small purtubration (like mutaion)
        if np.random.rand()<p_rate:
            xg[j]=0
            temp= min(math.ceil(R[j]/xp),math.floor((l_max-Length(xg))/(Y_b[j]/U)))
            temp= max(0,temp)
            #xg[j]=max(0,temp)
            xg[j]=rng.integers(0,temp+1)
            
    return xg,xp

def Update_X(XX,GBest,PBest,λλ, ϕ=0.5, p_rate=.05):
    X=deepcopy(XX)
    λ=deepcopy(λλ)
    
    lens=[len(i) for i in [X['G'],GBest['G'],PBest['G'],λ['λ']]]
    min_len=min(lens)
    XG=[]
    XP=[]
    R= Update_Res(R=Q_b,GG=XG,PP=XP)
    for i in range(min_len):
        if λ['λ'][i] > ϕ:
            #print('Gbest')
            #xg,xp=Perturbation(xg=GBest['G'][i],xp=GBest['P'][i],R=R,p_rate=p_rate)
            xg=GBest['G'][i]
            xp=GBest['P'][i]
                           
        elif λ['λ'][i] < -ϕ:
            #print('Pbest')
            #xg,xp=Perturbation(xg=GBest['G'][i],xp=GBest['P'][i],R=R,p_rate=p_rate)
            xg=PBest['G'][i]
            xp=PBest['P'][i]
        else:
            #print('X')
            xg,xp= Perturbation(xg=X['G'][i],xp=X['P'][i],R=R,p_rate=p_rate) #Perturbation function
            
        XG.append(xg)
        XP.append(xp)
        R= Update_Res(R=Q_b,GG=XG,PP=XP)
        if max(R)<=0:
            #print('break')
            return {'G':XG,'P':XP}

    for i in range(min_len, len(X['G'])):
        xg,xp= Perturbation(xg=X['G'][i],xp=X['P'][i],R=R,p_rate=p_rate) #Perturbation function
        XG.append(xg)
        XP.append(xp)
        R=Update_Res(R=Q_b,GG=XG,PP=XP)
        if max(R)<=0:
            #print('break')
            return {'G':XG,'P':XP}
 
    if max(R)>0:
        #print(R)
        #Use H1 or H3 algorithm to pack all sizes
        randint =rng.integers(2)
        if randint==0:
            #print('H1')
            h=H1(Q=R,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
        else:
            #print('H3')
            h=H3(Q=R,Y=Y_b,Pm=[P_min,P_max],U=U,e=e,l_max=l_max)
            
        g,p = h.values()    
        #print(g,p)            
        XG=XG+g
        XP=XP+p

    return {'G':XG,'P':XP}
# newX= Update_X(X,Gbest,Pbest,newY)
# newX


# In[22]:


y=[1,2,3,4]
c=[1,2]
print(y[:len(c)])


# In[23]:


def Update_dimension(XX,VV, in_vel_range=[-0.5,0.5]):
    mm= len(XX['G'])
    m= len(VV['V'])
    
    if mm <= m:
        return {'V':VV['V'][:m]}
    else:
        a,b= in_vel_range
        v=(b-a) * np.random.random_sample(mm-m) +a  #http://bit.ly/3To2OWe
        v=v.tolist()
        V=VV['V']+v
        return {'V':V}


# In[24]:


def Get_Gbest(p_o_s):
    
    gbest=p_o_s[0]
    for i in range(len(p_o_s)):
        if Fitness(p_o_s[i])<Fitness(gbest):
            gbest= p_o_s[i]
    return gbest
# Gbest=Get_Gbest(Pool_of_Sol)
# Gbest


# In[25]:


# newX= Update_X(X,Gbest,Pbest,newY)
# newX


# In[26]:


# Fitness(newX)


# In[27]:


#Pool_of_Sol


# # Main PSO

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
def PSO(swarmsize,iteration,ϕ=.7,c1=2,c2=2,w=1, in_vel_range=[-0.6,0.6],p_rate=.2):
    
    P_of_S= GeneratePopulation(swarmsize)
    P_of_Pbest=P_of_S
    P_of_Velocity= [initial_velocity(in_vel_range,P_of_S[i]) for i in range(len(P_of_S))]
    Gbest= P_of_S[rng.integers(0,swarmsize)]
    o= Gbest
    bests=[Fitness(Gbest)]
    for i in range(iteration):
        for j in range(len(P_of_S)):
            X=P_of_S[j]
            Pbest=P_of_Pbest[j]
            V= P_of_Velocity[j]
            Y= Get_Y(X=X,GBest=Gbest,PBest=Pbest)

            newV= New_V(YY=Y,VV=V,c1=c1,c2=c2,w=w)
            
            λ= Get_λ(YY=Y,VV=newV)

            newX= Update_X(XX=X,GBest=Gbest,PBest=Pbest,λλ=λ,ϕ=ϕ, p_rate=p_rate)

            P_of_S[j]=newX
            
            newV= Update_dimension(XX=newX,VV= newV, in_vel_range=in_vel_range)
            P_of_Velocity[j]= newV
            
            f=Fitness(newX)
            if f < Fitness(Pbest):
                P_of_Pbest[j]= newX
            if f < Fitness(Gbest):
                Gbest=newX
        #print(Gbest, Fitness(Gbest))
        bests.append(Fitness(Gbest))
    
    xx=[i for i in range(len(bests))]
    fig=px.line(x=xx,
                y=bests,
                title=f'swarmsize={swarmsize},iteration= {iteration},ϕ={ϕ},c1= {c1},c2={c2},w={w}, Gbest={bests[-1]}',
                labels=dict(x='iteration',y='fitness'))
    fig.show()
    #plt.plot(xx,bests)
    #plt.title(f'swarmsize={swarmsize},iteration= {iteration},ϕ={ϕ},c1= {c1},c2={c2},w={w}, Gbest={bests[-1]}')
    
    return CleanZeros(Gbest)
PSO(swarmsize=50,iteration=250)


# In[33]:


ObjectiveFunction(o)


# In[34]:


ObjectiveFunction(g)


# In[37]:


Dataset={
    'demands':[[872,1743,3486,2614,1743],
               [12,67,131,187,191,138,79,27],
               [990,1980,3961,2971,1980],
               [193,501,1018,1249,998,564,250,128]],
    'consumption':[[0.6119,0.6315,0.6499,0.6721,0.6921],
                   [0.7198,0.7352,0.7614,0.7878,0.8146,0.8423,0.8579,0.8985],
                   [1.3086,1.3671,1.4183,1.4538,1.5122],
                   [1.3350,1.3998,1.4356,1.4826,1.5440,1.5878,1.6313,1.6908]],
    'price':[1.51,2.43,1.95,2.9]
}
df = pd.DataFrame(columns=['ϕ','c1','c2','w','p_rate','solution','fitness'])
for i in range(len(Dataset['demands'])):
    demand=Dataset['demands'][i]
    Q_b= Shortage_allowance(demand,d)
    Y_b=Dataset['consumption'][i]
    f=Dataset['price'][i]
    PSO(swarmsize=100,iteration=120,c1=1,c2=2,ϕ =.4,w=.75,p_rate=.2)
 


# In[ ]:





# In[29]:


from itertools import product


# In[30]:


ϕ=[.4,.5,.6,.7]
c1=[1,1.5,2]
c2=[1,1.5,2]
ww=[.6,.75,1,1.25]
p_rate=[.05,.1,.2,.3]
iteration=product(ϕ,c1,c2,ww,p_rate)
#print(list(iteration))


# In[31]:


df = pd.DataFrame(columns=['ϕ','c1','c2','w','p_rate','solution','fitness'])
for ϕ,c1,c2,ww,p_rate in product(ϕ,c1,c2,ww,p_rate):
    best=PSO(swarmsize=100,iteration=120,c1=c1,c2=c2,ϕ=ϕ,w=ww,p_rate=p_rate)
    fitness=Fitness(best)
    df = df.append({'ϕ':ϕ,'c1':c1,'c2': c2,'w': ww,'p_rate': p_rate,'solution':best,'fitness':fitness}, ignore_index=True)
    df.to_csv('PSO_GridSearch_from_Notebook8.csv')


# In[32]:


print(df[['ϕ','c1','c2','w','p_rate','fitness']])


# import plotly
# import plotly.graph_objs as go
# 
# 
# #Read cars data from csv
# 
# 
# #Set marker properties
# markersize = df['c2']
# markercolor = df['w']
# markershape = df['c1'].replace(1,"square").replace(1.5,"circle").replace(2,'diamond')
# 
# 
# #Make Plotly figure
# fig1 = go.scater3d( x=df['α'],
#                     y=df['p_rate'],
#                     z=df['fitness'],
#                     marker=dict(#size=markersize,
#                                 #color=markercolor,
#                                 #symbol=markershape,
#                                 opacity=0.9,
#                                 reversescale=True,
#                                 colorscale='dense'),
#                     line=dict (width=0.02),
#                     mode='markers')
# 
# #Make Plot.ly Layout
# mylayout = go.Layout(scene=dict(xaxis=dict( title='α'),
#                                 yaxis=dict( title='p_rate'),
#                                 zaxis=dict(title='fitness')),)
# 
# #Plot and save html
# plotly.offline.plot({"data": [fig1],
#                      "layout": mylayout},
#                      auto_open=True,
#                      filename=("6DPlot.html"))
# 

# In[33]:


import plotly.express as px #https://plotly.com/python/3d-scatter-plots/
fig = px.scatter_3d(df, x='ϕ', y='p_rate', z='fitness',
                    color='c1', symbol='c2', size='w')
fig.show()


# In[34]:


df['c1+c2']=df['c1'].map(str)+','+df['c2'].map(str)
df


# In[35]:


fig = px.scatter_3d(df, x='ϕ', y='p_rate', z='fitness',
                    color='c1+c2', symbol='w')
fig.show()


# In[36]:


fig = px.parallel_coordinates(df, color="fitness",
                              dimensions=['c1','ϕ', 'c2','p_rate','w','fitness','c1+c2'],
                              #color_continuous_scale=px.colors.diverging.Tealrose,
                              #color_continuous_midpoint=0
                             )
fig.show()


# In[37]:


df.sort_values('fitness').head(10)


# In[38]:


type(df['c1'][1])


# In[39]:


Dataset={
    'demands':[[872,1743,3486,2614,1743],
               [12,67,131,187,191,138,79,27],
               [990,1980,3961,2971,1980],
               [193,501,1018,1249,998,564,250,128]],
    'consumption':[[0.6119,0.6315,0.6499,0.6721,0.6921],
                   [0.7198,0.7352,0.7614,0.7878,0.8146,0.8423,0.8579,0.8985],
                   [1.3086,1.3671,1.4183,1.4538,1.5122],
                   [1.3350,1.3998,1.4356,1.4826,1.5440,1.5878,1.6313,1.6908]],
    'price':[1.51,2.43,1.95,2.9]
}


# In[40]:


for i in range(len(Dataset['demands'])):
    demand=Dataset['demands'][i]
    Q_b= Shortage_allowance(demand,d)
    Y_b=Dataset['consumption'][i]
    f=Dataset['price'][i]
    best=PSO(swarmsize=100,iteration=120,c1=1,c2=2,ϕ=.4,w=.75,p_rate=.2)
    fitness=Fitness(best)
    df = df.append({'ϕ':ϕ,'c1':c1,'c2': c2,'w': ww,'p_rate': p_rate,'solution':best,'fitness':fitness}, ignore_index=True)
    df.to_csv('PSO_GridSearch_from_Notebook7.csv')


# In[ ]:





# In[ ]:




