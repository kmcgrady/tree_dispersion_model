#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pandas as pd
from random import choice, choices


# ### Log-normal distribution
std_f= 1
mean_f = 0
scale_f = 18.36 #scale parameter for Fox as dispersal vector
Pf = 0.005

std_b= 1
mean_b= 0
scale_b= 2 #scale parameter for Bird as dispersal vector
Pb = 0.015


# ## Evolution


# Tree life stages
# t1=seed 1
# t2= seed 2
# t3= seedling
# t4= sapling 1
# t5= sapling 2
# t6= sapling 3
# t7= sapling 4
# t8= sapling 5
# t9= sapling 6
# t10= sapling 7
# t11= adult







#compute all elements in all positions
def spreadandevolve2Dfast(x, Nt, Npx,Npy, P):
    '''
    Nt: time iterations
    Npx: number of positions in x
    Npy: number of positions in y
    P: transition matrix
    '''

    # Initialisation
    Ns = len(P)
    px = np.linspace(0,Npx,Npx)
    py = np.linspace(0,Npy,Npy)

    for n in range(1,Nt):
        #print(n)

        
        ## stage transition
        wplant = np.where(x[:, :,:] > 0) # only if plants are present
        for px in wplant[1]:
            for py in wplant[2]:
                x[:,px,py] = np.dot(P,x[:,px,py])

        # spread
        for s in range(2): # seeds stage                   
            pfx = px + choices([-1,1], k=Npx)*lognorm.rvs(std_f, size=Npx, scale=scale_f)*np.exp(mean_f) # move by foxes
            pfx = pfx.astype('int')
            pfx = np.where(pfx < 0, 0, pfx); pfx = np.where(pfx > Npx-1, Npx-1, pfx)
            pfy = py + choices([-1,1], k=Npy)*lognorm.rvs(std_f, size=Npy,scale=scale_f)*np.exp(mean_f)
            pfy = np.where(pfy < 0, 0, pfy); pfy = np.where(pfy > Npy-1, Npy-1, pfy)
            pfy = pfy.astype('int')

            pbx = px + choices([-1,1], k=Npx)*lognorm.rvs(std_b, size=Npx,scale=scale_b)*np.exp(mean_b) # move by birds
            pbx = pbx.astype('int')
            pbx = np.where(pbx < 0, 0, pbx); pbx = np.where(pbx > Npx-1, Npx-1, pbx)
            pby = py + choices([-1,1], k=Npy)*lognorm.rvs(std_b, size=Npy,scale=scale_b)*np.exp(mean_b)
            pby = pby.astype('int')
            pby = np.where(pby < 0, 0, pby); pby = np.where(pby > Npy-1, Npy-1, pby)



            x[s, pfx,pfy] = x[s, pfx,pfy] + Pf*x[s, px,py]
            x[s, pbx,pby] = x[s, pbx,pby] + Pb*x[s, px,py]
            x[s, px,py] = (1-Pf-Pb)*x[s, px,py]
        
    return x


# Position and iteration
Np = 100
Nt = 10


# Prunus serotina
Pprunus = pd.read_csv('lightA.csv', delimiter=';')
Pprunus = Pprunus.to_numpy()
Nsprunus = len(Pprunus)
xprunus = np.zeros((Nt,Nsprunus,Np,Np))
xprunus[0,0,int(Np/2),int(Np/2)] = 1000 # seeds1 at time 0
xprunus[0,1,int(Np/2),int(Np/2)] = 500 # seeds2 at time 0

# Pinus sylvestris
PPinus = pd.read_csv('corrourA.csv', delimiter=';')
PPinus = PPinus.to_numpy()
Nspinus = len(PPinus)
xpinus = np.zeros((Nt,Nspinus,Np,Np))
xpinus[0,0,int(Np/4),int(Np/2)] = 1000 # seeds1 at time 0
xpinus[0,1,int(Np/4),int(Np/2)] = 500 # seeds2 at time 0

# Fagus sylvatica
PFagus = pd.read_csv('mediumA.csv', delimiter=';')
PFagus = PFagus.to_numpy()
Nsfagus = len(PFagus)
xfagus = np.zeros((Nt,Nsfagus,Np,Np))
xfagus[0,0,int(Np*3/4),int(Np/2)] = 1000 # seeds1 at time 0
xfagus[0,1,int(Np*3/4),int(Np/2)] = 500 # seeds2 at time 0


# run
print('run')
n = 0
for n in range(1,Nt):
    print(n)

    xprunus[n,:, :,:] = spreadandevolve2Dfast(xprunus[n-1,:, :,:], 2, Np,Np, Pprunus)
    xpinus[n,:, :,:] = spreadandevolve2Dfast(xpinus[n-1,:, :,:], 2, Np,Np, PPinus)
    xfagus[n,:, :,:] = spreadandevolve2Dfast(xfagus[n-1,:, :,:], 2, Np,Np, PFagus)

    wprunus0 = np.where(xprunus[n-1,:, :,:] > 0); pprunus0 = np.zeros((Np,Np))
    wpinus0 = np.where(xpinus[n-1,:, :,:] > 0);   ppinus0 = np.zeros((Np,Np))
    wfagus0 = np.where(xfagus[n-1,:, :,:] > 0);   pfagus0 = np.zeros((Np,Np))

    wprunus = np.where(xprunus[n,:, :,:] > 0); pprunus = np.zeros((Np,Np))
    wpinus = np.where(xpinus[n,:, :,:] > 0); ppinus = np.zeros((Np,Np))
    wfagus = np.where(xfagus[n,:, :,:] > 0); pfagus = np.zeros((Np,Np))

    # already occupied
    for px in wprunus0[0]:
        for py in wprunus0[1]:
            pprunus0[px,py] = 1

    for px in wpinus0[0]:
        for py in wpinus0[1]:
            ppinus0[px,py] = 1

    for px in wfagus0[0]:
        for py in wfagus0[1]:
            pfagus0[px,py] = 1

    for px in wprunus[0]:
        for py in wprunus[1]:
            pprunus[px,py] = 1

    for px in wpinus[0]:
        for py in wpinus[1]:
            ppinus[px,py] = 1

    for px in wfagus[0]:
        for py in wfagus[1]:
            pfagus[px,py] = 1

    wpp = np.where( abs(pprunus - ppinus0) > 0)
    wpf = np.where( abs(pprunus - pfagus0) > 0)
    for px in wpp[0]:
        for py in wpp[1]:
            xprunus[n,:,px,py] = xprunus[n-1,:,px,py]
    for px in wpf[0]:
        for py in wpf[1]:
            xprunus[n,:,px,py] = xprunus[n-1,:,px,py]


    wpp = np.where( abs(ppinus - pprunus0) > 0)
    wpf = np.where( abs(ppinus - pfagus0) > 0)
    for px in wpp[0]:
        for py in wpp[1]:
            xpinus[n,:,px,py] = xpinus[n-1,:,px,py]
    for px in wpf[0]:
        for py in wpf[1]:
            xpinus[n,:,px,py] = xpinus[n-1,:,px,py]


    wfp = np.where( abs(pfagus - pprunus0) > 0)
    wpf = np.where( abs(pfagus - ppinus0) > 0)
    for px in wfp[0]:
        for py in wfp[1]:
            xfagus[n,:,px,py] = xfagus[n-1,:,px,py]
    for px in wpf[0]:
        for py in wpf[1]:
            xfagus[n,:,px,py] = xfagus[n-1,:,px,py]

    n = n + 1



import os

savefile = 'prunus'
if not os.path.exists(savefile):
    os.makedirs(savefile)

# animation
print('animation')
import matplotlib.animation as animation 
anim = [] 
fig = plt.figure(1)
plt.title('seeds')
#plt.xlim(0,100)
#plt.ylim(0,100)

for n in range(0,xprunus.shape[0],1):

    an = plt.imshow(np.log(1+xprunus[n,0,:,:].T+xprunus[n,1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/seeds00' + str(n) + '.png')

    anim.append(an)


anim = [] 
fig = plt.figure(1)
plt.title('adult')

for n in range(0,xprunus.shape[0],1):

    an = plt.imshow(np.log(1+xprunus[n,-1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/adults00' + str(n) + '.png')

    anim.append(an)




savefile = 'pinus'
if not os.path.exists(savefile):
    os.makedirs(savefile)

# animation
anim = [] 
fig = plt.figure(1)
plt.title('seeds')
#plt.xlim(0,100)
#plt.ylim(0,100)

for n in range(0,xpinus.shape[0],1):

    an = plt.imshow(np.log(1+xpinus[n,0,:,:].T+xpinus[n,1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/seeds00' + str(n) + '.png')

    anim.append(an)


anim = [] 
fig = plt.figure(1)
plt.title('adult')

for n in range(0,xprunus.shape[0],1):

    an = plt.imshow(np.log(1+xpinus[n,-1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/adults00' + str(n) + '.png')

    anim.append(an)



savefile = 'fagus'
if not os.path.exists(savefile):
    os.makedirs(savefile)

# animation
anim = [] 
fig = plt.figure(1)
plt.title('seeds')
#plt.xlim(0,100)
#plt.ylim(0,100)

for n in range(0,xfagus.shape[0],1):

    an = plt.imshow(np.log(1+xfagus[n,0,:,:].T+xfagus[n,1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/seeds00' + str(n) + '.png')

    anim.append(an)


anim = [] 
fig = plt.figure(1)
plt.title('adult')

for n in range(0,xfagus.shape[0],1):

    an = plt.imshow(np.log(1+xfagus[n,-1,:,:].T), cmap='jet')
    

    plt.savefig(savefile +'/adults00' + str(n) + '.png')

    anim.append(an)