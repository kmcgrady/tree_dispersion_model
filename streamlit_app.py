#!/usr/bin/env python
# coding: utf-8

import plotly.express as px
import numpy as np
from scipy.stats import lognorm
import pandas as pd
from random import choice, choices
import streamlit as st

st.title("Dispersion and evolution Model of Tree Species")
st.write("prunus serotina, oinus sylvestris, fagus sylvatica")

# ### Log-normal distribution
with st.expander("Log-normal distribution"):
    st.write("Fox as Dispersal Vector")
    c1, c2, c3, c4 = st.columns(4)
    std_f= c1.number_input("STD_F", value=1.0)
    mean_f = c2.number_input("MEAN_F", -100.0, 100.0, 0.0)
    scale_f = c3.number_input("SCALE_F", 0.0, 100.0, 18.36) #scale parameter for Fox as dispersal vector
    Pf = c4.number_input("P_F", 0.0, 1.0, 0.005)

    st.write("Bird as Dispersal Vector")
    c1, c2, c3, c4 = st.columns(4)
    std_b= c1.number_input("STD_B", value=1.0)
    mean_b = c2.number_input("MEAN_B", -100.0, 100.0, 0.0)
    scale_b = c3.number_input("SCALE_B", 0.0, 100.0, 2.0) #scale parameter for Bird as dispersal vector
    Pb = c4.number_input("P_B", 0.0, 1.0, 0.015)

# Position and iteration
# species = c1.selectbox("Select a Tree Species", ["prunus serotina", "pinus sylvestris", "fagus sylvatica"])
c1, c2 = st.columns(2)
Np = c1.number_input("Position", 0, 1000, value=100)
Nt = c2.number_input("Iteration", 0, 1000, value=10)

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
def spreadandevolve2Dfast(x, Nt, Npx, Npy, P):
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

# Prunus serotina
Pprunus = pd.read_csv('lightA.csv', delimiter=';')
Pprunus = Pprunus.to_numpy()
Nsprunus = len(Pprunus)
xprunus = np.zeros((Nt,Nsprunus,Np,Np))
xprunus[0,0,int(Np/2),int(Np/2)] = 1000 # seeds1 at time 0
xprunus[0,1,int(Np/2),int(Np/2)] = 500 # seeds2 at time 0
HabitatPrunus = np.array([ [choice( ('good','poor','medium')) for col in range(Np)] for row in range(Np)])

# Pinus sylvestris
PPinus = pd.read_csv('corrourA.csv', delimiter=';')
PPinus = PPinus.to_numpy()
Nspinus = len(PPinus)
xpinus = np.zeros((Nt,Nspinus,Np,Np))
xpinus[0,0,int(Np/4),int(Np/2)] = 1000 # seeds1 at time 0
xpinus[0,1,int(Np/4),int(Np/2)] = 500 # seeds2 at time 0
HabitatPinus = np.array([ [choice( ('good','poor','medium')) for col in range(Np)] for row in range(Np)])

# Fagus sylvatica
PFagus = pd.read_csv('mediumA.csv', delimiter=';')
PFagus = PFagus.to_numpy()
Nsfagus = len(PFagus)
xfagus = np.zeros((Nt,Nsfagus,Np,Np))
xfagus[0,0,int(Np*3/4),int(Np/2)] = 1000 # seeds1 at time 0
xfagus[0,1,int(Np*3/4),int(Np/2)] = 500 # seeds2 at time 0
HabitatFagus = np.array([ [choice( ('good','poor','medium')) for col in range(Np)] for row in range(Np)])

@st.cache_data
def run_simulation():
    # run
    n = 0
    progress_bar = st.progress(0, text="Initializing Iterations")
    for n in range(1, Nt):
        progress_bar.progress(float(n) / float(Nt - 1), text=f"Processing Iteration {n}")

        xprunus[n,:, :,:] = spreadandevolve2Dfast(xprunus[n-1,:, :,:], 2, Np,Np, Pprunus)
        xpinus[n,:, :,:] = spreadandevolve2Dfast(xpinus[n-1,:, :,:], 2, Np,Np, PPinus)
        xfagus[n,:, :,:] = spreadandevolve2Dfast(xfagus[n-1,:, :,:], 2, Np,Np, PFagus)

        wprunus0 = np.where(xprunus[n-1,:, :,:] > 0); pprunus0 = np.zeros((Np,Np))
        wpinus0 = np.where(xpinus[n-1,:, :,:] > 0);   ppinus0 = np.zeros((Np,Np))
        wfagus0 = np.where(xfagus[n-1,:, :,:] > 0);   pfagus0 = np.zeros((Np,Np))

        wprunus = np.where(xprunus[n,:, :,:] > 0); pprunus = np.zeros((Np,Np))
        wpinus = np.where(xpinus[n,:, :,:] > 0); ppinus = np.zeros((Np,Np))
        wfagus = np.where(xfagus[n,:, :,:] > 0); pfagus = np.zeros((Np,Np))

        # positions
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

        # habitat suitability: already occupied and quality
        wpp = np.where( abs(pprunus - ppinus0) > 0)
        wpf = np.where( abs(pprunus - pfagus0) > 0)
        for px in wpp[0]:
            for py in wpp[1]:
                if HabitatPrunus[px,py] == 'good':
                    xprunus[n,:,px,py] = xprunus[n-1,:,px,py]
                elif HabitatPrunus[px,py] == 'poor':
                    xprunus[n,:,px,py] = 0
                else:
                    xprunus[n,:,px,py] = choice((xprunus[n-1,:,px,py],0))
        for px in wpf[0]:
            for py in wpf[1]:
                if HabitatPrunus[px,py] == 'good':
                    xprunus[n,:,px,py] = xprunus[n-1,:,px,py]
                elif HabitatPrunus[px,py] == 'poor':
                    xprunus[n,:,px,py] = 0
                else:
                    xprunus[n,:,px,py] = choice((xprunus[n-1,:,px,py],0))


        wpp = np.where( abs(ppinus - pprunus0) > 0)
        wpf = np.where( abs(ppinus - pfagus0) > 0)
        for px in wpp[0]:
            for py in wpp[1]:
                if HabitatPinus[px,py] == 'good':
                    xpinus[n,:,px,py] = xpinus[n-1,:,px,py]
                elif HabitatPinus[px,py] == 'poor':
                    xpinus[n,:,px,py] = 0
                else:
                    xpinus[n,:,px,py] = choice((xpinus[n-1,:,px,py],0))
        for px in wpf[0]:
            for py in wpf[1]:
                if HabitatPinus[px,py] == 'good':
                    xpinus[n,:,px,py] = xpinus[n-1,:,px,py]
                elif HabitatPinus[px,py] == 'poor':
                    xpinus[n,:,px,py] = 0
                else:
                    xpinus[n,:,px,py] = choice((xpinus[n-1,:,px,py],0))

        wfp = np.where( abs(pfagus - pprunus0) > 0)
        wpf = np.where( abs(pfagus - ppinus0) > 0)
        for px in wfp[0]:
            for py in wfp[1]:
                if HabitatFagus[px,py] == 'good':
                    xfagus[n,:,px,py] = xfagus[n-1,:,px,py]
                elif HabitatFagus[px,py] == 'poor':
                    xfagus[n,:,px,py] = 0
                else:
                    xfagus[n,:,px,py] = choice((xfagus[n-1,:,px,py],0))
        for px in wpf[0]:
            for py in wpf[1]:
                if HabitatFagus[px,py] == 'good':
                    xfagus[n,:,px,py] = xfagus[n-1,:,px,py]
                elif HabitatFagus[px,py] == 'poor':
                    xfagus[n,:,px,py] = 0
                else:
                    xfagus[n,:,px,py] = choice((xfagus[n-1,:,px,py],0))

        n = n + 1

    progress_bar.empty()

if 'ran' not in st.session_state:
    st.session_state.ran = False

if st.button("Rerun Simulation" if st.session_state.ran else "Run Simulation"):
    run_simulation.clear()
    st.session_state.ran = True

if not st.session_state.ran:
    st.stop()

run_simulation()

prunus_seeds = st.expander("Prunus seeds")
for n in range(0,xprunus.shape[0],1):
    fig = px.imshow(np.log(1+xprunus[n,0,:,:].T+xprunus[n,1,:,:].T), aspect='equal')
    st.plotly_chart(fig)

prunus_adult = st.expander("Prunus Adult")
for n in range(0,xprunus.shape[0],1):
    fig = px.imshow(np.log(1+xprunus[n,-1,:,:].T), aspect='equal')
    st.plotly_chart(fig)

pinus_seeds = st.expander("Pinus seeds")
for n in range(0,xpinus.shape[0],1):
    fig = px.imshow(np.log(1+xpinus[n,0,:,:].T+xpinus[n,1,:,:].T), aspect='equal')
    st.plotly_chart(fig)

pinus_adult = st.expander("Pinus Adult")
for n in range(0,xpinus.shape[0],1):
    fig = px.imshow(np.log(1+xpinus[n,-1,:,:].T), aspect='equal')
    st.plotly_chart(fig)

fagus_seeds = st.expander("Fagus seeds")
for n in range(0,xfagus.shape[0],1):
    fig = px.imshow(np.log(1+xfagus[n,0,:,:].T+xfagus[n,1,:,:].T), aspect='equal')
    st.plotly_chart(fig)

fagus_adult = st.expander("Fagus Adult")
for n in range(0,xfagus.shape[0],1):
    fig = px.imshow(np.log(1+xfagus[n,-1,:,:].T), aspect='equal')
    st.plotly_chart(fig)
