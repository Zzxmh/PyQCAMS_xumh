# analysis.py (Complete Modified Version)

import numpy as np
import pandas as pd
from scipy import integrate
import os
from constants import Boh2m, ttos, K2Har
fact_sig = (Boh2m * 1e2)**2  # Bohr^2 to cm^2
fact_k3 = (Boh2m * 1e2) / ttos  # sigma * v [cm/s]

def opacity(input, GB=True, vib=True, rot=True, output=None, mode='a'):
    '''
    Calculate P(E,b) of a QCT calculation with multiple dissociation channels.
    '''
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = pd.read_csv(input)
    
    # Ensure the input DataFrame has the expected columns
    expected_cols = ['e','b','vi','ji','n12','n23','n31','nd','nc','v','vw','j','jw']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Input data must contain columns: {expected_cols}")
    
    df = df.loc[:, expected_cols]
    
    # Group by necessary indices
    stats = df.groupby(['e','b','vi','ji']).sum().loc[:,:'nc']
    nTraj = stats[['n12','n23','n31','nd']].sum(axis=1)
    
    # Handle Gaussian Binning
    weights = df.set_index(['vi','ji'])
    weights['w'] = weights['vw'] * weights['jw']
    
    # Identify reaction channels
    is_AB = (weights['n23'] == 1) | (weights['n31'] == 1)  # Example condition for reactions
    AB = weights[is_AB].groupby(['vi','ji','e','v','j','b']).sum()
    BB = weights[weights['n12'] == 1].groupby(['vi','ji','e','b','v','j']).sum()
    Diss = weights[weights['nd'] == 1].groupby(['vi','ji','e','b']).sum()
    
    # Net weights including dissociation
    net_w = weights.groupby(['vi','ji','e','b']).sum()
    net_w['w'] += net_w['nd']
    
    # Calculate probabilities
    pR = (AB['w'] / net_w['w']).fillna(0)
    pR_err = (np.sqrt(AB['w']) / net_w['w'] * np.sqrt((net_w['w'] - AB['w']) / net_w['w'])).fillna(0)
    pQ = (BB['w'] / net_w['w']).fillna(0)
    pQ_err = (np.sqrt(BB['w']) / net_w['w'] * np.sqrt((net_w['w'] - BB['w']) / net_w['w'])).fillna(0)
    pDiss = (Diss['nd'] / net_w['w']).fillna(0)
    pDiss_err = (np.sqrt(Diss['nd']) / net_w['w'] * np.sqrt((net_w['w'] - Diss['nd']) / net_w['w'])).fillna(0)
    
    # Histogram binning (similar to Gaussian binning)
    hR = AB[['n23','n31']].sum(axis=1) / nTraj
    hR_err = np.sqrt(AB[['n23','n31']].sum(axis=1)) / nTraj * np.sqrt((nTraj - AB[['n23','n31']].sum(axis=1)) / nTraj)
    hQ = BB[['n12']].sum(axis=1) / nTraj
    hQ_err = np.sqrt(BB[['n12']].sum(axis=1)) / nTraj * np.sqrt((nTraj - BB[['n12']].sum(axis=1)) / nTraj)
    hDiss = Diss[['nd']].sum(axis=1) / nTraj
    hDiss_err = np.sqrt(Diss[['nd']].sum(axis=1)) / nTraj * np.sqrt((nTraj - Diss[['nd']].sum(axis=1)) / nTraj)
    
    # Initialize opacity DataFrame
    if GB:
        opacity = pd.DataFrame({
            'pR': pR,
            'pR_err': pR_err,
            'pQ': pQ,
            'pQ_err': pQ_err,
            'pDiss': pDiss,
            'pDiss_err': pDiss_err
        }).reset_index()
    else:
        opacity = pd.DataFrame({
            'pR': hR,
            'pR_err': hR_err,
            'pQ': hQ,
            'pQ_err': hQ_err,
            'pDiss': hDiss,
            'pDiss_err': hDiss_err
        }).reset_index()
    
    # Sort the DataFrame for consistency
    sort_cols = []
    if vib and rot:
        sort_cols = ['v','j']
    elif vib:
        sort_cols = ['v']
    elif rot:
        sort_cols = ['j']
    
    if sort_cols:
        opacity = opacity.sort_values(by=sort_cols)
    
    # Fill NaN values
    opacity = opacity.fillna(0)
    
    # Save to CSV if required
    if output is not None:
        opacity.to_csv(output, mode=mode, header=not os.path.isfile(output) or os.path.getsize(output) == 0, index=False)
    
    return opacity

def crossSection(input, GB=True, vib=True, rot=True, output=None, mode='a'):
    '''
    Calculate cross section, sigma(E), of a QCT calculation.
    '''
    # If input is opacity DataFrame
    if isinstance(input, pd.DataFrame):
        opac = input.copy()
    else:
        opac = opacity(input, GB=GB, vib=vib, rot=rot).copy()
    
    opac = opac.sort_values(by=['b'])  # Sort by impact parameter
    
    def integ(integrand, x):
        return 2 * np.pi * integrate.trapz(integrand, x=x)
    
    # Initialize sigma DataFrame
    if 'v' not in opac.columns and 'j' not in opac.columns:
        sig_R = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pR * g.b, x=g.b)) * fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pR_err * g.b, x=g.b)) * fact_sig
        sig_Q = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pQ * g.b, x=g.b)) * fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pQ_err * g.b, x=g.b)) * fact_sig
        sigma = pd.DataFrame({
            'sig_R': sig_R,
            'sig_Rerr': sig_Rerr,
            'sig_Q': sig_Q,
            'sig_Qerr': sig_Qerr
        })
    elif 'v' in opac.columns and 'j' not in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pR * g.b, x=g.b)) * fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pR_err * g.b, x=g.b)) * fact_sig
        sig_Q = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pQ * g.b, x=g.b)) * fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pQ_err * g.b, x=g.b)) * fact_sig
        sigma = pd.DataFrame({
            'sig_R': sig_R,
            'sig_Rerr': sig_Rerr,
            'sig_Q': sig_Q,
            'sig_Qerr': sig_Qerr
        }).reset_index()
    elif 'v' not in opac.columns and 'j' in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pR * g.b, x=g.b)) * fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pR_err * g.b, x=g.b)) * fact_sig
        sig_Q = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pQ * g.b, x=g.b)) * fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pQ_err * g.b, x=g.b)) * fact_sig
        sigma = pd.DataFrame({
            'sig_R': sig_R,
            'sig_Rerr': sig_Rerr,
            'sig_Q': sig_Q,
            'sig_Qerr': sig_Qerr
        }).reset_index()
    elif 'v' in opac.columns and 'j' in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pR * g.b, x=g.b)) * fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pR_err * g.b, x=g.b)) * fact_sig
        sig_Q = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pQ * g.b, x=g.b)) * fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pQ_err * g.b, x=g.b)) * fact_sig
        sigma = pd.DataFrame({
            'sig_R': sig_R,
            'sig_Rerr': sig_Rerr,
            'sig_Q': sig_Q,
            'sig_Qerr': sig_Qerr
        }).reset_index()
    
    # Add dissociation cross sections
    sigma['sig_Diss'] = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pDiss * g.b, x=g.b)) * fact_sig
    sigma['sig_Diss_err'] = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pDiss_err * g.b, x=g.b)) * fact_sig
    
    # Save to CSV if required
    if output is not None:
        sigma.to_csv(output, mode=mode, header=not os.path.isfile(output) or os.path.getsize(output) == 0, index=False)
    
    return sigma

def rate(input, mu, GB=True, vib=True, rot=True, output=None, mode='a'):
    '''
    Calculate rate coefficients, k(E), from cross sections.
    '''
    sigma = crossSection(input, GB=GB, vib=vib, rot=rot).copy()
    sigma['P0'] = np.sqrt(sigma['e'] * 2 * mu * K2Har)
    sigma[['sig_R', 'sig_Rerr', 'sig_Q', 'sig_Qerr', 'sig_Diss', 'sig_Diss_err']] = \
        sigma[['sig_R', 'sig_Rerr', 'sig_Q', 'sig_Qerr', 'sig_Diss', 'sig_Diss_err']].multiply(sigma['P0'] / mu * fact_k3, axis='index')
    
    # Rename columns for clarity
    sigma = sigma.rename(columns={
        'sig_R': 'k_R',
        'sig_Rerr': 'k_Rerr',
        'sig_Q': 'k_Q',
        'sig_Qerr': 'k_Qerr',
        'sig_Diss': 'k_Diss',
        'sig_Diss_err': 'k_Diss_err'
    })
    
    # Drop intermediate column
    sigma = sigma.drop(columns='P0')
    
    # Save to CSV if required
    if output is not None:
        sigma.to_csv(output, mode=mode, header=not os.path.isfile(output) or os.path.getsize(output) == 0, index=False)
    
    return sigma
