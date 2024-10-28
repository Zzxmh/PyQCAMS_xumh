import numpy as np
import scipy.linalg as linalg
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar, fsolve
import pandas as pd
import os, time
import matplotlib.pyplot as plt
from pyqcams import util, constants, analysis
import warnings
# from joblib import Parallel, delayed
import multiprocess as mp
import torch
from potentials import morse, load_MLP_model, load_scaler, mlp_potential_function, mlp_derivative_function_numba
from util import hamiltonian, jac2cart, get_results
import logging
# Paths
model_path = 'models/so2_plus_mlp.pth'
scaler_path = 'models/scaler.joblib'

# Load scaler
scaler = load_scaler(scaler_path)

# Load model
model = load_MLP_model(model_path, input_dim=3, neuron=50)

# Create potential functions
V_MLP = mlp_potential_function(model, scaler)
dV_MLP = mlp_derivative_function_numba(V_MLP)
class Molecule:
    '''
    Represents a molecule with its physical properties and interaction potentials.
    '''
    def __init__(self, potential_type='analytical', potential_params=None, **kwargs):
        self.mi = kwargs.get('mi')  # Mass of atom i (atomic units)
        self.mj = kwargs.get('mj')  # Mass of atom j (atomic units)
        self.mu = self.mi * self.mj / (self.mi + self.mj)  # Reduced mass
        self.vi = kwargs.get('vi')  # Initial vibrational state
        self.ji = kwargs.get('ji')  # Initial rotational state
        self.Ei = kwargs.get('Ei')  # Initial energy (Hartree)
        self.xmin = kwargs.get('xmin', 0.5)
        self.xmax = kwargs.get('xmax', 30.0)
        self.npts = kwargs.get('npts', 1000)
        
        # Initialize potential
        self.potential_type = potential_type
        self.potential_params = potential_params
        self.Vij, self.dVij = self.initialize_potential()
        
        # Effective potential including rotational term
        self.Veff = lambda x: self.Vij(x) + self.ji * (self.ji + 1) / (2 * self.mu * x**2)
        
        # Attributes for equilibrium and turning points
        self.E = None
        self.rp = None
        self.rm = None
        self.re = None
        self.bdry = None
        self.bdx = None
        
        # Attributes for vibrational and rotational primes
        self.vPrime = None
        self.jPrime = None
    
    def initialize_potential(self):
        '''
        Initialize the potential based on the specified type.
        Supports analytical potentials and Machine Learning Potentials (MLP).
        '''
        if self.potential_type == 'analytical':
            # Example: Use Morse potential by default
            V, dV = morse(**self.potential_params) if self.potential_params else morse()
            return V, dV
        elif self.potential_type == 'MLP':
            # Load scaler and model
            scaler_path = self.potential_params.get('scaler_path')
            model_path = self.potential_params.get('model_path')
            scaler = load_scaler(scaler_path)
            model = load_MLP_model(model_path, input_dim=self.potential_params.get('input_dim', 3), neuron=self.potential_params.get('neuron', 50))
            V_MLP = mlp_potential_function(model, scaler)
            dV_MLP = mlp_derivative_function_numba(V_MLP)
            return V_MLP, dV_MLP
        else:
            raise ValueError(f"Unsupported potential type: {self.potential_type}")

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'mu:{self.mu!r}, vi:{self.vi!r}, ji:{self.ji!r})')
    
    def get_vi(self):
        return self.vi

    def set_E(self,value):
        self.E = value

    def set_rp(self,value):
        self.rp = value

    def set_rm(self,value):
        self.rm = value

    def set_re(self,value):
        self.re = value
    
    def set_vPrime(self,value):
        self.vPrime = value

    def set_jPrime(self,value):
        self.jPrime = value
        # New Veff for new jprime
        self.Veff = lambda x: self.Vij(x) + self.jPrime*(self.jPrime+1)/(2*self.mu*x**2)

    def checkBound(self, rf):
        '''
        Check if the molecule is bound. Bound molecules 
        have a defined equilibrium point and have energy
        less than the boundary. 
        '''
        # Bound molecule has defined equilibrium
        if self.re is not None:
            if self.bdry == 0: # No rotation
                if self.E < 0: 
                    return True
                else:
                    return False
            elif (self.E < self.bdry) and (rf < self.bdx): # Rotation
                return True
            else:
                return False
        else:
            return False
        
    def DVR(self):
        '''
        Returns the energy spectrum for a potential energy with a bound state.
        '''
        xl = float(self.xmax-self.xmin)
        dx = xl/self.npts
        n = np.arange(1,self.npts)

        x = float(self.xmin) + dx*n
        VX = np.diag(self.Vij(x)) # Potential energy matrix
        _i = n[:,np.newaxis]
        _j = n[np.newaxis,:]
        m = self.npts + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore divide by 0 warn
            # Off-diagonal elements of kinetic energy matrix
            T = ((-1.)**(_i-_j)
                * (1/np.square(np.sin(np.pi/(2*m)*(_i-_j)))
                - 1/np.square(np.sin(np.pi/(2*m)*(_i+_j)))))
        # Diagonal elements of KE matrix
        T[n-1,n-1] = 0 
        T += np.diag((2*m**2+1)/3
             -1/np.square(np.sin(np.pi*n/m)))
        T *= np.pi**2/4/xl**2/self.mu
        HX = T + VX # Hamiltonian 
        # evals, evecs = np.linalg.eigh(HX) # Solve the eigenvalue problem
        evals, evecs = linalg.eigh(HX)
        # evals = linalg.eigvalsh(HX)
        # if self.j == 0:
        #     evals = linalg.eigvalsh(HX, subset_by_index=[0,self.neigs])
        
        # To include the rotational coupling term 
        # E(v,j) = we*(v+.5) + wexe*(v+.5)^2 + bv*j*(j+1)
        # Where bv = be - ae*(v+.5) = (hbar^2/2m)<psi_v|1/r^2|psi_v>
        # We calculate the expectation value of the rotational energy of a 
        # vibrational eigenstate evecs
        # print((evecs[:,]**2).sum(axis=1)) # Check evecs is normalized
        # bv = []
        # for i in range(evecs.shape[1]):
        #     bv.append(np.trapz(evecs[:,i]**2/(x**2),dx=dx)/2/self.mu/dx)
        # bv = np.asarray(bv)
        # bv *= self.j*(self.j+1)
        self.evj = evals #+ bv
        return evals, evecs

    def rebound(self):
        '''
        Find re, bound point for j > 0
        '''
        if (hasattr(self,'jPrime')) and (self.jPrime is not None):
            dVeff = lambda x: self.dVij(x) - self.jPrime*(self.jPrime+1)/(self.mu*x**3)
        else:
            dVeff = lambda x: self.dVij(x) - self.ji*(self.ji+1)/(self.mu*x**3)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Ignore too many calls
                re = fsolve(dVeff, self.xmin)
                self.set_re(re) # set re
            # If no bound states exist, and large root is found
            if self.re > self.xmax:
                self.set_re(None)
            elif self.Veff(self.re) > 0:
                raise Exception('Equilibrium distance not found.')
        except Exception:
            self.set_re(None)
        # For j > 0
        if self.re is not None:
            try:
                self.bdx = root_scalar(dVeff, bracket = [self.re*1.001,self.xmax]).root
                self.bdry = self.Veff(self.bdx)
            except ValueError: # If j=0, boundary is 0
                self.bdry = 0

    def turningPts(self, initial = False):
        '''
        Set classical turning points and vprime for the product molecule.
        initial, Boolean
            True if initial molecule,
            False if product molecule

        '''
        self.rebound() # Find re, bound
        
        if initial:
            # If spectrum is not known, use DVR
            if self.Ei is None:
                self.DVR() # Set evj attribute
                self.Ei = self.evj[self.vi] # Set internal rovib energy

            diff = lambda x: self.Ei - self.Vij(x) - self.ji*(self.ji+1)/(2*self.mu*x**2)
            # Find outer turning point
            try:
                if self.bdry == 0:
                    self.rp = root_scalar(diff, bracket = [self.re, self.xmax]).root # set rplus
                else:
                    self.rp = root_scalar(diff, bracket = [self.re, self.bdx]).root # set rplus
            except:
                raise Exception(f'No outer turning point found, energy is too high.')

            # Find inner turning point
            self.rm = root_scalar(diff, bracket = [self.xmin, self.re]).root # set rminus
            
            # Set oscillation period, vprime
            self.tau = quad(lambda x: 1/np.sqrt(diff(x)),self.rm,self.rp)[0]
            self.tau *=np.sqrt(2*self.mu)
            vib = quad(lambda x: np.sqrt(diff(x)),self.rm,self.rp)[0]
            vib*=np.sqrt(2*self.mu)/np.pi
            vib+= -0.5 
            self.vi = np.round(vib)
            
        else:
            diff = lambda x: self.E - self.Vij(x) - self.jPrime*(self.jPrime+1)/(2*self.mu*x**2)
            # Find outer turning point
            # Sometimes E < Veff(re)
            if diff(self.re) < 0:
                raise Exception(f'E below minimum of potential.')
            try:
                if self.bdry == 0:
                    self.rp = root_scalar(diff, bracket = [self.re, self.xmax]).root # set rplus
                else:
                    self.rp = root_scalar(diff, bracket = [self.re, self.bdx]).root # set rplus
            except:
                raise Exception(f'No outer turning point found, energy is too high.')

            # Find inner turning point
            self.rm = root_scalar(diff, bracket = [self.xmin, self.re]).root # set rminus
            
            # Set vprime
            vib = quad(lambda x: np.sqrt(diff(x)),self.rm,self.rp)[0]
            vib*=np.sqrt(2*self.mu)/np.pi
            vib+= -0.5 
            self.set_vPrime(vib)
    
    def gaussBin(self,j_eff):
        '''
        Gaussian bin for the molecule's rovibrational number, 
        with sigma = 0.05
        Returns:
        w, float
            weight associated with vibrational product
        '''
        self.vt = np.round(self.vPrime)
        self.vw = np.exp(-np.abs(self.vPrime - self.vt)**2/0.05**2)
        self.vw *= 1/np.sqrt(np.pi)/0.05

        # j is rounded to the nearest integer
        self.jw = np.exp(-np.abs(j_eff - self.jPrime)**2/0.05**2)
        self.jw *= 1/np.sqrt(np.pi)/0.05


class Trajectory:
    '''
    Manages the simulation of molecular trajectories.
    '''
    def __init__(self, **kwargs):
        self.mol_12 = kwargs.get('mol_12')  # Initial molecule (SO2+)
        self.mol_23 = kwargs.get('mol_23')  # Potential product molecule (e.g., SO+)
        self.mol_31 = kwargs.get('mol_31')  # Another potential product molecule (e.g., O)
        self.vi = self.mol_12.vi
        self.ji = self.mol_12.ji
        self.m1 = kwargs.get('m1')  # Mass of Sulfur
        self.m2 = kwargs.get('m2')  # Mass of Oxygen
        self.m3 = kwargs.get('m3')  # Mass of Oxygen (dissociation)
        self.E0 = kwargs.get('E0')  # Collision energy in Hartree
        self.b = kwargs.get('b0')   # Impact parameter
        self.R0 = kwargs.get('R0')  # Initial separation (Bohr)
        self.v1 = self.mol_12.Veff  # Potential of mol_12 (SO2+)
        self.v2 = self.mol_23.Veff  # Potential of mol_23 (SO+)
        self.v3 = self.mol_31.Veff  # Potential of mol_31 (O)
        self.vtrip = kwargs.get('vt')  # Three-body potential if any
        self.dvtdr12 = kwargs.get('dvtdr12')  # Derivative of three-body potential
        self.dvtdr23 = kwargs.get('dvtdr23')
        self.dvtdr31 = kwargs.get('dvtdr31')
        self.seed = kwargs.get('seed')  # Seed for RNG
        self.t_stop = kwargs.get('integ')['t_stop']
        self.r_stop = kwargs.get('integ')['r_stop']
        self.a_tol = kwargs.get('integ')['a_tol']
        self.r_tol = kwargs.get('integ')['r_tol']
        self.econs = kwargs.get('integ')['econs']
        self.lcons = kwargs.get('integ')['lcons']
        self.mtot = self.m1 + self.m2 + self.m3
        self.mu12 = self.m1 * self.m2 / (self.m1 + self.m2)
        self.mu23 = self.m2 * self.m3 / (self.m2 + self.m3)
        self.mu31 = self.m1 * self.m3 / (self.m1 + self.m3)
        self.mu312 = self.m3 * (self.m1 + self.m2) / self.mtot
        self.C1 = self.m1 / (self.m1 + self.m2)
        self.C2 = self.m2 / (self.m1 + self.m2)
        
        # Initialize dissociation channels
        self.dissociation_channels = kwargs.get('dissociation_channels', ['SO+', 'O'])
        
        # Initialize logging
        logging.basicConfig(
            filename='qct_simulation.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
        # Initialize counters
        self.count = {channel: 0 for channel in self.dissociation_channels}
        self.count['dissociation'] = 0  # Overall dissociation count
        self.count['rejected'] = 0  # Rejected trajectories due to conservation violations
        
        # Initialize final state
        self.fstate = {
            'channels': {channel: {'v': None, 'vw': 0.0, 'j': None, 'jw': 0.0} for channel in self.dissociation_channels}
        }
        
        # Initialize random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Initialize initial conditions
        self.iCond()
    
    def iCond(self):
        '''
        Initialize initial conditions based on E0 and impact parameter.
        '''
        # Example: Initialize positions and momenta
        # This needs to be tailored based on your system
        pass  # Implement initial condition setup
    
    def runT(self):
        '''
        Run one trajectory with enhanced dissociation handling.
        '''
        logging.info(f"Starting trajectory: E0={self.E0} Hartree, b={self.b}")
        
        # Define stopping conditions
        def stop_r(t, y):
            # Example stopping condition based on separation
            r = np.linalg.norm(y[:3] - y[3:6])  # Distance between particles
            return r - self.r_stop
        stop_r.terminal = True
        stop_r.direction = 1
        
        # Define the ODE system
        def hamEq(t, y):
            '''
            Hamiltonian equations of motion.
            '''
            # y contains positions and momenta: [x1, y1, z1, x2, y2, z2, p1x, p1y, p1z, p2x, p2y, p2z]
            r1 = y[:3]
            r2 = y[3:6]
            p1 = y[6:9]
            p2 = y[9:12]
            
            # Compute distances
            r12 = np.linalg.norm(r1 - r2)
            
            # Compute forces
            F12 = -self.mu12 * self.v1(r12)  # Assuming v1 is potential energy, derivative gives force
            # Similarly, compute other forces if necessary
            
            # Example: Simple two-body interaction
            drdt = np.zeros_like(y)
            drdt[:3] = p1 / self.mu12
            drdt[3:6] = p2 / self.mu12
            drdt[6:9] = F12 * (r1 - r2) / r12
            drdt[9:12] = -F12 * (r1 - r2) / r12
            
            return drdt
        
        # Integrate equations of motion
        sol = solve_ivp(
            fun=hamEq,
            t_span=(0, self.t_stop),
            y0=self.w0,  # Initial state vector
            method='RK45',
            rtol=self.r_tol,
            atol=self.a_tol,
            events=stop_r
        )
        
        self.wn = sol.y
        self.t = sol.t
        
        # Compute Hamiltonian properties
        etot, epot, ekin, ll = hamiltonian(self)
        self.etot = etot
        self.epot = epot
        self.ekin = ekin
        self.ll = ll
        
        # Check conservation
        self.delta_e = self.etot[-1] - self.etot[0]
        self.delta_l = self.ll[-1] - self.ll[0]
        if abs(self.delta_e) > self.econs or abs(self.delta_l) > self.lcons:
            self.count['rejected'] += 1
            logging.warning(f"Trajectory rejected due to conservation violation: ΔE={self.delta_e}, ΔL={self.delta_l}")
            return
        
        # Convert Jacobi to Cartesian
        r12, r23, r31 = jac2cart(self.wn[:6], self.C1, self.C2)
        
        # Determine final states based on dissociation channels
        final_states = self.determine_final_states(r12[-1], r23[-1], r31[-1])
        
        # Assign final states to appropriate channels
        for channel, state in final_states.items():
            if state['bound']:
                self.count[channel] += 1
                self.fstate['channels'][channel] = state
            else:
                self.count['dissociation'] += 1
        
        logging.info(f"Trajectory completed: Counts={self.count}")
    
    def determine_final_states(self, r12, r23, r31):
        '''
        Determine the final states of the trajectory based on final internuclear distances.
        
        Parameters:
        - r12, r23, r31: Final internuclear distances.
        
        Returns:
        - final_states: Dictionary with channel names as keys and state information as values.
        '''
        final_states = {}
        for channel in self.dissociation_channels:
            if channel == 'SO+':
                # Define criteria for SO+ + O formation
                # Example criteria: r12 < equilibrium_SO and r23 > cutoff
                bound = (r12 < self.mol_12.xmin + 0.5) and (r23 > self.mol_23.xmin + 0.5)
                final_states[channel] = {
                    'bound': bound,
                    'v': self.get_final_vibrational_state(channel),
                    'vw': 0.0,  # Assign weights as needed
                    'j': self.get_final_rotational_state(channel),
                    'jw': 0.0
                }
            elif channel == 'O':
                # Define criteria for free Oxygen
                bound = r23 > self.mol_23.xmin + 0.5
                final_states[channel] = {
                    'bound': not bound,  # Free if not bound
                    'v': None,
                    'vw': 0.0,
                    'j': None,
                    'jw': 0.0
                }
            # Add more channels as needed
        return final_states
    
    def get_final_vibrational_state(self, channel):
        '''
        Placeholder function to determine the final vibrational state.
        Implement based on your criteria or external data.
        '''
        return None
    
    def get_final_rotational_state(self, channel):
        '''
        Placeholder function to determine the final rotational state.
        Implement based on your criteria or external data.
        '''
        return None
def runOneT(*args,output=False,**kwargs):
    '''
    Runs one trajectory. Use this method as input into loop.
    '''
    input_dict = kwargs.get('input_dict')
    try:
        traj = Trajectory(**input_dict)
        traj.runT()
        res = util.get_results(traj,*args)
        if output:
            out = {k:[v] for k,v in res.items()} # turn scalar to list
            out = pd.DataFrame(out)
            out.to_csv(output,mode = 'a', index = False,
                    header = os.path.isfile(output) == False or os.path.getsize(output) == 0)
        return res
    except Exception as e:
        print(e)
        pass
    return

def runN(nTraj, input_dict, cpus = os.cpu_count(), attrs = None,
        short_out = None, long_out = None):
    t0 = time.time()
    result = []
    with mp.Pool(processes=cpus) as p:
        if attrs:
            event = [p.apply_async(runOneT, args = (*attrs,),kwds=({'output':long_out,'input_dict':input_dict})) for i in range(nTraj)]
        else:
            event = [p.apply_async(runOneT,kwds=({'output':long_out,'input_dict':input_dict})) for i in range(nTraj)]
        for res in event:
            result.append(res.get())
    result = [i for i in result if i is not None]
    full = pd.DataFrame(result)
    cols = ['vi','ji','e','b','n12','n23','n31','nd','nc']
    counts = full.loc[:,cols].groupby(['vi','ji','e','b']).sum() # sum counts
    counts['time'] = time.time() - t0
    # Short output
    if short_out:
        counts.to_csv(short_out, mode = 'a',
                    header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    
    return full, counts

if __name__ == '__main__':
    from potentials import *
    from constants import *
    m1 = 1.008*constants.u2me
    m2 = 1.008*constants.u2me
    m3 = 40.078*constants.u2me

    E0 = 40000 # collision energy (K)
    b0 = 0
    R0 = 50 # Bohr

    # Potential parameters in atomic units
    v12, dv12 = morse(de = 0.16456603489, re = 1.40104284795, alpha = 1.059493476908482)
    v23, dv23 = morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)
    v31, dv31 = morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)

    v123, dv123dr12, dv123dr23, dv123dr31 = axilrod(C=0)

    # Define molecule dictionaries
    mol_12 = {'mi': m1, 'mj': m2, 'vi': 1, 'ji': 10, 'Vij':v12, 'dVij':dv12, 'xmin': .5, 'xmax': 30, 
        'npts':1000}
    mol_23 = {'mi': m2, 'mj': m3, 'Vij': v23, 'dVij': dv23, 'xmin': 1, 'xmax': 40}
    mol_31 = {'mi': m3, 'mj': m1, 'Vij': v31, 'dVij': dv31, 'xmin': 1, 'xmax': 40}

    # Initiate molecules
    mol12 = Molecule(mi = m1, mj = m2, vi = 1, ji = 0,Vij = v12, dVij = dv12, 
                        xmin = .5, xmax = 30, npts=1000)
    mol23 = Molecule(mi = m2, mj = m3, Vij = v23, dVij = dv23, xmin = 1, xmax = 40)
    mol31 = Molecule(mi = m3, mj = m1, Vij = v31, dVij = dv31, xmin = 1, xmax = 40)

    input_dict = {'m1':m1,'m2':m2,'m3':m3,
    'E0': E0, 'b0': b0, 'R0': R0, 'seed': None,
    'mol_12': mol12,'mol_23': mol23,'mol_31': mol31,
    'vt': v123, 'dvtdr12': dv123dr12, 'dvtdr23': dv123dr23, 'dvtdr31': dv123dr31,
    'integ':{'t_stop': 2, 'r_stop': 2, 'r_tol': 1e-12, 'a_tol':1e-10,'econs':1e-5,'lcons':1e-5}}


    ################################################
    import plotters
    bi = np.arange(100)
    input_dict['b0'] = 3.75
    input_dict['seed'] = 27
    traj = Trajectory(**input_dict)
    traj.runT()
    print(util.get_results(traj))
    plotters.traj_plt(traj)
    plt.show()
    # for b in bi:
    #     print(f'Running b={b}')
    #     input_dict['seed']=b
    #     runOneT(input_dict=input_dict)
    # traj = Trajectory(**input_dict)
    # traj.runT()
    # plotters.traj_plt(traj)
    # plt.show()
    # print(traj.mol_23.__dict__)
    # runN(30, input_dict, short_out='tryshort.txt', long_out='trylong.txt')
    # input_dict['seed'] = 63
    # traj = QCT(**input_dict)
    # traj.runT()
    # print(traj.delta_e)
    # plotters.traj_plt(traj)
    # plt.title(traj.count)
    # plt.show()
    # input_dict['E0'] = 40000
    # t0 = time.time()
    # nTraj = 20
    # runN(nTraj,input_dict, attrs=('delta_e',),long_out='long_test.txt', opacity='opac_test.txt', vib=False,rot=False)
    # print(f'Time: {time.time()-t0}')
    # n_jobs = 8
    # attrs = ('delta_e',)
    # r = Parallel(n_jobs=n_jobs)(delayed(runOneT)(*attrs,**input_dict) for i in range(nTraj))
    # df = pd.DataFrame(r)
    # print(df)
    # analysis.opacity(df,GB = False, vib = True, rot = False, output = 'opacity_test.txt', mode = 'a')
    ######## Test batch of trajectories ##########
    # from pyqcams2 import plotters
    # fig, axs = plt.subplots(2,5)
    # axs = axs.ravel()
    # for i in np.arange(0,10):    
    #     print(i)
    #     traj = Trajectory(**input_dict)
    #     try:
    #         traj.runT()
    #         plotters.traj_plt(traj, ax = axs[i])
    #         axs[i].set_title(f'{i}:{traj.count}')
    #     except Exception as e:
    #         print(e)
    #         pass
    # plt.show()