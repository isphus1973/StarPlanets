import numpy as np
from george.kernels import ExpSquaredKernel #For Gaussian Processes
from george.kernels import ExpSine2Kernel #For Gaussian Processes
import george
import sys
sys.path.append('/home/arthur/temp/PolyChord') # Put correct path here.
import PyPolyChord.PyPolyChord as PolyChord

#First data
data = np.loadtxt('Inputs/data/rvs_0001.txt')
obser = [data[:,0],data[:,1],data[:,2]]

#From https://en.wikipedia.org/wiki/True_anomaly and
#http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf
#Returns the mean anomaly in the 2 pi interval
def mean_anomaly(M0,t,t0,tau):
    return (2 * np.pi * (t-t0) / tau + M0 ) % (2*np.pi)

#Returns the [cos(f),sin(f)], where f is the true anomaly.
def real_anomaly(M,e):
    ec_anol = M+op.newton(lambda x: e * np.sin(M+x)-x,e*np.sin(M),lambda y: e * np.cos(M+y)-1,\
                          fprime2=lambda z: -e * np.sin(M+z))
    return [(np.cos(ec_anol)-e)/(1-e * np.cos(ec_anol)),(np.sqrt(1-e**2) * np.sin(ec_anol))/(1-e*np.cos(ec_anol))]

def RV(t, K, p, e, w, M0, t0=0):
    p = np.abs(p)
    K = np.abs(K)
    w = np.abs(w)%(2*np.pi)
    e=np.abs(e)
    if e>1: e=0.9
    M0 = np.abs(M0)%(2*np.pi)
    cosf, sinf = real_anomaly(mean_anomaly(M0,t,t0,p),e)
    return K * (np.cos(w)*cosf-np.sin(w)*sinf+e*np.cos(w))
#Radial velocity for one planet


#Model for velocity function.
#Input: Planets parameters: Last 2 ate C and J. Five extra parameters for each planet
#Special thanks to Ben Nelson
def model(params, times, nPlanets):
    planets = [params[i*5:i*5+5] for i in range(nPlanets)]
    try:
        p, K, e, w, M = np.transpose(planets)
    except ValueError:
        p, K, e, w, M = 99999999., 0., 0., 0., 0.
    offset = params[5*nPlanets]
    jitter = params[-1]
    mod = np.zeros(len(times))
    for j in range(len(times)):
        mod[j] += sum([RV(times[j], K[i], p[i], e[i], w[i], M[i])\
                       for i in range(nPlanets)]) + offset
    return mod

#Gaussian Processes kernel
var, length_e, length_s, period = 3.0, 50., 0.5, 20.
gamma = 1/(2.*length_s*length_s)
kernel = var * ExpSquaredKernel(length_e*length_e) * ExpSine2Kernel(gamma=gamma, log_period=np.log(period))

### Define the likelihood function ###
#Thanks to Ben.
def lnlike(params, obs, nPlanets):
    times, rvs, sigs = obs
    jit = params[-1]
    sigs_plus_jitter = np.sqrt(sigs*sigs + jit*jit*np.ones(len(times)))
    gp = george.GP(kernel)
    gp.compute(times, sigs_plus_jitter)
    #    return gp.lnlikelihood(rvs - model_1(params, times, nPlanets))
    return gp.lnlikelihood(rvs - model(params, times, nPlanets))


#Must define priors!
def log_prior( params, nPlanets):
    value=0
    planets = [params[i*5:i*5+5] for i in range(nPlanets)]
    try:
        p, K, e, w, M = np.transpose(planets)
    except ValueError:
        p, K, e, w, M = 99999999., 0., 0., 0., 0.
    offset = params[5*nPlanets]
    jitter = params[-1]
    if (offset<-1000) or (offset>1000) or (jitter<0) or (jitter>99):
        return -np.infty
    value+=-np.log(2000) #offset priors
    value+=-np.log(1+jitter)-np.log(np.log(100)) #Jitter prior
    if nPlanets>0:
        if ((K<=0) + (K>999) + (e<0) + (e>=1) + (w<0) + (w>2*np.pi) + (M<0) + (M>2*np.pi)).any():
            return -np.infty
        value+=-np.log(p)-np.log(np.log(10000/1.25)) #Period prior
        value+=-np.log(1+K)-np.log(np.log(1000)) #K prior
        value+=-np.log(2*np.pi**2) #w, M priors
        value+=np.log(e/0.04)-e**2/0.08-np.log(-np.expm1(-e**2/0.08)) #e prior
    return value.sum()

def log_posterior(params, obs, nPlanets):
    return log_prior(params, nPlanets) + lnlike(params, obs, nPlanets)

def ln_l(x):
    return log_posterior(x,obser,0), [0.0]*0

#For the case of 0 planets!
#Code based on Boris Leistedt article: https://ixkael.github.io/efficiently-sampling-mixture-models-exploration-and-comparison-of-methods/
PolyChord.mpi_notification()
PolyChord.run_nested_sampling(ln_l, 2, 0, file_root='zeroplan', do_clustering='TRUE' )
