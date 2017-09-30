import numpy as np
from george.kernels import ExpSquaredKernel #For Gaussian Processes
from george.kernels import ExpSine2Kernel #For Gaussian Processes
import george
from scipy import optimize as op

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

#Radial velocity for one planet
def RV(t, K, p, e, w, M0, t0=0):
    cosf, sinf = real_anomaly(mean_anomaly(M0,t,t0,p),e)
    return K * (np.cos(w)*cosf-np.sin(w)*sinf+e*np.cos(w))

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
    return gp.lnlikelihood(rvs - model(params, times, nPlanets))

def likel(x, data, nplanet):
    return lnlike(x, data, nplanet), [0.0]*0

def likeld(x, data, nplanet):
    if nplanet > 0:
        x_planets = [x[i*5:i*5+5] for i in range(nplanet)]
        try:
            xp, xK, xe, xw, xM = np.transpose(x_planets)
        except ValueError:
            xp, xK, xe, xw, xM = 0, 0, 0, 0, 0
        hder = xe*np.sin(xw)
        kder = xe*np.cos(xw)
        ider = xw + xM
        derived = np.concatenate((hder,kder,ider))
    else:
        derived = np.array([0.0]*0)
    return lnlike(x, data, nplanet), derived.tolist()

#Priors on the hypercube
def prior_hc(cube,n_dim,bound=[]):
    #n_plan = (len(cube)-2)/5 #You should test if it is the right integer. You may discard this.
    n_plan = int((n_dim-2)/5)
    theta = [0.0] * n_dim
    x_planets = [cube[i*5:i*5+5] for i in range(n_plan)]
    try:
        xp, xK, xe, xw, xM = np.transpose(x_planets)
    except ValueError:
        xp, xK, xe, xw, xM = 0, 0, 0, 0, 0
    xc = cube[5*n_plan]
    xj = cube[-1]
    if n_plan>0 and len(bound) == 0:
        xp = np.sort(xp) #p1<p2<p3
        bound = np.array([[1.25,10000]]*n_plan)
    c = -1000+2000*xc
    j = np.power(10,2*xj)-1
    p = bound[:,0] * np.power(bound[:,1]/bound[:,0],xp )
    K = np.power(10, 3*xK) -1
    e = np.sqrt(-0.08*np.log(1-(1-np.exp(-1/0.08))*xe))
    w = 2*np.pi*xw
    M = 2*np.pi*xM
    theta[5*n_plan]=c
    theta[-1]=j
    for i in range(n_plan):
        theta[i*5]=p[i]
        theta[i*5+1]=K[i]
        theta[i*5+2]=e[i]
        theta[i*5+3]=w[i]
        theta[i*5+4]=M[i]
    return theta
