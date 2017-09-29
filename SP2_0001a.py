from func import *
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings as PCS
#from PyPolyChrod.priors import UniformPrior

#First data
data = np.loadtxt('Inputs/data/rvs_0001.txt')
obser = [data[:,0],data[:,1],data[:,2]]

nplan = 2
ndim = 2+nplan*5
nderived = 0
rootf = '2pd0001a'

def loglkhd(x):
    return likel(x, obser, nplan)

def prior(x):
    return prior_hc(x,ndim)

settings = PCS(ndim, nderived)#, nlive=100*ndim, num_repeats=10*ndim)
settings.file_root = rootf
settings.do_clustering = True

ouput = PyPolyChord.run_polychord(loglkhd, ndim, nderived, settings, prior)


