from func import *
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings as PCS
import sys

#Import the arguments: full perior, nplanets, data, derived
args = sys.argv
if len(args) != 5:
    raise ValueError("It is expected 5 arguments: period range (a or b), #planets, data, derived")

pr = args[1]
nplan = int(args[2])
datan = args[3]
der = int(args[4])

#First data
data = np.loadtxt('Inputs/data/rvs_'+datan+'.txt')
obser = [data[:,0],data[:,1],data[:,2]]

ndim = 2+nplan*5
nderived = nplan*3*der
rootf = str(nplan)+'pd'+datan+pr

def loglkhd(x):
    return likel(x, obser, nplan)

def loglkhddev(x):
    return likeld(x, obser, nplan)

if pr == 'b':
    bound = np.loadtxt('Inputs/data/prior_bounds_'+datan+'.txt',delimiter=',',usecols=(2,3))
else:
    bound = []

def prior(x):
    return prior_hc(x,ndim,bound)

minimi = loglkhd

if der>0:
    minimi = loglkhddev

settings = PCS(ndim, nderived, nlive=2000*ndim, num_repeats=10*ndim)
settings.file_root = rootf
settings.do_clustering = True

ouput = PyPolyChord.run_polychord(minimi, ndim, nderived, settings, prior)
