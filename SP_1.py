from func import *
import PyPolyChord.PyPolyChord as PolyChord

#First data
data = np.loadtxt('Inputs/data/rvs_0001.txt')
obser = [data[:,0],data[:,1],data[:,2]]

nplan = 1
ndim = 2+nplan*5
nderived = 0
rootf = '1pd0001'

def loglkhd(x):
    return likel(x, obser, nplan)

def prior(x):
    return prior_hc(x,ndim)


PolyChord.run_nested_sampling(loglkhd, ndim, nderived, prior=prior,\
                              file_root= rootf, do_clustering='FALSE',\
                              nlive=100*ndim, update_files=100*ndim,\
                              num_repeats=10*ndim, boost_posterior=5)
