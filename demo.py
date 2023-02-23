import numpy as np
from scipy.io import loadmat
from glerb import GLERB
from utils import report, binarize

# load dataset
dataset = 'ns'
data = loadmat('datasets/%s.mat' % dataset)
X, D = data['features'], data['label_distribution']
Y = binarize(D)

# train GLERB
glerb = GLERB().fit(X, Y)
Drec = glerb.label_distribution_

# report the results
report(Drec, D, ds=dataset)