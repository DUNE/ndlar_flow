import numpy as np
import sys

N_CHANNELS = 96

infile = sys.argv[1]
outfile = sys.argv[2]

di = np.load(infile)
di = di['spectrum'] # noise power spectrum
di = np.clip(di, di[di>0].min(), None) # clip 0 bins to estimate noise amplitude
di = np.sqrt(di)

# remap channels
do = np.zeros((N_CHANNELS, di.shape[-1]))

# (-x, -z)
do[ 0: 6] = di[0,30:24:-1]
do[ 6:12] = di[0,23:17:-1]
do[12:18] = di[0,14:8:-1]
do[18:24] = di[0,7:1:-1]

# (+x, -z)
do[24:30] = di[1,62:56:-1]
do[30:36] = di[1,55:49:-1]
do[36:42] = di[1,46:40:-1]
do[42:48] = di[1,39:33:-1]

# (+x, +z)
do[48:54] = di[0,62:56:-1]
do[54:60] = di[0,55:49:-1]
do[60:66] = di[0,46:40:-1]
do[66:72] = di[0,39:33:-1]

# (-x, +z)
do[72:78] = di[1,30:24:-1]
do[78:84] = di[1,23:17:-1]
do[84:90] = di[1,14:8:-1]
do[90:96] = di[1,7:1:-1]

np.save(outfile, do)
               
