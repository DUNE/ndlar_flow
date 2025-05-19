def get_pixels_yz( f_manager, io_group ):
    #Assumes hits is charge/calib_prompt_hits/data/
    #If io group is on  module 2 use LArpixv2b specs else use v2a
    if io_group==5 or io_group==6:
        npix_y = 320
        npix_z = 160
    else:
        npix_y = 280
        npix_z = 140

    #check if we find the number of positions we expect, otherwise look at
    #a different file
    nfile = 1
    while nfile:
        hits = f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/data']
        #hits = f_manager['charge/calib_prompt_hits/data']
        #Make array with correct shape and get unique positions for y and z
        y = hits[hits['io_group']==io_group]['y']
        y_sort = np.unique(y)
        z = hits[hits['io_group']==io_group]['z']
        z_sort = np.unique(z)

        #for i, z in enumerate(z_sort):
        #    if i == 0:
        #        print(i, z)
        #   else:
        #       print(i, z, z - z_sort[i-1])

        if len(y_sort) < npix_y:
            nfile+=1
        else:
            break

    positions = np.zeros((npix_y*npix_z, 2 ) )
    #return array with each pixel yz for the anode (io group)
    i = 0
    for j in range(npix_y):
        for k in range (npix_z):
            positions[i] = ( y_sort[j], z_sort[k] )
            i+=1
           
    return positions

import numpy as np
from math import ceil, floor
import h5py

import h5flow
from h5flow.data import dereference, dereference_chain

from ROOT import TH1D, TH2D, TFile

#f_name = '/global/homes/l/lzazueta/rockmuondatav2.hdf5'
#f_name = '/global/homes/l/lzazueta/rockmuonmc_datav3.hdf5' #data without filter
#f_name = '/global/homes/l/lzazueta/rockmuon_Datawfilter_july8.hdf5' #uses final hits

#f_name = '/global/homes/l/lzazueta/rockmuonmc_mr62.hdf5'
#f_name = '/global/homes/l/lzazueta/rockmuon_Datafilterv2_july8.hdf5'
f_name = '/global/homes/l/lzazueta/rockmuon_MCMiniru64.hdf5'


#f_name = '/global/homes/l/lzazueta/rockmuon_Datafilter_july8.hdf5'

#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/datav3/packet-0050017-2024_07_09_00_14_34_CDT.FLOW.proto_nd_flow.hdf5'
#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/dataReflow/packet-0050017-2024_07_09_00_14_34_CDT.FLOW.proto_nd_flow.hdf5'#this file works
#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/dataReflow/packet-0050017-2024_07_08_13_43_25_CDT.FLOW.proto_nd_flow.hdf5'
#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/dataReflow/packet-0050017-2024_07_10_01_17_14_CDT.FLOW.proto_nd_flow.hdf5'

#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/datafilter/packet-0050017-2024_07_08_13_43_25_CDT.FLOW.proto_nd_flow.hdf5'

#f_name = '/global/cfs/cdirs/dune/users/lzazueta/rockmuon/test4/packet-0050017-2024_07_08_13_43_25_CDT.FLOW.proto_nd_flow.hdf5'

#f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')
f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')
#f_manager_sim = h5flow.data.H5FlowDataManager(f_name_sim, 'r') 

#f_h5 = h5py.File(f_name,'r')
#print(f_h5.keys())

#Get pixel yz positions for the anode
#hits = f_manager['link1/charge/calib_prompt_hits/data']
io_group = 1
positions = get_pixels_yz( f_manager, io_group )

if io_group==5 or io_group==6:
    npix_y = 320
    npix_z = 160
else:
    npix_y = 280
    npix_z = 140

print(positions.shape)
ymax, zmax = np.amax(positions, 0) + 0.44
ymin, zmin = np.amin(positions, 0)

#max, zmax = ceil(ymax), ceil(zmax)
#ymin, zmin = floor(ymin), floor(zmin)

print(ymax, zmax)
print(ymin, zmin)

outfilename = "/pscratch/sd/l/lzazueta/pixelQ_iogroup"+str(io_group)+"_mcmr64_940files.root" 
outfile = TFile(outfilename, "recreate")
outfile.cd() 

#make a histogram for each pixel, put them on a list
npix = len(positions) 
print("there are ", npix, " pixels")
hist = []
for n in range(npix):
    h = TH1D( 'hpix'+str(n), 'hpix'+str(n), 20, 0, 60 ) 
    hist.append(h)
hq = TH2D('anodecharge', 'Anode charge', npix_z, zmin, zmax, npix_y, ymin, ymax )
hh = TH2D('anodehits', 'Anode hits', npix_z, zmin, zmax, npix_y, ymin, ymax )
hmean = TH2D('anodemeanq', 'Anode mean q', npix_z, zmin, zmax, npix_y, ymin, ymax )
hrms = TH2D('anodermsq', 'Anode rms q', npix_z, zmin, zmax, npix_y, ymin, ymax )
hq_corrected = TH2D('anodecharge_corrected', 'Anode charge corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
hmean_corrected = TH2D('anodemeanq_corrected', 'Anode mean q corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
hrms_corrected = TH2D('anodermsq_corrected', 'Anode rms q corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
hq_test = TH2D('anodecharge_test', 'Anode charge test', npix_z, zmin, zmax, npix_y, ymin, ymax )
hq_test_corr = TH2D('anodecharge_test_corr', 'Anode charge test corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
sum = 0
#loop for each file, hardcoded how many
for n in range(1,941):
    #print(n)

    try:
        f_manager['link' + str(n) + '/analysis/rock_muon_segments/ref/charge/calib_prompt_hits/ref']
        #f_manager['analysis/rock_muon_segments/ref/charge/calib_prompt_hits/ref']
    except KeyError:
        print("KeyError")
        continue
    
    #hits = f_manager['link' + str(n) + '/charge/calib_prompt_hits/data']
    
    #get the rock muon segment hits as calib_prompt_hits
    
    hits = dereference(
    f_manager['link' + str(n) + '/analysis/rock_muon_tracks/data']['rock_muon_id'],     # indices of A to load references for, shape: (n,)
    f_manager['link' + str(n) + '/analysis/rock_muon_tracks/ref/charge/calib_prompt_hits/ref'],  # references to use, shape: (L,)
    f_manager['link' + str(n) + '/charge/calib_prompt_hits/data']
    )
    
    #angles = f_manager['link' + str(n) + '/analysis/rock_muon_tracks/data']['theta_yz']
    hits_group = hits[hits['io_group']==io_group]
    #for index, track in enumerate(hits):
        #angle = angles[index] - 90.
       
    
        #print("The shape of hits is " + str(hits_group.shape[0]))
        #i=0
    for hit in hits_group:
        y = hit['y']
        z = hit['z']
        q = hit['Q']
        
        if not q > 0:
            continue
        
        #ignored nan positions for now
        if np.isnan(y) or np.isnan(z):
            continue
        
        #if q > 60:
        #    print("Q is ", q, "file is", n)

        #find where this hit is on yz. i_pix is the row index for the positions
        where_y = np.where( np.logical_and(positions[:,0] == y, positions[:,1] == z ) )
        #i_pix = np.where(positions[ where_y ] == z)[0].item()
        i_pix = where_y[0].item()
        #print(i_pix)
        hist[i_pix].Fill(q)
        hq.Fill(z,y,q)
        hh.Fill(z,y)

        #i+=1
    if n % 50 == 0:
        print(n)

gains = np.zeros((npix_y*npix_z, 1 ) )
for pix in range(npix):
    gains[pix] = hist[pix].GetMean()
    hmean.Fill(positions[pix][1], positions[pix][0], hist[pix].GetMean())
    hrms.Fill(positions[pix][1], positions[pix][0], hist[pix].GetRMS())
    hq_test.Fill(positions[pix][1], positions[pix][0], hist[pix].GetSumOfWeights() )

gains.round(3)
print(np.mean(gains))
gains = gains / np.mean(gains)

for pix in range(npix):
    #if gains[pix] > 1.5:
    #    print("Pixel ", positions[pix][1], positions[pix][0], " gain is " + str(gains[pix]))
    if gains[pix] == 0.:
        hq_corrected.Fill(positions[pix][1], positions[pix][0], 0.0 )
        hq_test_corr.Fill(positions[pix][1], positions[pix][0], 0.0 )
        hmean_corrected.Fill(positions[pix][1], positions[pix][0], hist[pix].GetMean() )
        hrms_corrected.Fill(positions[pix][1], positions[pix][0], hist[pix].GetRMS() )
    else:
        hq_corrected.Fill(positions[pix][1], positions[pix][0],
                           1.0/gains[pix].item() )
        hq_test_corr.Fill(positions[pix][1], positions[pix][0],
                            hist[pix].GetSumOfWeights()/gains[pix].item() )
        hmean_corrected.Fill(positions[pix][1], positions[pix][0], hist[pix].GetMean()/gains[pix].item() )
        hrms_corrected.Fill(positions[pix][1], positions[pix][0], hist[pix].GetRMS()/gains[pix].item() )

print('Done. File at: ' + outfilename)
#outfile.Write()
for h in hist:
    h.Write()
hh.Write()
hq.Write()
hmean.Write()
hrms.Write()
hq_corrected.Write()
hq_test.Write()
hmean_corrected.Write()
hrms_corrected.Write()
hq_test_corr.Write()
outfile.Close()
