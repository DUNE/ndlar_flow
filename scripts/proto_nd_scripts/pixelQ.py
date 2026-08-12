import argparse
import json
import numpy as np
from math import ceil, floor, sqrt
import h5py

import h5flow
from h5flow.data import dereference, dereference_chain

from ROOT import TH1D, TH2D, TFile, TCanvas, TF1, TF1Convolution, gStyle


import matplotlib.pyplot as plt

def get_pixels_yz( f_manager, io_group ):
    #Assumes hits is charge/calib_prompt_hits/data/
    #If io group is on  module 2 use LArpixv2b specs else use v2a
    if io_group==5 or io_group==6:
        npix_y = 320
        npix_z = 160
    else:
        npix_y = 280
        npix_z = 140

    print("getting pixel positions for io group ", io_group)
    #check if we find the number of positions we expect, otherwise look at
    #a different file
    nfile = 1
    while nfile:
        try:
            hits = f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/data']
        except KeyError:
            try:
                hits = f_manager['/charge/calib_prompt_hits/data']
            except FileNotFoundError:
                print("file not found")

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
            print("looked at ", nfile, "files.")
            break

    positions = np.zeros((npix_y*npix_z, 2 ) )
    #return array with each pixel yz for the anode (io group)
    i = 0
    for j in range(npix_y):
        for k in range (npix_z):
            positions[i] = ( y_sort[j], z_sort[k])
            i+=1
    print("pixel positions found: ", positions.shape)       
    return positions

def build_hit_lookup(f_manager, io_group):
    """
    Build mapping: (y, z) -> network_id,
    """
    #check if file with the lookup already exists
    try:
        import json
        with open(f"/global/homes/l/lzazueta/hit_lookup_iogroup{io_group}.json", "r") as f:
            j = json.load(f)
        lookup = { tuple(map(float, k.split(','))): int(v) for k, v in j.items() }
        print("Loaded existing lookup from file.")
        return lookup
    except FileNotFoundError:
        print("Lookup file not found, building new lookup.")

    if io_group==5 or io_group==6:
        npix_y = 320
        npix_z = 160
    else:
        npix_y = 280
        npix_z = 140
    nfile = 1 
    lookup = {}
    while nfile:
        try:
            hits = f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/data']
        except KeyError:
            try:
                hits = f_manager['/charge/calib_prompt_hits/data']
            except FileNotFoundError:
                print("file not found")


        mask = hits["io_group"] == io_group
        ys = hits['y'][mask]
        zs = hits['z'][mask]
        ids = hits['id'][mask]

        packets = dereference(hits['id'],
                f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/ref/charge/packets/ref'],
                    f_manager['link'+str(nfile)+'/charge/packets/data'])
        unique_ids = unique_channel_id(packets)
        #unique_ids = network_agnostic_id(packets)

        for y, z, hid in zip(ys, zs, ids):
            if (y, z) not in lookup:   # keep first occurrence
                lookup[(y, z)] = unique_ids[hid]
        if len(lookup)< npix_y * npix_z:
            nfile += 1
            print("looking at file ", nfile, "found ", len(lookup), "ids")
        else:
            break
    #write the loopup dictionary to a file
    try:
        import json
        out_lookup = { f"{k[0]},{k[1]}": int(v) for k, v in lookup.items() }
        outpath = f"/global/homes/l/lzazueta/hit_lookup_iogroup{io_group}.json"
        with open(outpath, "w") as jf:
            json.dump(out_lookup, jf, indent=2)
        print("Wrote lookup to", outpath)
    except Exception as e:
        print("Could not write lookup:", e)

    print("found ", len(lookup), " unique pixel positions")
    return lookup

#Function that relates network id to the index of the position in the positions array
def get_networkid_to_position_index(lookup, positions):
    """
    Build mapping: positions_index -> network_id
    """
    networkid_to_index = {}
    for idx, pos in enumerate(positions):
        networkid_to_index[idx] = lookup.get(tuple(pos), None)
    return networkid_to_index


def get_pixel_ids_by_position(f_manager, positions):
    network_ids = {}
    nfile = 1
    while nfile:
        try:
            hits = f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/data']
            packets = dereference(hits['id'],
                        f_manager['link'+str(nfile)+'/charge/calib_prompt_hits/ref/charge/packets/ref'],
                            f_manager['link'+str(nfile)+'/charge/packets/data'])
            unique_ids = unique_channel_id(packets)
            #unique_ids = network_agnostic_id(packets)
        except KeyError:
            try:
                hits = f_manager['/charge/calib_prompt_hits/data']
            except FileNotFoundError:
                print("file not found")
        for pix_id, p in enumerate(positions):
            if pix_id % 1000 == 0:
                print(pix_id)
            y = p[0]
            z = p[1]

            mask = np.logical_and( hits['y']==y, hits['z']==z )
            id = np.unique(hits['id'][mask])
            if len(id) >= 1:
                _id = id[0].item()
                network_ids.update({pix_id: unique_ids[_id]})
            if len(id) == 0:
                continue
        if len(network_ids) < len(positions):
            nfile += 1
            print("looking at file ", nfile, "found ", len(network_ids), "ids")
        else:
            break
    return network_ids
            
def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def network_agnostic_id(d):
    return ((d['io_group'].astype(int)*1000+((d['io_channel'].astype(int)-1)//4+1))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def unique_to_io_group(unique):
    return ((unique // (100*1000*1000)) % 1000)

def unique_to_io_channel(unique):
    return (unique//(100*1000)) % 1000

def unique_to_chip_id(unique):
    return (unique // 100) % 1000

def unique_to_channel_id(unique):
    return (unique % 100)

def load_thresholds(filename):
    '''load  thresholds from a .json file'''
    with open(filename, 'r') as f:
        thresholds = json.load(f)
    print(len(thresholds), " thresholds loaded from ", filename)
    return thresholds

def get_thresholds_by_position_index(positions, lookup, thresholds):
    """
    Map position indices to threshold values.
    
    Parameters:
        positions (array): array of [y, z] positions
        lookup (dict): maps (y, z) tuple → network_id
        thresholds (dict): maps network_id → threshold value
    
    Returns:
        thresh_by_idx (dict): maps position index → threshold value
                              Returns None if position not in lookup or network_id not in thresholds
    """
    thresh_by_idx = {}
    threshold_keys = [int(k) for k in thresholds.keys()]
    missed_count = 0
    for pix_idx, pos in enumerate(positions):
        pos_tuple = (pos[0], pos[1])  # (y, z)
        
        if pos_tuple in lookup:
            network_id = lookup[pos_tuple]
            if network_id in threshold_keys:
                thresh_by_idx[pix_idx] = thresholds[str(network_id)]
            else:
                #print("network_id ", network_id, " not found in thresholds")
                missed_count += 1
                thresh_by_idx[pix_idx] = None
        else:
            print("position ", pos_tuple, " not found in lookup")
            thresh_by_idx[pix_idx] = None
    #print("mapped thresholds to position indices")
    print("missed thresholds ", missed_count)
    return thresh_by_idx

def intersect_pca_line_with_square(centroid, direction, center_yz, half_size):
    """
    Intersect a 3D line (centroid + t * direction) with an axis-aligned square in the yz plane.

    Parameters:
        centroid (array-like, 3) : point on the PCA line [x,y,z]
        direction (array-like, 3): unit (or not) direction vector of the line [dx,dy,dz]
        center_zy (array-like, 2) : square center given as [y, z]
        half_size (float)         : half side length of the square (pixel_pitch/2)

    Returns:
        intersections (list of np.ndarray): list of 3D points where the line intersects the square,
                                            sorted by t (ascending). Empty list if no intersection.
    """
    import numpy as _np

    c = _np.asarray(centroid, dtype=float)
    d = _np.asarray(direction, dtype=float)
    if d.shape[0] != 3 or c.shape[0] != 3:
        raise ValueError("centroid and direction must be length-3 arrays")

    # center_yz is given as [y, z]
    #print("center yz: ", center_yz)
    y0, z0 = center_yz[0], center_yz[1]
    #print("center yz components: ", y0, z0)
    y_min, y_max = y0 - half_size, y0 + half_size
    z_min, z_max = z0 - half_size, z0 + half_size

    c_y, c_z = c[1], c[2]
    d_y, d_z = d[1], d[2]
    #print("centroid yz: ", c_y, c_z, " direction yz: ", d_y, d_z)
    t_candidates = []

    eps = 1e-20
    precision = 1e-9 #cm
    # Intersections with y = y_min and y = y_max (vertical edges in yz plane)
    if abs(d_y) > eps:
        for y_edge in (y_min, y_max):
            t = (y_edge - c_y) / d_y
            z_at = c_z + t * d_z
            #print("t candidate at y edge ", y_edge, " is ", t, " z at t is ", z_at)
            if z_min - precision <= z_at <= z_max + precision:
                t_candidates.append(t)

    # Intersections with z = z_min and z = z_max (horizontal edges in yz plane)
    if abs(d_z) > eps:
        for z_edge in (z_min, z_max):
            t = (z_edge - c_z) / d_z
            y_at = c_y + t * d_y
            #print("t candidate at z edge ", z_edge, " is ", t, " y at t is ", y_at)
            if y_min - precision <= y_at <= y_max + precision:
                t_candidates.append(t)

    # unique and sorted
    if not t_candidates:
        #print("no intersections found with pixel at ", center_yz)
            # --------------------------------------------------
        # NO INTERSECTION: compute distance to square center
        # --------------------------------------------------

        # Project center onto line in yz-plane
        p = _np.array([y0, z0])
        c_yz = _np.array([c_y, c_z])
        d_yz = _np.array([d_y, d_z])

        denom = _np.dot(d_yz, d_yz)
        if denom < eps:
            # line is degenerate in yz → distance to point
            distance = _np.linalg.norm(p - c_yz)
        else:
            t_closest = _np.dot(p - c_yz, d_yz) / denom
            closest_point = c_yz + t_closest * d_yz
            distance = _np.linalg.norm(p - closest_point)
        #print("distance to pixel center ", center_yz, " is ", distance)
        # For debugging: plot the line and the pixel square
        '''
        t = _np.linspace(-60, 60, 100)
        mean = _np.array([centroid[2], centroid[1]])
        direction = _np.array([direction[2], direction[1]])
        line_points = mean + t[:, None] * direction
        plt.scatter(z0, y0, label='Data')
        plt.plot(line_points[:, 0], line_points[:, 1], color='red', label='PCA Line')
        plt.show()
        plt.savefig('testplot.png')
        '''

        return []

    t_unique = _np.unique(_np.array(t_candidates))
    t_sorted = _np.sort(t_unique)

    intersections = [c + t * d for t in t_sorted]
    return intersections

def distance_between_intersections(centroid, direction, center_zy, half_size):
    """
    Return (dist_3d, dist_yz) between the two intersection points of the PCA line
    with an axis-aligned square in the yz plane (square center given as [z, y]).
    """
    intersections = intersect_pca_line_with_square(centroid, direction, center_zy, half_size)
    if len(intersections) < 2:
        #print("found ", len(intersections), " intersections only for pixel at ", center_zy)
        return 0.0, 0.0

    p1 = np.asarray(intersections[0], dtype=float)
    p2 = np.asarray(intersections[1], dtype=float)

    dist_3d = np.linalg.norm(p2 - p1)
    # yz projected distance (y index 1, z index 2 in your 3D points)
    dist_yz = np.linalg.norm(p2[1:] - p1[1:])

    return dist_3d, dist_yz

def langau_fit(h, highstat=False):    
    f_conv = TF1Convolution("landau", "gaus", 0 , 120, True)
    f_conv.SetRange(0, 100)
    f_conv.SetNofPointsFFT(1000)
    f = TF1("f", f_conv, 15.0, 120.0, f_conv.GetNpar())
    f.SetParameters(1.0, 30.0 , 20, 2.0, 20, 1.0 )
    f.SetParLimits(3, 0, 100) #Keeping this parameter positive makes the fit more stable
    f.SetParLimits(1, 0, 100)
        
    c1 = TCanvas("c1", "c1", 800, 1000)

    # Fit and draw result of the fit
    #do chi-square for hight stat, negative log likelihood for low stats.
    if highstat:
        h.Fit("f", "R")
    else:
        h.Fit("f","LQR")

    mpv = f.GetMaximumX()

    chi2 = f.GetChisquare()
    #c1.SaveAs("fitConvolution.png")

    return mpv, chi2

def main():

    is_mc = args.is_mc
    io_group = args.io_group

    #f_name = '/global/homes/l/lzazueta/rockmuondatav2.hdf5'
    #f_name = '/global/homes/l/lzazueta/rockmuonmc_datav3.hdf5' #data without filter
    #f_name = '/global/homes/l/lzazueta/rockmuon_Datawfilter_july8.hdf5' #uses final hits

    #f_name = '/global/homes/l/lzazueta/rockmuonmc_mr62.hdf5'

    if is_mc:
        f_name = '/global/homes/l/lzazueta/rockmuon_MCMinirun65_pcainfo.hdf5'
    else:
        #f_name = '/global/homes/l/lzazueta/rockmuon_Datafilterv2_july8.hdf5'
        #f_name = '/global/homes/l/lzazueta/rockmuon_Datav10_160f.hdf5'
        f_name = '/global/homes/l/lzazueta/rockmuon_Datav11_64files.hdf5'

    #f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')
    f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')
    #f_manager_sim = h5flow.data.H5FlowDataManager(f_name_sim, 'r') 

    #f_h5 = h5py.File(f_name,'r')
    #print(f_h5.keys())

    #Get pixel yz positions for the anode
    #hits = f_manager['link1/charge/calib_prompt_hits/data']
    positions = get_pixels_yz( f_manager, io_group )

    #network_ids = get_pixel_ids_by_position(f_manager, positions)
    lookup = build_hit_lookup(f_manager, io_group)

    thresholds = load_thresholds('/global/homes/l/lzazueta/ndlar_flow/data/proto_nd_flow/thresholds_2x2.json')
    '''
    thr_count = 0
    channel_id_list = []
    for id, thr in thresholds.items():
        if ((float(id) // (100*1000*1000)) % 1000) == 3:
            #print("id ", id, "threshold ", thr)
            #print("io group ", unique_to_io_group(int(id)), " io channel ", unique_to_io_channel(int(id)), " chip id ", unique_to_chip_id(int(id)), " channel id ", unique_to_channel_id(int(id)), " threshold ", thr)
            if unique_to_channel_id(int(id)) not in channel_id_list:
                channel_id_list.append(unique_to_channel_id(int(id)))
            thr_count += 1
    print("thresholds loaded for io group: ", io_group, ": ",thr_count)
    print(sorted(channel_id_list))

    lookup_channel_ids = []
    for id in lookup.values():
        if unique_to_channel_id(int(id)) not in lookup_channel_ids:
                lookup_channel_ids.append(unique_to_channel_id(int(id)))
    print(sorted(lookup_channel_ids))
    '''
    threshold_idx = get_thresholds_by_position_index(positions, lookup, thresholds)

    if io_group==5 or io_group==6:
        pixel_pitch = 0.387975
    else:
        pixel_pitch = 0.4434

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
    if is_mc:
        outfilename = "/pscratch/sd/l/lzazueta/pixelQ_iogroup"+str(io_group)+"_mcmr65_"+ str(args.n_files) +"_v503.root" 
    else:
        outfilename = "/pscratch/sd/l/lzazueta/pixelQ_iogroup"+str(io_group)+"_datav11_"+ str(args.n_files) +"_v503.root" 

    gStyle.SetOptStat(1100)
    gStyle.SetOptFit(1)

    outfile = TFile(outfilename, "recreate")
    outfile.cd() 

    #make a histogram for each pixel, put them on a list
    npix = len(positions) 
    print("there are ", npix, " pixels")
    hist = []
    #dx_hist = []
    #dqdx_hist = []
    for n in range(npix):
        #h = TH1D( 'hpix'+str(n), 'hpix'+str(n), 20, 0, 60 ) 
        #dx = TH1D('dx'+str(n), 'dx per pixel'+str(n), 20, 0, 8 )
        #dqdx = TH1D('dqdx'+str(n), 'dQ/dx per pixel'+str(n), 20, 0, 60 )
        h = TH1D('dqdx'+str(n), 'dQ/dx per pixel'+str(n), 30, 0, 120 )
        hist.append(h)
        #dx_hist.append(dx)
        #dqdx_hist.append(dqdx)


    hqsum = TH1D('hqsum', 'Anode charge sum', 60, 0, 60 )
    #a histogram to count the frequency of hits per pixel per segment
    hfreq = TH1D('hit_freq', 'Hit frequency per pixel', 10, 0, 10 )
    #make 6 histograms for the number of sum hits per pixel
    hqsum1 = TH1D('hqsum1', 'hit per pixel per segment 1', 60, 0, 60 )
    hqsum2 = TH1D('hqsum2', 'hit per pixel per segment 2', 60, 0, 60 )
    hqsum3 = TH1D('hqsum3', 'hit per pixel per segment 3', 60, 0, 60 )

    hqmiss1 = TH1D('hqmiss1', 'missed hit per pixel per segment 1', 60, 0, 60 )
    hqmiss2 = TH1D('hqmiss2', 'missed hit per pixel per segment 2+', 60, 0, 60 )

    hdx = TH1D('dx', 'dx per pixel', 100, 0, 1)
    hdx_zoom = TH1D('dx_zoom', 'dx for pixel', 1000, 0.35, 1.0)
    hdqdx = TH1D('dqdx', 'dQ/dx per pixel', 1000, 0, 200 )

    hdqdx_dxbin1 = TH1D('dqdx_dxbin1', 'dQ/dx per pixel with dx less than 0.443', 100, 0, 200 )
    hdqdx_dxbin2 = TH1D('dqdx_dxbin2', 'dQ/dx per pixel with dx (0.4431, 0.45)', 100, 0, 200 )
    hdqdx_dxbin3 = TH1D('dqdx_dxbin3', 'dQ/dx per pixel with dx (0.451, 0.5)', 100, 0, 200 )
    hdqdx_dxbin4 = TH1D('dqdx_dxbin4', 'dQ/dx per pixel with dx 0.5+', 100, 0, 200 )

    #hhseg = TH1D('hpseg', 'hits per segment', 10, 0, 10 )
    #hqseg = TH1D('hqseg', 'charge/nhits per segment', 40, 0, 80)

    hsegdqdx = TH1D('segdqdx', 'dQ/dx per segment', 120, 0, 120 )

    hdq = TH1D('dq', 'dQ per segment', 100, 0, 400 )
    hgains = TH1D('gains', 'Gain correction', 1000, 0.5, 1.5 )
    hmpvs = TH1D("mpvs", "pixel MPVs ke/cm", 1000, 20, 70 )

    #hgains_bin1 = TH1D('gains_bin1', 'Gain hit 10-24 hits', 120, 0.5, 1.5 )
    #hgains_bin2 = TH1D('gains_bin2', 'Gain hit 25-39 hits', 120, 0.5, 1.5 )
    #hgains_bin3 = TH1D('gains_bin3', 'Gain hit 40+ hits', 120, 0.5, 1.5 )

    hthreshold = TH1D('thresholds', 'Thresholds', 100, 0, 20)
    for t in thresholds.values():
        hthreshold.Fill(t)
    #can = TCanvas("can", "can", 800, 1000)
    #hthreshold.Draw()
    #can.SaveAs("thresholds.png")

    #hq = TH2D('anodecharge', 'Anode charge', npix_z, zmin, zmax, npix_y, ymin, ymax )
    hh = TH2D('anodehits', 'Anode hits', npix_z, zmin, zmax, npix_y, ymin, ymax )
    hhighmpv = TH2D('highmpv', 'investigation of high mpv', npix_z, zmin, zmax, npix_y, ymin, ymax )
    hmvp_thresholds = TH2D('mpv_thresholds', 'MPV vs Thresholds', 100, 0, 20, 1000, 20, 70 )
    hdqdxdx = TH2D('dqdxdx', 'dqdx vs dx', 200, 0, 200, 100, 0, 1 )
    hdqdxdx2 = TH2D('dqdxdx2', 'dqdx vs dx without single hits', 200, 0, 200, 100, 0.35, 1.0 )

    hmean = TH2D('anodemeanq', 'Anode mean q', npix_z, zmin, zmax, npix_y, ymin, ymax )
    hrms = TH2D('anodermsq', 'Anode rms q', npix_z, zmin, zmax, npix_y, ymin, ymax )
    #hq_corrected = TH2D('anodecharge_corrected', 'Anode charge corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
    #hmean_corrected = TH2D('anodemeanq_corrected', 'Anode mean q corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
    #hrms_corrected = TH2D('anodermsq_corrected', 'Anode rms q corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
    #hq_test = TH2D('anodecharge_test', 'Anode charge test', npix_z, zmin, zmax, npix_y, ymin, ymax )
    #hq_test_corr = TH2D('anodecharge_test_corr', 'Anode charge test corrected', npix_z, zmin, zmax, npix_y, ymin, ymax )
    sum = 0
    #loop for each file, hardcoded how many
    for n in range(1,args.n_files+1):
        #print(n)

        try:
            f_manager['link' + str(n) + '/analysis/rock_muon_segments/ref/charge/calib_prompt_hits/ref']
            #f_manager['analysis/rock_muon_segments/ref/charge/calib_prompt_hits/ref']
        except KeyError:
            print("KeyError")
            continue
        
        #load datasets and references
        tracks = f_manager['link' + str(n) + '/analysis/rock_muon_tracks/data']
        #tracks_segment_ref = f_manager['link' + str(n) + '/analysis/rock_muon_tracks/ref/analysis/rock_muon_segments/ref']
        track_hits_ref = f_manager['link' + str(n) + '/analysis/rock_muon_tracks/ref/charge/calib_prompt_hits/ref']
        segments = f_manager['link' + str(n) + '/analysis/rock_muon_segments/data']
        #segments_hits_ref = f_manager['link' + str(n) + '/analysis/rock_muon_segments/ref/charge/calib_prompt_hits/ref']
        hits = f_manager['link' + str(n) + '/charge/calib_prompt_hits/data']

        iogroup_mask = segments['io_group']==io_group
        seg_dqdx = segments[iogroup_mask]['dQ'] / segments[iogroup_mask]['dx']
        for s in seg_dqdx:
            if s > 0:
                hsegdqdx.Fill(s)

        #packets = f_manager['link' + str(n) + '/charge/packets/data']
        #print(packets.dtype)
        
        #get the rock muon segment hits as calib_prompt_hits
        #track2segments = dereference( tracks['rock_muon_id'], tracks_segment_ref, segments )
        #segment_hits = dereference( segments['rock_segment_id'], segments_hits_ref, hits )
        #track_hits = dereference( tracks['rock_muon_id'], track_hits_ref, hits )
        
        #track_hits_group = track_hits[group_mask]
        
        for track in tracks:        
            track_hits = dereference( track['rock_muon_id'], track_hits_ref, hits ).flatten()
            group_mask = track_hits['io_group']==io_group

            if track_hits[group_mask].shape[0] < 2:
                continue

            center = [track['pca_mean_x'], track['pca_mean_y'], track['pca_mean_z']]
            direction = [track['pca_direction_x'], track['pca_direction_y'], track['pca_direction_z']]

            #positions_hits = np.array([track_hits['x'], track_hits['y'], track_hits['z']]).transpose()
            #Q = track_hits['Q']
            #Z = track_hits['z']
            #Y = track_hits['y']

            #dx = distance_between_intersections(center, direction, positions_hits, pixel_pitch/2  )
            #print(tr['rock_muon_id'])
        
            pix_list = []
            pix_dict = {}
            dx_dict = {}
            #position_dict = {}
            count_dict = {}
            dQ = 0.

            for hit in track_hits[group_mask]:
                if hit['is_disabled']:
                    continue    

                #x = hit['x']
                y = hit['y']
                z = hit['z']
                q = hit['Q']
                #print("hit y,z,q: ", y, z, q)

                if not q > 0:
                    continue
            
                #ignored nan positions for now
                if np.isnan(y) or np.isnan(z):
                    continue
            
                #find where this hit is on yz. i_pix is the row index for the positions
                where_y = np.where( np.logical_and( positions[:,0] == y, positions[:,1] == z ) )
                if len(where_y[0]) == 0:
                    print("Could not find pixel for hit at y ", y, " z ", z)
                    continue
                i_pix = where_y[0].item()
                #print("i_pix is ", i_pix, " for hit at y ", y, " z ", z, " with q ", q)

                #check if the pixel is already in the list
                if i_pix in pix_list:
                    pix_dict.update({str(i_pix): pix_dict[str(i_pix)] + q})
                    count_dict.update({str(i_pix): count_dict[str(i_pix)] + 1})
                else:
                    pix_list.append(i_pix)
                    pix_dict.update({str(i_pix): q})
                    count_dict.update({str(i_pix): 1})
                    #position_dict.update({str(i_pix): (y,z)} )
                dQ += q   
                hh.Fill(z,y)
                #hq.Fill(z,y,q)
            #pos = []
            #for pix in pix_dict.keys(): 
            #    pos.append( positions[int(pix)] )
            #print(pos)
            #plt.scatter( np.array(pos)[:,1], np.array(pos)[:,0], label='Data', color='green' )

            if dQ > 0:
                hdq.Fill(dQ)
            
            for pixel in pix_dict.keys():
                dx_3d, dx = distance_between_intersections(center, direction, positions[int(pixel)], pixel_pitch/2  )
                #store dx for each pixel
                hdx.Fill(dx_3d)
                hdx_zoom.Fill(dx_3d)
                dx_dict.update({pixel: dx_3d})

            for pix, sumq in pix_dict.items():
                #print("Pixel ", pix, " sumq ", sumq, " count ", count_dict[pix], " dx ", dx_dict[pix])
                #hist[int(pix)].Fill(sumq)
                hfreq.Fill(count_dict[pix])
                hqsum.Fill(sumq)
                _dx = dx_dict[pix]
                if _dx > 0:
                    dqdx = sumq / _dx
                    hdqdxdx.Fill(dqdx, _dx)
                if _dx > pixel_pitch*0.95:
                    #if dx_dict[pix] < 0.11:
                    #    print("Pixel ", pix, "count", count_dict[pix], " sumq ", sumq, " dx ", dx_dict[pix], " dqdx ", dqdx)
                    if count_dict[pix] == 1:
                        hqsum1.Fill(sumq)
                        if io_group==5 or io_group==6:
                            hist[int(pix)].Fill(dqdx)  
                            hdqdx.Fill(dqdx)

                    elif count_dict[pix] == 2:
                        hist[int(pix)].Fill(dqdx)
                        #dqdx_hist[int(pix)].Fill(dqdx)
                        hqsum2.Fill(sumq)
                        hdqdx.Fill(dqdx)
                        hdqdxdx2.Fill(dqdx, _dx)
                        '''
                        if _dx < 0.443:
                            hdqdx_dxbin1.Fill(dqdx)
                        elif 0.4431 <= _dx <= 0.45:
                            hdqdx_dxbin2.Fill(dqdx)
                        elif 0.451 < _dx <= 0.5:
                            hdqdx_dxbin3.Fill(dqdx)   
                        else:
                            hdqdx_dxbin4.Fill(dqdx)
                            '''
                    else:
                        hist[int(pix)].Fill(dqdx)
                        hqsum3.Fill(sumq)
                        hdqdx.Fill(dqdx)
                        hdqdxdx2.Fill(dqdx, _dx)
                else:
                    if count_dict[pix] == 1:
                        hqmiss1.Fill(sumq)
                    elif count_dict[pix] >= 2:
                        hqmiss2.Fill(sumq)
                    
        if n % 50 == 0:
            print(n)

    mpv_fit, chi2_fit = langau_fit(hsegdqdx, highstat=True)
    pixel_mpv, chi2_pixel = langau_fit(hdqdx, highstat=True)
    #mpv_fit = landau_fit(hsegdqdx, highstat=True)

    for pix in range(npix):
        hmean.Fill(positions[pix][1], positions[pix][0], hist[pix].GetMean())
        hrms.Fill(positions[pix][1], positions[pix][0], hist[pix].GetRMS())

    print('Done. File at: ' + outfilename)
    #outfile.Write()
    for h in hist:
        h.Write()

    hh.Write()
    #hq.Write()
    hqsum.Write()
    hfreq.Write()
    hqsum1.Write()
    hqsum2.Write()  
    hqsum3.Write()
    hqmiss1.Write()
    hqmiss2.Write()

    hdx.Write()
    hdx_zoom.Write()
    hdqdx.Write()
    hdqdx_dxbin1.Write()
    hdqdx_dxbin2.Write()
    hdqdx_dxbin3.Write()
    hdqdx_dxbin4.Write()
    hdq.Write()

    hsegdqdx.Write()
    hdqdxdx.Write()
    hdqdxdx2.Write()
    hhighmpv.Write()
    hthreshold.Write()
    hmvp_thresholds.Write()

    hgains.Write()
    #hgains_bin1.Write()
    #hgains_bin2.Write()
    #hgains_bin3.Write()
    hmpvs.Write()

    hmean.Write()
    hrms.Write()
    #hq_corrected.Write()
    #hq_test.Write()
    #hmean_corrected.Write()
    #hrms_corrected.Write()
    #hq_test_corr.Write()
    outfile.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_mc', action='store_true', help='Flag to indicate if the input file is MC')
    parser.add_argument('--io_group', type=int, help='IO group number (1-8)')
    parser.add_argument('--n_files', type=int, default=100, help='Number of files to process')
    args = parser.parse_args()

    main()
