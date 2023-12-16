import numpy as np
import numpy.ma as ma
import logging
from scipy.interpolate import interp1d, pchip_interpolate
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.optimize as optimize
from copy import deepcopy

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

class GeometryTestFix(H5FlowStage):
    '''
        Used to test the in_fid() function in the Geometry resource of proto_nd_flow. Assumes
        distance units of [cm] for hits datasets and MiniRun4 geometry (i.e. 2x2 geometry from
        Summer/Autumn 2023) for tests. 

        Parameters:
         - ``geo_fix_dset_name``: ``str``, path to output dataset
         - ``hits_dset_name``: ``str``, path to input charge hits dataset
         - ``charge_dset_name``: ``str``, path to input charge dataset (1:1 with hits dataset, requires ``"Q"`` field)
         - fid_cut=20, # cm [ not currently used ]
         - cathode_fid_cut=20, # cm [ not currently used ]
         - anode_fid_cut=20, # cm [ not currently used ]

        Requires Geometry, RunData, and Units resource in workflow.
    '''
    class_version = '0.0.0'

    default_geo_dset_name = 'sel/geo_fix'
    default_hits_dset_name = 'charge/calib_final_hits'
    default_charge_dset_name = 'charge/calib_final_hits'

    #default_dbscan_eps = 2.5
    default_fid_cut=2.0, # cm
    default_cathode_fid_cut=2.0, # cm
    default_anode_fid_cut=2.0, # cm

    geo_dtype = np.dtype([('sel', 'u1')])


    def __init__(self, **params):
        super(GeometryTestFix, self).__init__(**params)

        self.geo_dset_name = params.get('geo_dset_name', self.default_geo_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.charge_dset_name = params.get('charge_dset_name', self.default_charge_dset_name)

        self._fid_cut = params.get('fid_cut', self.default_fid_cut)
        self._anode_fid_cut = params.get('anode_fid_cut', self.default_anode_fid_cut)
        self._cathode_fid_cut = params.get('cathode_fid_cut', self.default_cathode_fid_cut)
        
    def init(self, source_name):
        super(GeometryTestFix, self).init(source_name)

        self.data_manager.set_attrs(self.geo_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    hits_dset=self.hits_dset_name,
                                    charge_dset=self.charge_dset_name,
                                    fid_cut=self._fid_cut,
                                    anode_fid_cut=self._anode_fid_cut,
                                    cathode_fid_cut=self._cathode_fid_cut,
                                    )

        self.data_manager.create_dset(self.geo_dset_name, self.geo_dtype)

    def run(self, source_name, source_slice, cache):
        super(GeometryTestFix, self).run(source_name, source_slice, cache)

 # load arrays of event-level, hit-level, and track-level info
        
        events = cache[source_name]
        hits = ma.array(cache[self.hits_dset_name], shrink=False)
        q = ma.array(cache[self.charge_dset_name], shrink=False)
        q = q.reshape(hits.shape)
        

        if events.shape[0]:

            # calculate hit positions and charge
            hit_q = q['Q'] # convert mV -> ke
            # filter out bad channel ids            
            hit_mask = (hits['y'] != 0.0) & (hits['z'] != 0.0) & ~hit_q.mask & ~hits['t_drift'].mask            
            hit_q.mask = hit_q.mask | ~hit_mask
            hit_xyz = ma.array(np.concatenate([
                hits['x'][..., np.newaxis], hits['y'][..., np.newaxis],
                hits['z'][..., np.newaxis]], axis=-1), shrink=False, mask=np.zeros(hits['y'].shape + (3,), dtype=bool) | hit_q.mask[...,np.newaxis] | ~hit_mask[...,np.newaxis])
            #print("Hit XYZ reshaped Shape:", hit_xyz.reshape(-1, 3).shape)
            hit_in_fid = resources['Geometry'].in_fid(
                hit_xyz.reshape(-1, 3), cathode_fid=1.0, field_cage_fid=2.0, anode_fid=1.0).reshape(hit_xyz.shape[:-1])
            #print("Hit XYZ in FID Shape:", hit_xyz[hit_in_fid].shape)
            #print("Hit XYZ not in FID Shape:", hit_xyz[~hit_in_fid].shape)
            #print("Hit XYZ Mask Shape:", hit_xyz[hit_xyz.mask].shape)
            #print("Hit XYZ Not Mask Shape:", hit_xyz[~hit_xyz.mask].shape)
            #print("Hit XYZ in FID Shape NOT Mask:", hit_xyz[~hit_xyz.mask].shape)
            #print("Hit XYZ not in FID Shape NOT Mask:", hit_xyz[~hit_xyz.mask][~hit_in_fid].shape)
            #print("Hit XYZ Shape:", hit_xyz[~hit_xyz.mask].shape)
            #print("Hit XYZ Shape:", hit_xyz.shape)
            #print("Hit in FID shape:", hit_in_fid.shape)
            ##print("Hit in FID:", hit_in_fid[~hit_xyz.mask])
            #print("Hit XYZ FID F MASK FF Shape:", hit_xyz[~hit_in_fid & ~hit_q.mask & ~hit_mask].shape)
            #print("Hit XYZ FID T MASK FF Shape:", hit_xyz[hit_in_fid & ~hit_q.mask & ~hit_mask].shape)
            #print("Hit XYZ FID F MASK TT Shape:", hit_xyz[~hit_in_fid & hit_q.mask & hit_mask].shape)
            #print("Hit XYZ FID T MASK TT Shape:", hit_xyz[hit_in_fid & hit_q.mask & hit_mask].shape)
            #print("Hit XYZ FID F MASK FT Shape:", hit_xyz[~hit_in_fid & ~hit_q.mask & hit_mask].shape)
            #print("Hit XYZ FID T MASK FT Shape:", hit_xyz[hit_in_fid & ~hit_q.mask & hit_mask].shape)
            #print("Hit XYZ FID F MASK TF Shape:", hit_xyz[~hit_in_fid & hit_q.mask & ~hit_mask].shape)
            #print("Hit XYZ FID T MASK TF Shape:", hit_xyz[hit_in_fid & hit_q.mask & ~hit_mask].shape)
            #print("Hit XYZ FID F MASK FT:", hit_xyz[~hit_in_fid & ~hit_q.mask & hit_mask])
#

        tests = np.zeros(96, dtype=bool)
        tests_truth = np.zeros(96, dtype=bool)
        #print("Geometry LAr Bounds:", resources['Geometry'].lar_detector_bounds)
        #print("---- Tests with cathode_fid = 2.0 cm, anode_fid = 2.0 cm, field_cage fid = 1.0 cm ----")
        #print("---- Points in Modules (Should be TRUE) ----")
        tests[0] = resources['Geometry'].in_fid(xyz=np.array([300, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[1] = resources['Geometry'].in_fid(xyz=np.array([124, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[2] = resources['Geometry'].in_fid(xyz=np.array([-223, -3032, 12783])/10., cathode_fid = 20/10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[3] = resources['Geometry'].in_fid(xyz=np.array([-51, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[4] = resources['Geometry'].in_fid(xyz=np.array([500, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[5] = resources['Geometry'].in_fid(xyz=np.array([423, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[6] = resources['Geometry'].in_fid(xyz=np.array([-555, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[7] = resources['Geometry'].in_fid(xyz=np.array([-378, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[:8] = [True for test in tests_truth[:8]]
        #print("---- Points in Modules violating Cathode FID Cut (Should be FALSE) ----")
        tests[8] = resources['Geometry'].in_fid(xyz=np.array([328, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[9] = resources['Geometry'].in_fid(xyz=np.array([330, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[10] = resources['Geometry'].in_fid(xyz=np.array([-333, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[11] = resources['Geometry'].in_fid(xyz=np.array([-321, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[12] = resources['Geometry'].in_fid(xyz=np.array([347, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[13] = resources['Geometry'].in_fid(xyz=np.array([350, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[14] = resources['Geometry'].in_fid(xyz=np.array([-338, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[15] = resources['Geometry'].in_fid(xyz=np.array([-354, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[8:16] = [False for test in tests_truth[8:16]]
        #print("---- Points in Modules violating Anode FID Cut (Should be FALSE) ----")
        tests[16] = resources['Geometry'].in_fid(xyz=np.array([35, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[17] = resources['Geometry'].in_fid(xyz=np.array([49, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[18] = resources['Geometry'].in_fid(xyz=np.array([-38, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[19] = resources['Geometry'].in_fid(xyz=np.array([-44, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[20] = resources['Geometry'].in_fid(xyz=np.array([625, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[21] = resources['Geometry'].in_fid(xyz=np.array([633, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[22] = resources['Geometry'].in_fid(xyz=np.array([-621, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[23] = resources['Geometry'].in_fid(xyz=np.array([-638, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[16:24] = [False for test in tests_truth[16:24]]
        #print("---- Points in Modules violating Y_min Field Cage FID Cuts (Should be FALSE) ----")
        tests[24] = resources['Geometry'].in_fid(xyz=np.array([300, -3291, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[25] = resources['Geometry'].in_fid(xyz=np.array([124, -3295, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[26] = resources['Geometry'].in_fid(xyz=np.array([-223, -3297, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[27] = resources['Geometry'].in_fid(xyz=np.array([-51, -3293, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[28] = resources['Geometry'].in_fid(xyz=np.array([500, -3298, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[29] = resources['Geometry'].in_fid(xyz=np.array([423, -3292, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[30] = resources['Geometry'].in_fid(xyz=np.array([-555, -3294, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[31] = resources['Geometry'].in_fid(xyz=np.array([-378, -3296, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[24:32] = [False for test in tests_truth[24:32]]
        #print("---- Points in Modules violating Y_max Field Cage FID Cuts (Should be FALSE) ----")
        tests[32] = resources['Geometry'].in_fid(xyz=np.array([300, -2055, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[33] = resources['Geometry'].in_fid(xyz=np.array([124, -2056, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[34] = resources['Geometry'].in_fid(xyz=np.array([-223, -2052, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[35] = resources['Geometry'].in_fid(xyz=np.array([-51, -2060, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[36] = resources['Geometry'].in_fid(xyz=np.array([500, -2053, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[37] = resources['Geometry'].in_fid(xyz=np.array([423, -2054, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[38] = resources['Geometry'].in_fid(xyz=np.array([-555, -2057, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[39] = resources['Geometry'].in_fid(xyz=np.array([-378, -2058, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[32:40] = [False for test in tests_truth[32:40]]
        #print("---- Points in Modules violating Z_min Field Cage FID Cuts (Should be FALSE) ----")
        tests[40] = resources['Geometry'].in_fid(xyz=np.array([300, -2883, 12358])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[41] = resources['Geometry'].in_fid(xyz=np.array([124, -2350, 13028])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[42] = resources['Geometry'].in_fid(xyz=np.array([-223, -3032, 12362])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[43] = resources['Geometry'].in_fid(xyz=np.array([-51, -2124, 13034])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[44] = resources['Geometry'].in_fid(xyz=np.array([500, -2883, 12364])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[45] = resources['Geometry'].in_fid(xyz=np.array([423, -2350, 13031])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[46] = resources['Geometry'].in_fid(xyz=np.array([-555, -3032, 12360])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[47] = resources['Geometry'].in_fid(xyz=np.array([-378, -2124, 13033])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[40:48] = [False for test in tests_truth[40:48]]
        #print("---- Points in Modules violating Z_max Field Cage FID Cuts (Should be FALSE) ----")
        tests[48] = resources['Geometry'].in_fid(xyz=np.array([300, -2883, 12970])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[49] = resources['Geometry'].in_fid(xyz=np.array([124, -2350, 13636])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[50] = resources['Geometry'].in_fid(xyz=np.array([-223, -3032, 12971])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[51] = resources['Geometry'].in_fid(xyz=np.array([-51, -2124, 13637])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[52] = resources['Geometry'].in_fid(xyz=np.array([500, -2883, 12966])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[53] = resources['Geometry'].in_fid(xyz=np.array([423, -2350, 13640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[54] = resources['Geometry'].in_fid(xyz=np.array([-555, -3032, 12967])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[55] = resources['Geometry'].in_fid(xyz=np.array([-378, -2124, 13642])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[48:56] = [False for test in tests_truth[48:56]]
        #print("---- Points Outside Modules in Z (Should be False) ----")
        tests[56] = resources['Geometry'].in_fid(xyz=np.array([300, -2883, 12980])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[57] = resources['Geometry'].in_fid(xyz=np.array([124, -2350, 13001])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[58] = resources['Geometry'].in_fid(xyz=np.array([-223, -3032, 12999])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[59] = resources['Geometry'].in_fid(xyz=np.array([-51, -2124, 13015])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[60] = resources['Geometry'].in_fid(xyz=np.array([500, -2883, 12987])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[61] = resources['Geometry'].in_fid(xyz=np.array([423, -2350, 13021])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[62] = resources['Geometry'].in_fid(xyz=np.array([-555, -3032, 12994])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[63] = resources['Geometry'].in_fid(xyz=np.array([-378, -2124, 13007])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[56:64] = [False for test in tests_truth[56:64]]
        #print("---- Points Outside Modules in X (Should be False) ----")
        tests[64] = resources['Geometry'].in_fid(xyz=np.array([0, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[65] = resources['Geometry'].in_fid(xyz=np.array([15, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[66] = resources['Geometry'].in_fid(xyz=np.array([-3, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[67] = resources['Geometry'].in_fid(xyz=np.array([-28, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[68] = resources['Geometry'].in_fid(xyz=np.array([666, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[69] = resources['Geometry'].in_fid(xyz=np.array([640, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[70] = resources['Geometry'].in_fid(xyz=np.array([-679, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[71] = resources['Geometry'].in_fid(xyz=np.array([-800, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[64:72] = [False for test in tests_truth[64:72]]
        #print("---- Points Outside Modules in Y (Should be False) ----")
        tests[72] = resources['Geometry'].in_fid(xyz=np.array([300, -2000, 12640])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[73] = resources['Geometry'].in_fid(xyz=np.array([124, -2050, 13300])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[74] = resources['Geometry'].in_fid(xyz=np.array([-223, -1987, 12783])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[75] = resources['Geometry'].in_fid(xyz=np.array([-51, -1600, 13162])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[76] = resources['Geometry'].in_fid(xyz=np.array([500, -3300, 12640])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[77] = resources['Geometry'].in_fid(xyz=np.array([423, -3500, 13300])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[78] = resources['Geometry'].in_fid(xyz=np.array([-555, -3672, 12783])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[79] = resources['Geometry'].in_fid(xyz=np.array([-378, -3317, 13162])/10., cathode_fid = 2., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[72:80] = [False for test in tests_truth[72:80]]
        #print("---- Points inside Cathode (assuming non-zero thickness) (Should be FALSE) ----")
        tests[80] = resources['Geometry'].in_fid(xyz=np.array([334, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[81] = resources['Geometry'].in_fid(xyz=np.array([334.5, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[82] = resources['Geometry'].in_fid(xyz=np.array([-334.2, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[83] = resources['Geometry'].in_fid(xyz=np.array([-333.7, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[84] = resources['Geometry'].in_fid(xyz=np.array([335.5, -2883, 12640])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[85] = resources['Geometry'].in_fid(xyz=np.array([336, -2350, 13300])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[86] = resources['Geometry'].in_fid(xyz=np.array([-336.2, -3032, 12783])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests[87] = resources['Geometry'].in_fid(xyz=np.array([-335.1, -2124, 13162])/10., cathode_fid = 20./10., anode_fid = 20./10., field_cage_fid = 10./10.)
        tests_truth[80:88] = [False for test in tests_truth[80:88]]
        #print("---- Tests with cathode_fid = 5.0 cm, anode_fid = 5.0 cm, field_cage fid = 3.0 cm ----")
        #print("---- Points in Modules (Should be MIXED) ----")
        tests[88] = resources['Geometry'].in_fid(xyz=np.array([300, -2883, 12640])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[89] = resources['Geometry'].in_fid(xyz=np.array([124, -2350, 13300])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[90] = resources['Geometry'].in_fid(xyz=np.array([-223, -3032, 12783])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[91] = resources['Geometry'].in_fid(xyz=np.array([-91, -2124, 13045])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[92] = resources['Geometry'].in_fid(xyz=np.array([500, -2883, 12640])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[93] = resources['Geometry'].in_fid(xyz=np.array([423, -2350, 13300])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[94] = resources['Geometry'].in_fid(xyz=np.array([-555, -3280, 12783])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests[95] = resources['Geometry'].in_fid(xyz=np.array([-378, -2124, 13162])/10., cathode_fid = 50./10., anode_fid = 50./10., field_cage_fid = 30./10.)
        tests_truth[88] = False
        tests_truth[89] = True
        tests_truth[90] = True
        tests_truth[91] = False
        tests_truth[92] = True
        tests_truth[93] = True
        tests_truth[94] = False 
        tests_truth[95] = False

        test_var = 0
        for i in range(len(tests)):
            if tests[i] != tests_truth[i]:
                print("Test", i, "failed. Expected", tests_truth[i], "but got", tests[i])
                test_var += 1
        if test_var == 0:
            print("All tests passed!")
        else:
            print("Number of failed tests:", test_var)

    @staticmethod
    def hit_xyz(hits):
        xyz = np.concatenate((
            np.expand_dims(hits['x'], axis=-1),
            np.expand_dims(hits['y'], axis=-1),
            np.expand_dims(hits['z'], axis=-1),
        ), axis=-1)
        return xyz
