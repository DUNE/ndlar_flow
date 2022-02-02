import numpy as np
import numpy.ma as ma
from math import ceil

from h5flow.core import H5FlowStage, resources, H5FLOW_MPI

from module0_flow.util import units


class LightCalibration(H5FlowStage):
    '''
        Generates calibration coefficients for SiPM + detector modules based on
        the charge information from the event.

        Applies the following data quality selection:
         - only 1 light triggers present in event
         - no charge signal within ``fid_cut`` of light detectors

        Calibration is based on the maximum waveform amplitude within the
        specified sample window. A cut can be placed to exclude waveforms with
        small expected visible energy (``vis_energy_cut``).

        Parameters:
         - ``calib_dset_name``: ``str``, path to output dataset within HDF5 file
         - ``larpix_gain``: ``float``, larpix gain in e/mV
         - ``fid_cut``: ``float``, fiducial cut away from light detectors
         - ``vis_energy_cut``: ``float``, do not collect calibration data on waveforms with less than this amount of expected energy
         - ``gain_prefactor``: ``dict`` of ``dict`` of ``<adc #>: <channel #>: <prefactor value>`` adjusts the gain correction on each channel by this amount (e.g. to account for multiple SiPM per module)
         - ``sample_window``: search between ``[<min sample>, <max sample>]`` for the maximum ADC
         - ``wvfm_dset_name``: ``str``, path to input waveform dataset
         - ``hit_drift_dset_name``: ``str``, path to charge hit drift data
         - ``hits_dset_name``: ``str``, path to input charge hits dataset

        All of ``wvfm_dset_name``, ``hits_dset_name``, and ``hit_drift_dset_name``
        are required in the cache.

        Requires Geometry, RunData, and LArData resources in workflow.

        ``calib`` datatype (1:1 with event)::

            id          u4,                 unique identifier
            vis_charge  f8(nadc,nchannel),  visible charge in e-
            vis_energy  f8(nadc,nchannel),  visible energy in keV
            sig         f8(nadc,nchannel),  maximum ADC value within sample window

    '''
    class_version = '0.0.0'

    defaults = dict(
        calib_dset_name='light/calib',
        fid_cut=20.0, # mm
        vis_energy_cut=1e3, # keV
        larpix_gain=221, # e/mV
        gain_prefactor=dict(),
        sample_window=[0,0], # FIXME: figure out a decent value for these
        )

    def calib_dtype(self,nadc,nchannel):
        return np.dtype([
            ('id', 'u4'),
            ('vis_charge', 'f8', (nadc,nchannel)),
            ('vis_energy', 'f8', (nadc,nchannel)),
            ('sig', 'f8', (nadc,nchannel)),
            ('mask', 'u1', (nadc,nchannel))
        ])


    def __init__(self, **params):
        super(LightCalibration, self).__init__(**params)

        for key,val in self.defaults.items():
            setattr(self, key, params.get(key,val))

        self.wvfm_dset_name = params['wvfm_dset_name']
        self.hits_dset_name = params['hits_dset_name']
        self.hit_drift_dset_name = params['hit_drift_dset_name']


    def init(self, source_name):
        super(LightCalibration, self).init(source_name)

        self.nadc, self.nchannel = self.data_manager.get_dset(self.wvfm_dset_name)['samples'].shape[1:-1]
        self.calib_dtype = self.calib_dtype(self.nadc, self.nchannel)

        self.data_manager.create_dset(self.calib_dset_name, dtype=self.calib_dtype)
        self.data_manager.create_ref(source_name, self.calib_dset_name)


    def finish(self, source_name):
        super(LightCalibration, self).finish(source_name)
        if self.rank == 0:
            print('\tADC\tCHANNEL\tGAIN MEAN (ADC/keV)\tGAIN MED (ADC/keV)\tGAIN STD (ADC/keV)')

        channels = [(adc,ch) for adc in range(self.nadc) for ch in range(self.nchannel)]
        if H5FLOW_MPI:
            subset = slice(
                ceil(len(channels)/self.size) * self.rank,
                ceil(len(channels)/self.size) * (self.rank+1),
                )
            if subset.start < len(channels):
                channels = channels[subset]
            else:
                channels = []
            self.comm.barrier()

        if len(channels):
            for adc,ch in channels:
                calib = self.data_manager.get_dset(self.calib_dset_name)
                vis_energy = calib['vis_energy'][:,adc,ch]
                sig = calib['sig'][:,adc,ch]
                mask = calib['mask'][:,adc,ch].astype(bool)

                gain = ma.array(sig, mask=~mask) / vis_energy * self.gain_prefactor.get(adc,{ch: 1}).get(ch,1)

                print(f'\t{adc}\t{ch}\t{gain.mean():0.04e}\t{ma.median(gain):0.04e}\t{gain.std():0.04e}')


    def run(self, source_name, source_slice, cache):
        super(LightCalibration, self).run(source_name, source_slice, cache)
        event = cache[source_name]
        wvfm = cache[self.wvfm_dset_name]
        hits = cache[self.hits_dset_name]
        hit_drift = cache[self.hit_drift_dset_name]
        hit_drift = hit_drift.reshape(hits.shape)
        hit_xyz = np.concatenate((
            hits['px'][...,np.newaxis],
            hits['py'][...,np.newaxis],
            hit_drift['z'][...,np.newaxis]), axis=-1)

        if len(np.r_[source_slice]) != 0:
            # remove events that don't pass the basic quality selection
            event_mask = (np.any(~wvfm.mask['samples'].reshape(wvfm.shape + (-1,)))
                & ((~wvfm.mask['samples']).any(axis=-1).any(axis=-1).any(axis=-1).sum(axis=-1) == 1))
            hit_in_fid = resources['Geometry'].in_fid(hit_xyz.reshape(-1,3), field_cage_fid=self.fid_cut)
            hit_in_fid = hit_in_fid.reshape(hits.shape)
            fid_cut = np.any(~hit_in_fid & ~hits.mask['id'], axis=-1)
            event_mask = event_mask & fid_cut

            # calculate event energy
            lifetime = resources['LArData'].electron_lifetime(event['unix_ts'])[0][...,np.newaxis]
            hit_q = hits['q'] * self.larpix_gain * np.exp(hit_drift['t_drift'] * resources['RunData'].crs_ticks/lifetime)
            hit_e = (hit_q * resources['LArData'].ionization_w # FIXME: assumes MIP for all tracks
                / resources['LArData'].ionization_recombination(2.12 * units.MeV / units.cm))

            # calculate detector solid angle and visible energy
            shape = (self.nadc, self.nchannel)
            adc = np.arange(self.nadc).reshape(self.nadc,1)
            ch = np.arange(self.nchannel).reshape(1,self.nchannel)
            adc = np.broadcast_to(adc, shape)
            ch = np.broadcast_to(ch, shape)
            tpc_id = resources['Geometry'].tpc_id[(adc.ravel(), ch.ravel())].reshape(shape)
            det_id = resources['Geometry'].det_id[(adc.ravel(), ch.ravel())].reshape(shape)

            shape = hits.shape + (self.nadc, self.nchannel)
            acc = resources['Geometry'].solid_angle(hit_xyz.reshape(-1,3), tpc_id.ravel(), det_id.ravel()) / (4 * np.pi)
            acc = acc.reshape(shape)
            acc = acc * (tpc_id[np.newaxis,np.newaxis,...]+1 == hits['iogroup'][...,np.newaxis,np.newaxis])

            vis_charge = (acc * hit_q[...,np.newaxis,np.newaxis]).sum(axis=1)
            vis_energy = (acc * hit_e[...,np.newaxis,np.newaxis]).sum(axis=1)

            # calculate detector signal
            # get first trigger in each event
            wvfm_samples = wvfm['samples'][:,0,...,self.sample_window[0]:self.sample_window[1]].reshape(len(event), self.nadc, self.nchannel, -1)
            sig = wvfm_samples.max(axis=-1)
            sig_mask = vis_energy >= self.vis_energy_cut
        else:
            vis_charge = np.zeros(event.shape + (self.nadc, self.nchannel), dtype=float)
            vis_energy = np.zeros_like(vis_charge)
            sig = np.zeros_like(vis_charge)
            sig_mask = np.zeros_like(vis_charge, dtype=bool)

        calib_data = np.empty(event.shape, dtype=self.calib_dtype)
        calib_data['vis_charge'] = vis_charge
        calib_data['vis_energy'] = vis_energy
        calib_data['sig'] = sig
        calib_data['mask'] = sig_mask

        calib_slice = self.data_manager.reserve_data(self.calib_dset_name, len(calib_data))
        if len(event):
            calib_data['id'] = np.r_[calib_slice]
        self.data_manager.write_data(self.calib_dset_name, calib_slice, calib_data)
        self.data_manager.write_ref(source_name, self.calib_dset_name,
            np.c_[source_slice,calib_slice] if len(event) else np.empty((0,2), dtype=int))
