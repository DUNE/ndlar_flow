import numpy as np
import numpy.ma as ma
import numpy.lib.recfunctions as rfn
import h5py
import logging

from h5flow.core import H5FlowStage


class ExternalTriggerFinder(H5FlowStage):
    '''
        Extracts external triggers from raw packets

        Parameters:
         - ``ext_trigs_dset_name`` : ``str``, required, path to output dataset
         - ``packets_dset_name`` : ``str``, required, path to input dataset containing packets
         - ``ts_dset_name`` : ``str``, required, path to input dataset containing corrected packet timestamps
         - ``larpix_trigger_channels`` : ``dict`` of ``<chip key> : [<channels>]`` pairs, optional
         - ``pacman_trigger_enabled`` : ``bool``, optional, true to extract pacman-level external triggers
         - ``pacman_trigger_word_filter`` : ``int``, optional, bitmask for pacman trigger word (3 == trigger bits 0 and 1 indicate external trigger)

        Both ``packets_dset_name`` and ``ts_dset_name`` are required in the data
        cache.

        The parameter `pacman_trigger_enabled` configures the `ExternalTriggerFinder` to
        extract packets of `packet_type == 7` as external triggers

        The parameter `larpix_trigger_channels` configures the `ExternalTriggerFinder` to
        extract triggers on particular larpix channels as external triggers. To specify,
        this parameter should be a dict of `<chip-key>: [<channel id>]` pairs. A special
        chip key of `'All'` can be used in the event that all triggers on a particular
        channel of any chip key should be extracted as external triggers.

        Example config::

            ext_trig_finder:
              classname: ExternalTriggerFinder
              requires:
                - 'charge/packets'
                - name: 'charge/packets_corr_ts'
                  path: ['charge/packets', 'charge/packets_corr_ts'] # get corrected timestamps for each packet
              params:
                ext_trigs_dset_name: 'charge/ext_trigs'
                packets_dset_name: 'charge/packets'
                ts_dset_name: 'charge/packets_corr_ts'
                pacman_trigger_enabled: True
                pacman_trigger_word_filter: 2

        ``ext_trigs`` datatype::

            id          u8, unique identifier per event
            ts          f8, corrected PPS timestamp [ticks]
            ts_raw      u8, PPS timestamp [ticks]
            type        i2, trigger type from PACMAN
            iogroup     u1, PACMAN id

    '''
    class_version = '1.0.0'

    default_pacman_trigger_enabled = True
    default_pacman_trigger_word_filter = 2
    default_larpix_trigger_channels = dict()

    ext_trigs_dtype = np.dtype([
        ('id', 'u8'),  # unique identifier
        ('ts', 'f8'),  # corrected PPS timestamp [ticks]
        ('ts_raw', 'u8'),  # PPS timestamp [ticks]
        ('type', 'i2'),  # trigger type (from PACMAN)
        ('iogroup', 'u1')  # PACMAN identifier
    ])
    larpix_trigger_channels_dtype = np.dtype([('key', h5py.string_dtype(encoding='utf-8')), ('val', h5py.vlen_dtype('u1'))])

    def __init__(self, **params):
        super(ExternalTriggerFinder, self).__init__(**params)

        self._larpix_trigger_channels = params.get('larpix_trigger_channels', self.default_larpix_trigger_channels)
        self._pacman_trigger_enabled = params.get('pacman_trigger_enabled', self.default_pacman_trigger_enabled)
        self._pacman_trigger_word_filter = params.get('pacman_trigger_word_filter', self.default_pacman_trigger_word_filter)
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')
        self.ts_dset_name = params.get('ts_dset_name')
        self.set_parameters()

    def init(self, source_name):
        super(ExternalTriggerFinder, self).init(source_name)

        # write all configuration variables to the dataset
        self.data_manager.set_attrs(self.ext_trigs_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    packets_dset=self.packets_dset_name,
                                    ts_dset=self.ts_dset_name,
                                    **self.get_parameters('pacman_trigger_enabled', 'pacman_trigger_word_filter')
                                    )
        larpix_trigger_channels = self.get_parameters('larpix_trigger_channels')['larpix_trigger_channels']
        larpix_trigger_channels_arr = np.empty((len(larpix_trigger_channels.keys()),), dtype=self.larpix_trigger_channels_dtype)
        for i, (key, val) in enumerate(larpix_trigger_channels.items()):
            larpix_trigger_channels_arr[i]['key'] = key
            larpix_trigger_channels_arr[i]['val'] = np.array(val)
        self.data_manager.set_attrs(self.ext_trigs_dset_name,
                                    larpix_trigger_channels=larpix_trigger_channels_arr
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.ext_trigs_dset_name, dtype=self.ext_trigs_dtype)
        self.data_manager.create_ref(source_name, self.ext_trigs_dset_name)

    def run(self, source_name, source_slice, cache):
        super(ExternalTriggerFinder, self).run(source_name, source_slice, cache)

        packets_data = cache[self.packets_dset_name]
        ts_data = cache[self.ts_dset_name].reshape(packets_data.shape)

        # find/join external triggers
        trigs = self.fit(packets_data, dict(ts=ts_data))

        # write external triggers datasets
        mask = ~rfn.structured_to_unstructured(trigs.mask).any(axis=-1)
        trigs_array = trigs[mask]
        trigs_slice = self.data_manager.reserve_data(self.ext_trigs_dset_name, len(trigs_array))
        trigs_idcs = np.arange(trigs_slice.start, trigs_slice.stop, dtype=int)
        trigs_array['id'] = trigs_idcs
        self.data_manager.write_data(self.ext_trigs_dset_name, trigs_slice, trigs_array)

        # write references
        #   just raw event -> trigs refs for now
        ev_id = np.expand_dims(np.arange(source_slice.start, source_slice.stop), axis=-1)
        ev_id = np.broadcast_to(ev_id, trigs.shape)
        ref = np.c_[ev_id[mask], trigs_array['id']]
        self.data_manager.write_ref(source_name, self.ext_trigs_dset_name, ref)

    def get_parameters(self, *args):
        rv = dict()
        for key in ('pacman_trigger_enabled', 'pacman_trigger_word_filter', 'larpix_trigger_channels'):
            if key in args or not args:
                rv[key] = getattr(self, '_{}'.format(key))
        return rv

    def set_parameters(self, **kwargs):
        self._pacman_trigger_enabled = kwargs.get(
            'pacman_trigger_enabled', self._pacman_trigger_enabled)
        self._larpix_trigger_channels = kwargs.get(
            'larpix_trigger_channels', self._larpix_trigger_channels)

    def fit(self, events, metadata):
        '''
        Pull external triggers from hit data within each event. ``metadata``
        is a ``dict`` of ``ts``: <array of clock corrected timestamps, same shape as events>

        Trigger types are inherited from the pacman trigger type bits (with
        `pacman_trigger_enabled`) or are given a value of `-1` for larpix external triggers.

        :returns: a list of a list of dicts (one list for each event), each dict describes a single external trigger with the following keys: `ts`-trigger timestamp, `type`-trigger type, `mask`-mask for which packets within the event are included in the trigger

        '''
        trigger_mask = np.zeros(events.shape, dtype=bool)

        if self._pacman_trigger_enabled:
            trigger_mask = (events['packet_type'] == 7)
            trigger_mask[trigger_mask] = (events['trigger_type'][trigger_mask] & self._pacman_trigger_word_filter).astype(bool)

        if self._larpix_trigger_channels:
            for chip_key, channels in self._larpix_trigger_channels.items():
                if chip_key == 'All':
                    key_mask = trigger_mask
                else:
                    io_group, io_channel, chip_id = chip_key.split('-')
                    key_mask = np.logical_and.reduce((
                        io_group == events['io_group'],
                        io_channel == events['io_channel'],
                        chip_id == events['chip_id'],
                        trigger_mask
                    ))
                for channel in channels:
                    trigger_mask = (((events['channel_id'] == channel) & key_mask)
                                    | trigger_mask)

        trigger_mask = trigger_mask & ~rfn.structured_to_unstructured(events.mask).any(axis=-1)

        trigs = np.empty(events.shape, dtype=self.ext_trigs_dtype)
        trigs['ts'] = metadata['ts']['ts']
        trigs['ts_raw'] = events['timestamp']
        trigs['type'] = events['trigger_type'] * (events['packet_type'] != 0) + -1 * (events['packet_type'] == 0)
        trigs['iogroup'] = events['io_group']
        return ma.array(trigs, mask=~trigger_mask)
