import numpy as np
import h5py
import logging

from h5flow.core import H5FlowStage

class ExternalTriggerFinder(H5FlowStage):
    '''
    A class to extract external triggers from packet arrays

    This class has three parameters: `pacman_trigger_enabled`, `pacman_trigger_word_filter`, `larpix_trigger_channels`

    The parameter `pacman_trigger_enabled` configures the `ExternalTriggerFinder` to
    extract packets of `packet_type == 7` as external triggers

    The parameter `larpix_trigger_channels` configures the `ExternalTriggerFinder` to
    extract triggers on particular larpix channels as external triggers. To specify,
    this parameter should be a dict of `<chip-key>: [<channel id>]` pairs. A special
    chip key of `'All'` can be used in the event that all triggers on a particular
    channel of any chip key should be extracted as external triggers.

    You can access and set the parameters at initialization::

        etf = ExternalTriggerFinder(pacman_trigger_enabled=True, larpix_trigger_channels=dict())

    or via the getter/setters::

        etf.get_parameters() # dict(pacman_trigger_enabled=True, larpix_trigger_channels=dict())
        etf.get_parameters('pacman_trigger_enabled') # dict(pacman_trigger_enabled=True)

        etf.set_parameters(pacman_trigger_enabled=True, larpix_trigger_channels={'1-1-1':[0]})

    '''
    class_version = '0.0.0'

    ext_trigs_dtype = np.dtype([
        ('trig_id', 'u8'),
        ('ts', 'f8'),
        ('ts_raw', 'i8'),
        ('type', 'i2'),
        ('iogroup', 'u1')
        ])
    larpix_trigger_channels_dtype = np.dtype([('key',h5py.string_dtype(encoding='utf-8')), ('val',h5py.vlen_dtype('u1'))])

    def __init__(self, pacman_trigger_enabled=True, pacman_trigger_word_filter=2, larpix_trigger_channels=None, **params):
        super(ExternalTriggerFinder, self).__init__(**params)

        if larpix_trigger_channels is None:
            larpix_trigger_channels = dict()
        self._larpix_trigger_channels = larpix_trigger_channels
        self._pacman_trigger_enabled = pacman_trigger_enabled
        self._pacman_trigger_word_filter = pacman_trigger_word_filter
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')
        self.ts_dset_name = params.get('ts_dset_name')
        self.set_parameters()

    def init(self, source_name):
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
        for i,(key,val) in enumerate(larpix_trigger_channels.items()):
            larpix_trigger_channels_arr[i]['key'] = key
            larpix_trigger_channels_arr[i]['val'] = np.array(val)
        self.data_manager.set_attrs(self.ext_trigs_dset_name,
            larpix_trigger_channels=larpix_trigger_channels_arr
            )

        # then set up new datasets
        self.data_manager.create_dset(self.ext_trigs_dset_name, dtype=self.ext_trigs_dtype)
        self.data_manager.create_ref(source_name, self.ext_trigs_dset_name)

    def run(self, source_name, source_slice, cache):
        packets_data = cache[self.packets_dset_name]
        ts_data = cache[self.ts_dset_name]

        # find/join external triggers
        trigs = self.fit(packets_data, dict(ts=ts_data))

        # write external triggers datasets
        trigs_array = np.concatenate(trigs, axis=0) if len(trigs) else np.empty((0,), dtype=self.ext_trigs_dtype)
        trigs_slice = self.data_manager.reserve_data(self.ext_trigs_dset_name, len(trigs_array))
        trigs_idcs = np.arange(trigs_slice.start, trigs_slice.stop)
        trigs_array['trig_id'] = trigs_idcs
        self.data_manager.write_data(self.ext_trigs_dset_name, trigs_slice, trigs_array)

        # write references
        #   just raw event -> trigs refs for now
        self.data_manager.reserve_ref(source_name, self.ext_trigs_dset_name, source_slice)
        ref = [trigs_idcs[i:i+len(trig)] for i,trig in enumerate(trigs)]
        self.data_manager.write_ref(source_name, self.ext_trigs_dset_name, source_slice, ref)

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
        is a ``dict`` of ``ts``: <list of clock corrected timestamps>

        Trigger types are inherited from the pacman trigger type bits (with
        `pacman_trigger_enabled`) or are given a value of `-1` for larpix external triggers.

        :returns: a list of a list of dicts (one list for each event), each dict describes a single external trigger with the following keys: `ts`-trigger timestamp, `type`-trigger type, `mask`-mask for which packets within the event are included in the trigger

        '''
        if metadata is None:
            metadata = dict()
        event_trigs = list()
        for i,event in enumerate(events):
            event_trigs.append(list())

            if self._pacman_trigger_enabled:
                # first check for any pacman trigger packets
                trigger_mask = event['packet_type'] == 7
                trigger_mask[trigger_mask] = (event['trigger_type'][trigger_mask] & self._pacman_trigger_word_filter).astype(bool)
                if np.any(trigger_mask) > 0:
                    for i, trigger in enumerate(event[trigger_mask]):
                        mask = np.zeros(len(event), dtype=bool)
                        mask[i] = 1
                        event_trigs[-1].append(dict(
                            ts=metadata['ts'][i][mask],
                            ts_raw=trigger['timestamp'],
                            type=trigger['trigger_type'],
                            iogroup=trigger['io_group'],
                            mask=mask
                        ))

            if self._larpix_trigger_channels:
                # then check for larpix external triggers
                trigger_mask = np.zeros(len(event), dtype=bool)
                for chip_key, channels in self._larpix_trigger_channels.items():
                    if chip_key == 'All':
                        key_mask = trigger_mask
                    else:
                        io_group, io_channel, chip_id = chip_key.split('-')
                        key_mask = np.logical_and.reduce((
                            io_group == event['io_group'],
                            io_channel == event['io_channel'],
                            chip_id == event['chip_id'],
                            trigger_mask
                        ))
                    for channel in channels:
                        trigger_mask = np.logical_or(np.logical_and(
                            event['channel_id'] == channel, key_mask), trigger_mask)
                trigger_mask = np.logical_and(
                    event['packet_type'] == 0, trigger_mask)
                if np.any(trigger_mask) > 0:
                    timestamps = event[trigger_mask]['timestamps']
                    split_indices = np.argwhere(
                        np.abs(np.diff(timestamps > 0)))
                    for idx, trigger in zip(split_indices, np.split(event[trigger_mask], split_indices)):
                        mask = np.zeros(len(event), dtype=bool)
                        mask[trigger_mask][idx:idx+len(trigger)] = 1
                        event_trigs[-1].append(dict(
                            ts=np.median(metadata['ts'][i][mask]),
                            ts_raw=np.median(trigger['timestamp']),
                            type=-1,
                            iogroup=np.median(trigger['io_group']),
                            mask=mask
                        ))

        # format into numpy arrays
        rv = list()
        for trigs in event_trigs:
            arr = np.empty((len(trigs),), dtype=self.ext_trigs_dtype)
            arr['ts'] = [trig['ts'] for trig in trigs]
            arr['ts_raw'] = [trig['ts_raw'] for trig in trigs]
            arr['type'] = [trig['type'] for trig in trigs]
            arr['iogroup'] = [trig['iogroup'] for trig in trigs]
            rv.append(arr)
        return rv
