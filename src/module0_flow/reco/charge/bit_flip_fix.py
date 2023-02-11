import numpy as np
import numpy.ma as ma
import logging

from h5flow.core import H5FlowStage, resources

class BitFlipFix(H5FlowStage):
    '''
    Fixes chip id bit flips for known, easily recognizable chip ids

    Example config::

      bit_flip_fix:
        classname: BitFlipFix
        path: module0_flow.reco.charge.bit_flip_fix
        requires:
          - 'charge/packets'
          - name: 'charge/packet_idx'
            path: ['charge/packets']
            index_only: True
        params:
          packets_name: 'charge/packets'
          packet_idx_name: 'charge/packet_idx'
          overwrite: True # True if file contents should be overwritten, otherwise just modifies chip ids in memory
          bit_flips: # which chip ids to modify
            <io_group>:
              <io_channel>:
                <good chip_id>: [<known bad chip ids>]
    '''
    class_version = '0.0.0'
    defaults = dict(
        packets_name='charge/packets',
        packet_idx_name='charge/packets_idx',
        overwrite=False,
        bit_flips=None
        )

    def __init__(self, **params):
        super(BitFlipFix, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
            
        if self.bit_flips is None:
            self.bit_flips = dict()

        # validate bit flip config
        for io_group in self.bit_flips:
            assert isinstance(io_group, int), f'Bad io group: {io_group}'
            for io_channel in self.bit_flips[io_group]:
                assert isinstance(io_channel, int), f'Bad io channel: {io_channel}'
                for chip_id in self.bit_flips[io_group][io_channel]:
                    assert isinstance(chip_id, int), f'Bad chip id: {chip_id}'
                    other_chip_ids = self.bit_flips[io_group][io_channel][chip_id]
                    assert isinstance(other_chip_ids, list), f'Bad bit-flip ids: {other_chip_ids}'
                    for other_id in other_chip_ids:
                        assert isinstance(other_id, int), f'Bad chip id: {other_id}'


    def init(self, source_name):
        super(BitFlipFix, self).init(source_name)

        if self.overwrite:
            # if overwriting data, keep a record of the settings used
            bit_flips = np.array([
                (io_group, io_channel, chip_id, flip_id)
                for io_group in self.bit_flips
                for io_channel in self.bit_flips[io_group]
                for chip_id in self.bit_flips[io_group][io_channel]
                for flip_id in self.bit_flips[io_group][io_channel][chip_id]
            ], dtype=np.dtype([('io_group','u1'), ('io_channel','u1'), ('chip_id', 'u1'), ('flip_id', 'u1')]))
            
            self.data_manager.set_attrs(self.packets_name,
                                       bit_flip_classname=self.classname,
                                       bit_flip_class_version=self.class_version,
                                       bit_flip_ids=bit_flips)
        

    def run(self, source_name, source_slice, cache):
        super(BitFlipFix, self).run(source_name, source_slice, cache)

        packets = cache[self.packets_name]
        packet_idx = cache[self.packet_idx_name]

        # find all packets that are impacted by bit flip
        packet_mask = ~(packets.mask['io_group'].copy())
        mask = np.zeros_like(packet_mask)
        for io_group, io_channel_spec in self.bit_flips.items():
            for io_channel, chip_spec in io_channel_spec.items():
                for chip_id, flip_ids in chip_spec.items():
                    for flip_id in flip_ids:
                        flip_mask = ((packets['io_group'] == io_group)
                            & (packets['io_channel'] == io_channel)
                            & (packets['chip_id'] == flip_id)
                            & packet_mask)

                        if np.any(flip_mask):
                            # mark modified packets
                            mask[flip_mask] = True

                            # update packet chip id
                            np.place(packets['chip_id'], flip_mask, chip_id)

        logging.info(f'flipped bits on {int(mask.sum())}/{int((~packets.mask["io_group"]).sum())} packets')

        # update output file
        if self.overwrite:
            idx = np.extract(mask, packet_idx)
            new_packet = np.extract(mask, packets)
            order = np.argsort(idx) # needs to be sorted for h5py
            if len(idx):
                self.data_manager.write_data(self.packets_name, idx[order], new_packet[order])
