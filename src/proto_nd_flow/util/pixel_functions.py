# functions for pixel/packet-related calculations (that don't make sense to be included in a LUT)

def get_pixel_unique_ids(packets, tile_ids):
        '''
        Helper function to get pixel unique ids for an array of charge packets. 

        :param packets: array of packets
        :param tile_ids: array of tile_ids, calculated with resources['Geometry'].tile_id
        :returns: ``unique ids array``

        '''
        #tile_id = resources['Geometry'].tile_id[packets['io_group'], packets['io_channel']]
        unique_ids = (packets['io_group'].astype(int)*1000_000_000
                            + tile_ids.astype(int)*100_000
                            + packets['chip_id'].astype(int)*100
                            + packets['channel_id'].astype(int))
        return unique_ids

def adc2mv(adc, vref, vcm, adc_counts):
    '''
    Helper function to convert pedestal-corrected ADC datawords to mV units.
    
    :param adc: array of ADC values
    :param vref: pixel vref configuration [mV] 
    :param vcm: pixel vcm configuration [mV]
    :param adc_counts: nominally 2^N, where N is the total bits in the ADC (usually 8)
    :returns: array of pedestal-corrected datawords converted to mV units
    '''
    return (vref-vcm) * adc/adc_counts + vcm

def dac2mv(dac, vdda, adc_counts):
    '''
    Helper function to convert vref or vcm DAC values to mV units.

    :param dac: DAC value int value
    :param vdda: VDDA voltage (usually ~1800 mV) [mV]
    :param adc_counts: nominally 2^N, where N is the total bits in the ADC (usually 8)
    '''
    return vdda * dac/adc_counts