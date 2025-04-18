#!/usr/bin/env python3

# Paste the output of this script into e.g.
# `yamls/ndlar_fow/reco/light/LightEventGeneratorMC.yaml` in the `params`
# section

# Assume 4 ADCs per module, as in FSD

# Assume larnd-sim outputs 6 LCMs, then 6 ACLs, then 6 LCMs, then 6 ACLs, ...

# In output, assume first ADC is ACL, then next is LCM, etc.

N_MODULES = 35
ADCS_PER_MODULE = 4
CHANS_PER_MODULE = 240
SIPMS_PER_TILE = 6


def gen_adc_sn():
    print('  adc_sn:')
    for i in range(N_MODULES * ADCS_PER_MODULE):
        print(f'   - {i}')


def gen_channel_map():
    print('  channel_map:')
    for i in range(N_MODULES * ADCS_PER_MODULE):
        if i % 2 == 0: # ACL ADC
            offset = SIPMS_PER_TILE
        else:
            offset = 0

        start0 = CHANS_PER_MODULE//2 * (i // 2) + offset

        print('   - [', end='')

        for j in range(5):
            start = start0 + j * 24
            vals = list(range(start, start+6)) + list(range(start+12, start+18))
            vals = vals[::-1]
            print(', '.join(map(str, vals)), end='')

            if j == 4:
                print(']\n')
            else:
                # Pad with a -1 in between blocks of 10 (total 64 channels per
                # ADC, including 4 unused)
                print(',\n      -1,\n      ', end='')


def main():
    gen_adc_sn()
    print()
    gen_channel_map()


if __name__ == '__main__':
    main()
