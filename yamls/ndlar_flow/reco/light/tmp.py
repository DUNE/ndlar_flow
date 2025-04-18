#!/usr/bin/env python3


for i_mod in range(1, 35):
    offset = 240 * i_mod
    def count(start, incr):
        return range(offset + start, offset + start + incr*30, incr)
    l0 = [*count(180, 1),  *[-1, -1], *count(210, 1),  *[-1, -1]]
    l1 = [*count(179, -1), *[-1, -1], *count(149, -1), *[-1, -1]]
    l2 = [*count(119, -1), *[-1, -1], *count(89, -1),  *[-1, -1]]
    l3 = [*count(0, 1),    *[-1, -1], *count(30, 1),   *[-1, -1]]
    for l in [l0, l1, l2, l3]:
        print(f'   - {l}')
