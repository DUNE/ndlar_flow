'''
This script uses the runlist text file or the file name to copy data from charge and light files into a common 
file

'''

import os
import argparse
import h5py
import tqdm
import multiprocessing

def match_files(filename, light_files, outpath, runlist):
    if runlist is None:
        match = [os.path.basename(filename) in os.path.basename(other) for other in light_files]
        try:
            i_match = match.index(True)
        except ValueError:
            return False
    else:
        i_match = None
        with open(runlist,'r') as f:
            for line in f.readlines()[1:]:
                fields = line.split()
                if not len(fields):
                    continue
                cfile = fields[1]
                lfile = fields[2]

                if cfile in os.path.basename(filename):
                    for i,light_filename in enumerate(light_files):
                        if lfile in os.path.basename(light_filename) or cfile in os.path.basename(light_filename):
                            i_match = i
                            break
                    break
        if i_match is None:
            return False

    light_filename = light_files[i_match]
    with h5py.File(os.path.join(outpath, os.path.basename(filename)), 'a') as fo:
        if not 'charge' in fo:
            with h5py.File(filename, 'r') as fi:
                fi.visititems(lambda k,v: fi.copy(fi[k], fo, name=k) if isinstance(v, h5py.Group) and not k in fo else None)

        with h5py.File(light_filename, 'r') as fi:
            fi.visititems(lambda k,v: fi.copy(fi[k], fo, name=k) if isinstance(v, h5py.Group) and not k in fo else None)
            
    return True
    

def main(outpath, charge_input, light_input, processes, runlist, **kwargs):
    processes = processes if processes is not None else multiprocessing.cpu_count()
    processes = min(processes, len(charge_input))
    
    print(f'Running on {processes} processes...')
    with multiprocessing.Pool(processes) as p:
        results = []
        for charge_file in charge_input:
            results.append(p.apply_async(match_files,
                                         tuple(),
                                         dict(
                                             filename=charge_file,
                                             light_files=light_input,
                                             outpath=outpath,
                                             runlist=runlist)))
        failed = 0
        to_finish = list(range(len(results)))
        pbar = tqdm.tqdm(total=len(to_finish), smoothing=0)
        while to_finish:
            i = 0
            results[to_finish[i]].wait(1)            
            while i < min(processes, len(to_finish)):
                if results[to_finish[i]].ready():
                    failed += int(not results[to_finish[i]].get())
                    del to_finish[i]
                    pbar.update()
                else:
                    i += 1

    print(f'Failed match on {failed}/{len(charge_input)} files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outpath', '-o', type=str,
                        required=True, help='''output directory for combined files''')
    parser.add_argument('--charge_input', '-c', nargs='+', type=str, required=True,
                        help='''input module0_flow files to use for charge data''')
    parser.add_argument('--light_input', '-l', nargs='+', type=str, required=True,
                        help='''input module0_flow files to use for light data''')
    parser.add_argument('--processes', '-p', type=int, default=None,
                        help='''number of files to process in parallel (defaults to number of cpus detected)''')
    parser.add_argument('--runlist', '-r', type=str, default=None,
                        help='''An optional runlist file (as spec'd in the RunData resource) to use to look up matching files (default is to do matching based on same filename)''')
    args = parser.parse_args()

    main(**vars(args))
