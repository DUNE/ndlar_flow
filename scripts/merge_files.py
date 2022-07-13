'''
This script uses the runlist text file or the file name to merge datasets from two sets of files ainto a common 
file.

The script will copy all datasets from files in group 1 and only missing datasets from group 2.

If a runlist file is provided, the charge filename will be matched against files in group 1 and the light filename will be matched against files in group 2
'''

import os
import argparse
import h5py
import tqdm
import multiprocessing

def match_files(filename, grp2_files, outpath, runlist, suffix, prefix, verbose):
    reduced_filename = os.path.basename(filename.removesuffix(suffix[0])).removeprefix(prefix[0])
    if runlist is None:
        match = [reduced_filename in os.path.basename(other.removesuffix(suffix[1])).removeprefix(prefix[1])
                 for other in grp2_files]
        try:
            i_match = match.index(True)
        except ValueError:
            if verbose:
                print(f'Failed to match {reduced_filename}')
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

                if cfile in reduced_filename:
                    for i,light_filename in enumerate(grp2_files):
                        if lfile in os.path.basename(light_filename.removesuffix(suffix[1])).removeprefix(prefix[0]) or cfile in os.path.basename(light_filename.removesuffix(suffix[1])).removeprefix(prefix[0]):
                            i_match = i
                            break
                    break
        if i_match is None:
            if verbose:
                print(f'Failed to match {reduced_filename} in {os.path.basename(runlist)}')
            return False

    grp2_filename = grp2_files[i_match]
    if verbose:
        print(f'Merging {os.path.basename(filename)} and {os.path.basename(grp2_filename)}')
    with h5py.File(os.path.join(outpath, os.path.basename(filename)), 'a') as fo:
        if filename != fo.filename:
            with h5py.File(filename, 'r') as fi:
                fi.visititems(lambda k,v: fi.copy(fi[k], fo, name=k) if isinstance(v, h5py.Group) and not k in fo else None)

        if grp2_filename != fo.filename:
            with h5py.File(grp2_filename, 'r') as fi:
                fi.visititems(lambda k,v: fi.copy(fi[k], fo, name=k) if isinstance(v, h5py.Group) and not k in fo else None)
            
    return True
    

def main(outpath, grp1_input, grp2_input, processes, runlist, suffix, prefix, verbose, **kwargs):
    processes = processes if processes is not None else multiprocessing.cpu_count()
    processes = min(processes, len(grp1_input))
    
    print(f'Running on {processes} processes...')
    with multiprocessing.Pool(processes) as p:
        results = []
        for charge_file in grp1_input:
            results.append(p.apply_async(match_files,
                                         tuple(),
                                         dict(
                                             filename=charge_file,
                                             grp2_files=grp2_input,
                                             outpath=outpath,
                                             suffix=suffix,
                                             prefix=prefix,
                                             verbose=verbose,
                                             runlist=runlist)))
        failed = 0
        to_finish = list(range(len(results)))
        with tqdm.tqdm(total=len(to_finish), smoothing=0) as pbar:
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

    print(f'Failed match on {failed}/{len(grp1_input)} files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outpath', '-o', type=str,
                        required=True, help='''output directory for combined files''')
    parser.add_argument('--grp1_input', '-i1', nargs='+', type=str, required=True,
                        help='''input module0_flow files to use for charge data''')
    parser.add_argument('--grp2_input', '-i2', nargs='+', type=str, required=True,
                        help='''input module0_flow files to use for light data''')
    parser.add_argument('--processes', '-p', type=int, default=None,
                        help='''number of files to process in parallel (defaults to number of cpus detected)''')
    parser.add_argument('--runlist', '-r', type=str, default=None,
                        help='''An optional runlist file (as spec'd in the RunData resource) to use to look up matching files (default is to do matching based on same filename)''')
    parser.add_argument('--suffix', nargs=2, type=str, default=('',''),
                        metavar=('GRP1','GRP2'),
                        help='''A suffix to remove from files before matching''')
    parser.add_argument('--prefix', nargs=2, type=str, default=('',''),
                        metavar=('GRP1','GRP2'),
                        help='''A prefix to remove from files before matching''')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='''Get info from each process''')
    args = parser.parse_args()

    main(**vars(args))
