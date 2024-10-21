# Import relevant libraries/packages
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from datetime import datetime
import glob
import json
import cmasher as cmr
import math
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import sys
import os
import sys
sys.path.append('/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_sim/run-ndlar-flow/ndlar_flow/event_display/LAr_evd/')
from lar2x2_evd import *
from collections import Counter
from multiprocessing import Pool, Value
import io
import gc
from PyPDF2 import PdfMerger


# Global variable to store the LArEventDisplay object
evd = None

# Function to initialize the global LArEventDisplay object
def init_evd(filedir, filename, show_light, beam_only):
    global evd
    evd = LArEventDisplay(filedir=filedir, filename=filename, nhits_min=0, ntrigs=0, show_light=show_light, show_colorbars=True, beam_only=beam_only)


# Function to process a single event
def process_event(ev_id):
    global evd
    hits_ini, cmap_ini, charge_norm_ini = evd.display_event(ev_id)

    # Save to in-memory buffer 
    buf = io.BytesIO()
    #output_file = output_dir + f.split('.hdf5')[0] + '_event_' + str(ev_id) + '.png'
    plt.savefig(buf, bbox_inches='tight', dpi=320)
    #plt.close()
    buf.seek(0) # moves file to beginning of buffer

    # Fix issue with colors (alpha channel -- if we don't do this, the background will be black in the PDF)
    img = Image.open(buf)
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])

    # Save image with fixed image to buffer
    buf = io.BytesIO()
    background.save(buf, 'PNG', compress_level=0, optimize=True)
    buf.seek(0)

    # Close all figures and clear memory
    del img, background
    gc.collect()

    return buf


# Make set of PNGs for every event in file
def main(file, output_dir=None, beam_only=False, show_light=False, n_evts=None):

    # Get file name and directory from input
    f = file.split('/')[-1]
    d = file.split('packet')[0]

    # Get output directory
    if output_dir is None:
        output_dir = os.path.dirname(__file__) + '/'
    else:
        output_dir = output_dir

    # Determine output file name based on beam_only and show_light
    pdf_file_modifiers = ''
    if beam_only:
        pdf_file_modifiers += '_beam_only'
    else:
        pdf_file_modifiers += '_all_events'
    if show_light:
        pdf_file_modifiers += '_show_light'
    else:
        pdf_file_modifiers += '_charge_only'

    # Initialize the LArEventDisplay object (global variable)
    init_evd(d, f, show_light, beam_only)
    event_ids = evd.events['id']

    # Go through all events in file
    if n_evts and n_evts < len(event_ids):
        num_events = n_evts
        event_ids = event_ids[:num_events]
    else:
        num_events = len(event_ids)
    print('Number of events to plot: ', num_events)

    # Create PNGs, then store to intermediary PDF
    # Process events in small batches to avoid memory issues
    total_cpus = os.cpu_count()
    min_cpus = 1
    max_cpus = 32
    if total_cpus == 1:
        cpus_to_use = 1
    else:
        cpus_to_use = int(total_cpus/8)
        if cpus_to_use > max_cpus:
            cpus_to_use = max_cpus
    evts_per_cpu = int(num_events/cpus_to_use)
    if evts_per_cpu < 1:
        cpus_to_use = num_events
    batch_size = cpus_to_use
    batch_id = 0
    batch_pdfs = []

    for i in range(0, num_events, batch_size):
        batch_id += 1
        print('------- PROCESSING EVENTS ', i, 'TO', i+batch_size, 'OF', num_events, '-------')
        if i+batch_size < num_events:
            batch_ids = event_ids[i:i+batch_size]
        else:
            batch_ids = event_ids[i:]
        args = [ev_id for ev_id in batch_ids]

        # Create a pool of workers to process events in parallel
        with Pool(initializer=init_evd, initargs=(d, f, show_light, beam_only), processes=cpus_to_use) as pool:
            output_imgs = pool.map(process_event, args)

        # Combine all PNGs into a single PDF
        images = [Image.open(buf) for buf in output_imgs]
        pdf_path = output_dir + f.split('.hdf5')[0] + '_EVENT_DISPLAYS'+pdf_file_modifiers+'_BATCH'+str(batch_id)+'.pdf'
        batch_pdfs.append(pdf_path)
        images[0].save(pdf_path, "PDF" ,resolution=320.0, save_all=True, append_images=images[1:])

    # Merge PDFs
    output_file = output_dir + f.split('.hdf5')[0] + '_EVENT_DISPLAYS'+pdf_file_modifiers+'.pdf'
    merger = PdfMerger()
    for pdf in batch_pdfs:
        merger.append(pdf)
    with open(output_file, 'wb') as fout:
        merger.write(fout)
    print('Final PDF saved to: ', output_file)

    # Remove intermediary PDFs
    for pdf in batch_pdfs:
        os.remove(pdf)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', default=None,type=str,help='''String of input file and location''')
    parser.add_argument('-od','--output_dir', default=None,type=str,help='''String of output file directory location. If no directory is given, current directory is used.''')
    parser.add_argument('-b','--beam_only', default=False, type=bool, help='''Bool telling whether or not to only save beam events.''')
    parser.add_argument('-l','--show_light', default=False, type=bool, help='''Bool telling whether or not to show light.''')
    parser.add_argument('-n','--n_evts', default=None, type=int, help='''Number of events to plot.''')
    args = parser.parse_args()
    main(**vars(args))
