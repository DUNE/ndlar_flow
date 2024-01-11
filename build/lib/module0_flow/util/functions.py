import pandas as pd
import subprocess
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


def plot_eff_maps(all_hist, hit_hist, extent, module, iogroup, plot_dir):

    fig, ax = plt.subplots(5, 5, figsize=(38, 30))

    print(hit_hist)
    print(all_hist)

    hist = hit_hist/all_hist

    hist_in_segs = [hist[0:20, 80:100], hist[20:40, 80:100], hist[40:60, 80:100], hist[60:80, 80:100], hist[80:100, 80:100],
                    hist[0:20, 60:80], hist[20:40, 60:80], hist[40:60, 60:80], hist[60:80, 60:80], hist[80:100, 60:80],
                    hist[0:20, 40:60], hist[20:40, 40:60], hist[40:60, 40:60], hist[60:80, 40:60], hist[80:100, 40:60],
                    hist[0:20, 20:40], hist[20:40, 20:40], hist[40:60, 20:40], hist[60:80, 20:40], hist[80:100, 20:40],
                    hist[0:20, 0:20], hist[20:40, 0:20], hist[40:60, 0:20], hist[60:80, 0:20], hist[80:100, 0:20]]


    cmap = plt.get_cmap('rainbow')
    norm = Normalize(vmin=np.min(hist), vmax=np.max(hist))

    k = 0

    for i in range(5):
        for j in range(5):

            pos = ax[i, j].imshow(hist_in_segs[k].T, cmap=cmap, norm=norm, origin='lower', extent=[extent[k][0], extent[k][1], extent[k][2], extent[k][3]])
            ax[i, j].tick_params(axis='x', which='major', labelsize=45)
            ax[i, j].tick_params(axis='y', which='major', labelsize=45)

            if i == 0 or i == 1 or i == 2 or i == 3:
                ax[i, j].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)

                if j == 1 or j == 2 or j == 3 or j == 4:
                    ax[i, j].tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False,  # ticks along the bottom edge are off
                        right=False,  # ticks along the top edge are off
                        labelleft=False)

            if i == 4:
                if j == 1 or j == 2 or j == 3 or j == 4:
                    ax[i, j].tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False,  # ticks along the bottom edge are off
                        right=False,  # ticks along the top edge are off
                        labelleft=False)

            k = k + 1

            ax[i, j].set_aspect('equal', adjustable='box')

        cbar_ax = fig.add_axes([0.965, 0.15, 0.01, 0.7])  # [x, y, width, height]
        cbar = fig.colorbar(pos, cax=cbar_ax, pad=10)
        cbar.ax.tick_params(labelsize=25)

    fig.add_subplot(111, frameon=False)
    fig.tight_layout()

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("X distance [in mm] to pixel center", fontsize='45', labelpad=40)
    plt.ylabel("Y distance [in mm] to pixel center", fontsize='45', labelpad=-30)

    plt.tight_layout()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if not os.path.exists(plot_dir + '/' + str(module) + '_iogroup_cbar_' + str(iogroup) + '5_x_5_pixel_overview.pdf'):
        plt.savefig(plot_dir + '/' + str(module) + '_iogroup_cbar_' + str(iogroup) + '5_x_5_pixel_overview.pdf', format = 'pdf')

    return 0

def get_extent_25(ps):

    extent_5x5 = [

        [-5 * ps / 2, -3 * ps / 2, 3 * ps / 2, 5 * ps / 2], [-3 * ps / 2, -ps / 2, 3 * ps / 2, 5 * ps / 2],
        [-ps / 2, ps / 2, 3 * ps / 2, 5 * ps / 2], [ps / 2, 3 * ps / 2, 3 * ps / 2, 5 * ps / 2],
        [3 * ps / 2, 5 * ps / 2, 3 * ps / 2, 5 * ps / 2],

        [-5 * ps / 2, -3 * ps / 2, ps / 2, 3 * ps / 2], [-3 * ps / 2, -ps / 2, ps / 2, 3 * ps / 2],
        [-ps / 2, ps / 2, ps / 2, 3 * ps / 2], [ps / 2, 3 * ps / 2, ps / 2, 3 * ps / 2],
        [3 * ps / 2, 5 * ps / 2, ps / 2, 3 * ps / 2],

        [-5 * ps / 2, -3 * ps / 2, -ps / 2, ps / 2], [-3 * ps / 2, -ps / 2, -ps / 2, ps / 2],
        [-ps / 2, ps / 2, -ps / 2, ps / 2], [ps / 2, 3 * ps / 2, -ps / 2, ps / 2],
        [3 * ps / 2, 5 * ps / 2, -ps / 2, ps / 2],

        [-5 * ps / 2, -3 * ps / 2, -3 * ps / 2, -ps / 2], [-3 * ps / 2, -ps / 2, -3 * ps / 2, -ps / 2],
        [-ps / 2, ps / 2, -3 * ps / 2, -ps / 2], [ps / 2, 3 * ps / 2, -3 * ps / 2, -ps / 2],
        [3 * ps / 2, 5 * ps / 2, -3 * ps / 2, -ps / 2],

        [-5 * ps / 2, -3 * ps / 2, -5 * ps / 2, -3 * ps / 2], [-3 * ps / 2, -ps / 2, -5 * ps / 2, -3 * ps / 2],
        [-ps / 2, ps / 2, -5 * ps / 2, -3 * ps / 2], [ps / 2, 3 * ps / 2, -5 * ps / 2, -3 * ps / 2],
        [3 * ps / 2, 5 * ps / 2, -5 * ps / 2, -3 * ps / 2]
    ]

    return extent_5x5

def remove_close_values(values, y):
    values.sort()  # Sort the values in ascending order
    i = 0
    while i < len(values) - 1:
        if abs(values[i] - values[i + 1]) < y:
            # Remove one of the close values
            del values[i + 1]
        else:
            i += 1

def pixel_bins_z_mid():

    ps = 4.434

    bins_pixels_z_ht = np.arange(0, 605, ps)

    return bins_pixels_z_ht    

def pixel_bins_mid_z_mod2():

    ps = 3.8

    bins_pixels_z_ht = np.arange(0, 605, ps)

    return bins_pixels_z_ht   

def d_pnt2line_test_XY(df, df_hough_w_endpoints):

    df_True = df.copy(deep=True)

    A = df_True[['start_X', 'start_Y']]
    B = np.array([df_hough_w_endpoints['xh_i'], df_hough_w_endpoints['yh_i']]).T
    C = np.array([df_hough_w_endpoints['xh_f'], df_hough_w_endpoints['yh_f']]).T

    for idx, (b, c) in enumerate(zip(B, C)):
        d = (c - b) / np.linalg.norm(c - b)
        v = A - b
        t = np.dot(v, d)
        P = np.array([t_temp * d + b for t_temp in t])

        df_True['d_hl_' + str(idx) + 'x_res'] = (A - P)['start_X']
        df_True['d_hl_' + str(idx) + 'y_res'] = (A - P)['start_Y']

        DIST = np.sqrt(np.einsum('ij,ij->i', P - A, P - A))
        df_True['d_hl_' + str(idx)] = DIST

    return df_True

def get_pixel_bins(module, side):

    p = 0
    ps = 4.434

    if side == 'edge':
        p = ps/2

    if module == 'mod0':

        bins_pixel_x_ht = np.arange(-308.173 - p, -3, ps)
        bins_pixel_x_ht2 = -np.arange(-308.173 - p, -3, ps)[::-1]
        bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

        bins_pixel_y_ht = -np.arange(-308.173 - p, -3, ps)[::-1]
        bins_pixel_y_ht2 = np.arange(316.661 - p, 612 + p, ps)
        bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
        bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]


    if module == 'mod1':

        bins_pixel_x_ht = np.arange(-308.163 - p, -1 - p, ps)
        bins_pixel_x_ht2 = -np.arange(-308.163 - p, -1 - p, ps)[::-1]
        bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

        bins_pixel_y_ht = -np.arange(-303.729 - p, -3, ps)[::-1]
        bins_pixel_y_ht2 = np.arange(317.031 - p, 615, ps)
        bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
        bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    if module == 'mod2':

        ps = 3.8
        p = ps/2

        bins_pixel_x_ht = np.arange(-305.29 - p, -4 + p, ps)
        bins_pixel_x_ht2 = -np.arange(-305.29 - p, -4 + p, ps)[::-1]
        bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

        bins_pixel_y_ht = -np.arange(-305.29 - p, -5 + p, ps)[::-1]
        bins_pixel_y_ht2 = np.arange(315.47 - p, 616 + p, ps)
        bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
        bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    if module == 'mod3':

        bins_pixel_x_ht = np.arange(-308.163 - p, -1, ps)
        bins_pixel_x_ht2 = -np.arange(-308.163 - p, -1, ps)[::-1]
        bins_pixel_x = np.r_[bins_pixel_x_ht, 0, bins_pixel_x_ht2]

        bins_pixel_y_ht = -np.arange(-614.109 - p, 0, ps)[::-1]
        bins_pixel_y = np.r_[-bins_pixel_y_ht[::-1], 0, bins_pixel_y_ht]

    return bins_pixel_x, bins_pixel_y


def HoughOnArray(hits, id, min_pixels=125, MAX_peaks=100):

    mask = (hits[0]['iogroup']==1) | (hits[0]['iogroup']==2) & (hits[0]['z']>0) & (hits[0]['z']<315)
    hits_use = hits[0][mask]

    file_in = 'singlemod_coordinates' + str(id) + '.csv'
    file_out = "./singlemod_hough3D_outputfile" + str(id) +".csv"

    if len(hits_use) < 10:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros([len(hits)])

    np.savetxt(file_in, hits_use[['x', 'y', 'z']], delimiter=',', header='x,y,z')

    subprocess.check_call([r"./Hough3Dpackage/hough3dlines",
                           file_in,
                           "-o", file_out,
                           "-minvotes", str(min_pixels),
                           "-nlines", str(MAX_peaks),
                           "-raw"])


    with open(file_out, 'r') as fin:

        data = fin.read().splitlines(True)

        if len(data) < 3:

            if os.path.exists(file_in):
                os.remove(file_in)

            if os.path.exists(file_out):
                os.remove(file_out)

            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros([len(hits)])

        t_min = float(np.floor(float(data[1:][0].split()[0])))
        aX, aY, aZ, bX, bY, bZ, npoints = data[2:][0].split()

        xh_i = float(aX) - t_min * float(bX)
        xh_f = float(aX) + t_min * float(bX)
        yh_i = float(aY) - t_min * float(bY)
        yh_f = float(aY) + t_min * float(bY)
        zh_i = float(aZ) - t_min * float(bZ)
        zh_f = float(aZ) + t_min * float(bZ)


    if os.path.exists(file_in):
        os.remove(file_in)

    if os.path.exists(file_out):
        os.remove(file_out)

    return float(aX), float(aY), float(aZ), float(bX), float(bY), float(bZ), int(npoints), float(xh_i), float(xh_f), float(yh_i), float(yh_f), float(zh_i), float(zh_f), mask

def d_pnt2line(hits, xh_i, xh_f, yh_i, yh_f, zh_i, zh_f):

    A = np.array([hits[0]['x'], hits[0]['y'], hits[0]['z']])
    B = np.array([[xh_i, yh_i, zh_i]])
    C = np.array([[xh_f, yh_f, zh_f]])


    for idx, (b, c) in enumerate(zip(B, C)):
        d = (c - b) / np.linalg.norm(c - b)
        b = b.reshape(3, 1)
        d = d.reshape(3, 1)
        v = A - b
        t = np.dot(v.T, d)
        P = np.array([t_temp * d.T + b.T for t_temp in t]).reshape(len(hits[0]), 3).T
        DIST = np.sqrt(np.einsum('ij,ij->i', (P - A).T, (P - A).T))

    return DIST

def trackLengthhoughcenter(hits, aX, aY, aZ, bX, bY, bZ, radius, dist):

    hits_cp = np.array([hits[0]['x'], hits[0]['y'], hits[0]['z'], hits[0]['Q'], hits[0]['E'], hits[0]['id'], hits[0]['next']])
    mask = dist < radius
    hits_cp = hits_cp.T[mask]
    dist_cp = dist[mask]

    if len(hits_cp) == 0:
        return 0, 0, 0, 0, np.zeros([len(hits)])

    avg = np.array([hits_cp[:,0] - aX,
                    hits_cp[:,1] - aY,
                    hits_cp[:,2] - aZ])

    hits_cp_avg_dist = np.linalg.norm(avg, axis=0)

    avg2 = np.array([hits_cp[:,0] - hits_cp[:,0][np.argmax(hits_cp_avg_dist)],
                     hits_cp[:,1] - hits_cp[:,1][np.argmax(hits_cp_avg_dist)],
                     hits_cp[:,2] - hits_cp[:,2][np.argmax(hits_cp_avg_dist)]])

    avg_dist_2 = np.linalg.norm(avg2, axis=0)

    unit = np.array([(hits_cp[:,0][np.argmax(hits_cp_avg_dist)] - aX) / bX,
                    (hits_cp[:,1][np.argmax(hits_cp_avg_dist)] - aY) / bY,
                    (hits_cp[:,2][np.argmax(hits_cp_avg_dist)] - aZ) / bZ])

    if np.all((unit < 0)):
        coeff = -1
    else:
        coeff = 1

    shift = hits_cp[:,0][np.argmax(hits_cp_avg_dist)] - (
                hits_cp[:,0][np.argmax(avg_dist_2)] -
                hits_cp[:,0][np.argmax(hits_cp_avg_dist)])

    shift *= 0.5

    aX_nl = aX + shift * coeff * bX
    aY_nl = aY + shift * coeff * bY
    aZ_nl = aZ + shift * coeff * bZ

    return np.max(avg_dist_2), aX_nl, aY_nl, aZ_nl, mask
