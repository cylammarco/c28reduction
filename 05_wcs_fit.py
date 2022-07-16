import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner

from .get_filelist import get_filelist


parser = argparse.ArgumentParser()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


def closest_nonzero(lst, start_index):
    nonzeros = [(i, x) for i, x in enumerate(lst) if x != 0]
    sorted_nonzeros = sorted(nonzeros, key=lambda x: abs(x[0] - start_index))
    idx = np.argmin(np.vstack(sorted_nonzeros)[:, 1])
    return sorted_nonzeros[idx][1]


def wcs_fit(filelist):
    obs_time = np.zeros(len(filelist))
    for i, filename in enumerate(filelist):
        # WCS fit
        filepath = os.path.join(filename)
        subprocess.call(
            "solve-field {} --ra='19:21:43.6' --dec='-15:57:18' --radius=1.0 --downsample=5 --scale-low=0.825 --scale-high=0.845 --scale-units='arcsecperpix' --depth=20 --resort --cpulimit 30000".format(
                filepath
            ),
            shell=True,
        )
        obs_time[i] = float(fits.open(filename, memmap=False)[0].header["JD"])

    # If WCS fit failed, apply the wcs from a frame with the least temporal difference
    # Only do this after all frames are tried to fit with a WCS
    #
    # Get the filelist of all the (supposedly) WCS fitted light frames
    filelist_wcs_fitted = [os.path.splitext(i)[0] + ".new" for i in filelist]

    obs_time_with_wcs = copy.deepcopy(obs_time)
    for i, filepath in enumerate(filelist_wcs_fitted):
        # If the wcs is not fitted, set the time to -999.0
        if not os.path.exists(filepath):
            obs_time_with_wcs[i] = -99999.0

    for idx, filepath in enumerate(filelist_wcs_fitted):
        # If the wcs is not fitted, find the nearest one
        if not os.path.exists(filepath):
            fits_to_add_wcs = fits.open(
                os.path.splitext(filepath)[0] + ".fts", memmap=False
            )[0]
            abs_diff = np.abs(obs_time_with_wcs - obs_time[i])
            closest_idx = np.where(abs_diff == closest_nonzero(abs_diff, i))[0][0]
            wcs_ref_filepath = filelist_wcs_fitted[closest_idx]
            wcs_reference = WCS(fits.open(wcs_ref_filepath, memmap=False)[0].header)
            fits_to_add_wcs.header.update(wcs_reference.to_header())
            fits_to_add_wcs.writeto(
                os.path.join(filepath),
                overwrite=True,
            )


for folder_i in folder_name:

    filelist_light_reduced_all = get_filelist(folder_name, frame_type="reduced")

    # Do WCS fit on all of the light frames
    wcs_fit(filelist_light_reduced_all)
