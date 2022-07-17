import argparse
import copy
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units
from ccdproc import Combiner

from get_filelist import get_filelist


parser = argparse.ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


output_folder = "output"
# Create output folder for photometry
if not os.path.exists(output_folder):

    os.mkdir(output_folder)


for folder_i in folder_name:

    filelist_dark_raw = get_filelist(folder_i, frame_type="dark")

    # get the master calibration frames
    dark_master = "dark_master.fits"
    bias_master = "bias_master.fits"

    if os.path.exists(bias_master):
        # Load the bias master if exists
        bias_master_fits = fits.open(bias_master, memmap=False)
    else:
        raise ValueError(
            "Bias master frame: {} does not exist. Please run 01_build_bias.py first".format(
                bias_master
            )
        )

    # Create or add to the master_dark
    # All darks are converted into count per second
    # add the nightly darks to the master if nightly darks are available
    if filelist_dark_raw != []:
        n_dark_nightly = 0
        n_dark_master = 0
        dark_ccddata_list = []
        dark_master_frame_name = []
        # if master dark exists
        if os.path.exists(dark_master):
            # Load the dark master if exists
            dark_master_fits = fits.open(dark_master, memmap=False)
            dark_master_fits_header = copy.deepcopy(dark_master_fits[0].header)
            # Go through the constitute dark frames name
            for i in dark_master_fits_header:
                if i.startswith("FRAME"):
                    dark_master_frame_name.append(dark_master_fits_header[i])
                    n_dark_master += 1
        # Go through the new dark frames name
        for filename in filelist_dark_raw:
            if filename not in dark_master_frame_name:
                filepath = os.path.join(folder_i, filename)
                fits_data = fits.open(filepath, memmap=False)[0]
                dark_ccddata_list.append(
                    CCDData(
                        (fits_data.data - bias_master_fits[0].data)
                        / float(fits_data.header["EXPTIME"]),
                        header=fits_data.header,
                        unit=units.count,
                    )
                )
                n_dark_nightly += 1
        if dark_ccddata_list != []:
            dark_master_nightly_data = Combiner(dark_ccddata_list, dtype=np.float64)
            dark_master_nightly_data.minmax_clipping(max_clip=65000)
            dark_master_nightly_data.sigma_clipping()
            dark_master_nightly_data_combined = (
                dark_master_nightly_data.average_combine()
            )
            dark_master_nightly_fits = fits.PrimaryHDU(
                dark_master_nightly_data_combined, fits.Header()
            )
            for i, filename in enumerate(filelist_dark_raw):
                dark_master_nightly_fits.header["FRAME_" + str(i)] = filename
            dark_master_nightly_fits.writeto(
                os.path.join(folder_i, "dark_master_nightly_{}.fits".format(folder_i)),
                overwrite=True,
            )
            # Update master dark
            if os.path.exists(dark_master):
                # weighted average combine of the nightly and total master dark
                new_dark_master_data = Combiner(
                    [
                        CCDData(
                            dark_master_fits[0].data,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                        CCDData(
                            dark_master_nightly_data_combined,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                    ],
                    dtype=np.float64,
                )
                new_dark_master_data.weights = np.array([n_dark_master, n_dark_nightly])
                new_dark_master_data.sigma_clipping()
                new_dark_master_data_combined = new_dark_master_data.average_combine()
                new_dark_master_fits = fits.PrimaryHDU(
                    new_dark_master_data_combined, dark_master_fits_header
                )
                dark_master_fits = None
                del dark_master_fits
                for i, filename in enumerate(filelist_dark_raw):
                    new_dark_master_fits.header[
                        "FRAME_" + str(i + n_dark_master)
                    ] = filename
                new_dark_master_fits.writeto("dark_master.fits", overwrite=True)
                new_dark_master_fits = None
                del new_dark_master_fits
            else:
                dark_master_nightly_fits.writeto("dark_master.fits", overwrite=True)
                dark_master_nightly_fits = None
                del dark_master_nightly_fits
    else:
        print("No new dark is added from {}.".format(folder_i))
