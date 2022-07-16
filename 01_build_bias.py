import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units
from ccdproc import Combiner

from .get_filelist import get_filelist


parser = argparse.ArgumentParser()

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


# get the master calibration frames
bias_master = "bias_master.fits"


for folder_i in folder_name:

    filelist_bias_raw = get_filelist(folder_i, frame_type="bias")

    # Create or add to the master_bias
    # All biass are converted into count per second
    # add the nightly biass to the master if nightly biass are available
    if filelist_bias_raw != []:
        n_bias_nightly = 0
        n_bias_master = 0
        bias_ccddata_list = []
        bias_nightly_frame_name = []
        bias_master_frame_name = []
        # if master bias exists
        if os.path.exists(bias_master):
            # Load the bias master if exists
            bias_master_fits = fits.open(bias_master, memmap=False)
            bias_master_fits_header = bias_master_fits[0].header
            # Go through the constitute bias frames name
            for i in bias_master_fits_header:
                if i.startswith("FRAME"):
                    bias_master_frame_name.append([i])
                    n_bias_master += 1
        # Go through the new bias frames name
        for filename in filelist_bias_raw:
            if filename not in bias_master_frame_name:
                filepath = os.path.join(folder_i, filename)
                fits_data = fits.open(filepath, memmap=False)[0]
                bias_ccddata_list.append(
                    CCDData(fits_data.data, header=fits_data.header, unit=units.count)
                )
                n_bias_nightly += 1
        if bias_ccddata_list != []:
            bias_master_nightly_data = Combiner(bias_ccddata_list, dtype=np.float64)
            bias_master_nightly_data.minmax_clipping(max_clip=65000)
            bias_master_nightly_data.sigma_clipping()
            bias_master_nightly_data_combined = (
                bias_master_nightly_data.average_combine()
            )
            bias_master_nightly_fits = fits.PrimaryHDU(
                bias_master_nightly_data_combined, fits.Header()
            )
            for i, filename in enumerate(filelist_bias_raw):
                bias_master_nightly_fits.header["FRAME_" + str(i)] = filename
            bias_master_nightly_fits.writeto(
                os.path.join(folder_i, "bias_master_nightly_{}.fits".format(folder_i)),
                overwrite=True,
            )
            # Update master bias
            if os.path.exists(bias_master):
                # weighted average combine of the nightly and total master bias
                new_bias_master_data = Combiner(
                    [
                        CCDData(
                            bias_master_fits[0].data,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                        CCDData(
                            bias_master_nightly_data_combined,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                    ],
                    dtype=np.float64,
                )
                new_bias_master_data.weights = np.array([n_bias_master, n_bias_nightly])
                new_bias_master_data.sigma_clipping()
                new_bias_master_data_combined = new_bias_master_data.average_combine()
                new_bias_master_fits = fits.PrimaryHDU(
                    new_bias_master_data_combined, bias_master_fits_header
                )
                bias_master_fits = None
                del bias_master_fits
                for i, filename in enumerate(bias_nightly_frame_name):
                    new_bias_master_fits.header[
                        "FRAME_" + str(i + n_bias_master)
                    ] = filename
                new_bias_master_fits.writeto("bias_master.fits", overwrite=True)
                new_bias_master_fits = None
                del new_bias_master_fits
            else:
                bias_master_nightly_fits.writeto("bias_master.fits", overwrite=True)
                bias_master_nightly_fits = None
                del bias_master_nightly_fits
    else:
        print("No new bias is added from {}.".format(folder_i))
