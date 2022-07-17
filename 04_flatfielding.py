import argparse
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

# get the master calibration frames
dark_master = "dark_master.fits"
bias_master = "bias_master.fits"
flat_master_B = "flat_master_B.fits"
flat_master_V = "flat_master_V.fits"
flat_master_R = "flat_master_R.fits"
flat_master_Ha = "flat_master_Ha.fits"
flat_master = {
    "B": flat_master_B,
    "V": flat_master_V,
    "R": flat_master_R,
    "HA": flat_master_Ha,
}

# We don't need shutter flats for H-alpha
shutter_flat_master_B02 = "shutter_flat_master_B02.fits"
shutter_flat_master_B05 = "shutter_flat_master_B05.fits"
shutter_flat_master_V = "shutter_flat_master_V.fits"
shutter_flat_master_R = "shutter_flat_master_R.fits"
shutter_flat_master = {
    "B02": shutter_flat_master_B02,
    "B05": shutter_flat_master_B05,
    "V": shutter_flat_master_V,
    "R": shutter_flat_master_R,
}

if os.path.exists(bias_master):
    # Load the bias master if exists
    bias_master_fits = fits.open(bias_master, memmap=False)
else:
    raise ValueError(
        "Bias master frame: {} does not exist.".format(bias_master)
    )

if os.path.exists(dark_master):
    # Load the dark master if exists
    dark_master_fits = fits.open(dark_master, memmap=False)
else:
    raise ValueError(
        "Dark master frame: {} does not exist.".format(dark_master)
    )

for folder_i in folder_name:

    filelist_light_raw = get_filelist(folder_i, frame_type="light")

    for filename in filelist_light_raw:
        # Get the filter
        _filter = filename.split("-")[2]
        outfile_name, outfile_extension = os.path.splitext(filename)
        outfile_name += "-reduced"
        # Get the file path
        filepath = os.path.join(folder_i, filename)
        outfile_filepath = os.path.join(folder_i, outfile_name + outfile_extension)
        # Load the light frame and exposure time
        light_fits = fits.open(filepath, memmap=False)
        light_fits_data = light_fits[0].data
        light_fits_header = light_fits[0].header
        exp_time = np.float64(light_fits_header["EXPTIME"])
        # scaled dark subtract
        light_fits_data = (
            light_fits_data - np.float64(dark_master_fits[0].data) * exp_time
        )
        # bias subtract
        light_fits_data = light_fits_data - np.float64(bias_master_fits[0].data)
        # Load the appropriate flat frame
        if _filter.upper() == "B":
            if exp_time == 0.2:
                _filter = _filter + "02"
            elif exp_time == 0.5:
                _filter = _filter + "05"
            else:
                raise ValueError(
                    "There isn't shutter flat for B band with exposure time: {}.".format(
                        exp_time
                    )
                )
            flat_fits = fits.open("flat_master_B.fits", memmap=False)
        else:
            flat_fits = fits.open(
                "flat_master_{}.fits".format(_filter), memmap=False
            )
        if _filter.upper() != "HA":
            shutter_flat_fits = fits.open(
                "shutter_flat_master_{}.fits".format(_filter), memmap=False
            )
            # Correct for the shutter shade, divide the ratio to the frame
            shutter_ratio = flat_fits[0].data / shutter_flat_fits[0].data
        light_fits_data = light_fits_data / np.float64(flat_fits[0].data)
        # correct for shutter shade if not in H-alpha
        if _filter.upper() != "HA":
            light_fits_data /= shutter_ratio
        reduced_light_fits = fits.PrimaryHDU(light_fits_data, light_fits_header)
        for i, dark_frame_name in enumerate(dark_master_fits[0].header):
            if dark_frame_name.startswith("FRAME"):
                reduced_light_fits.header["DARK_" + str(i)] = dark_frame_name
        for i, bias_frame_name in enumerate(bias_master_fits[0].header):
            if bias_frame_name.startswith("FRAME"):
                reduced_light_fits.header["BIAS_" + str(i)] = bias_frame_name
        for i, flat_frame_name in enumerate(flat_fits[0].header):
            if flat_frame_name.startswith("FRAME"):
                reduced_light_fits.header["FLAT_" + str(i)] = flat_frame_name
        reduced_light_fits.writeto(outfile_filepath, overwrite=True)

        print("Reduced frame is saved to {}.".format(outfile_filepath))
