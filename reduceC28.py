import argparse
import copy
import glob
import os
import subprocess
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner
from reproject import reproject_adaptive

sys.path.append("/home/mlam/git/py-hotpants")
from pyhotpants import *


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
            "solve-field {} --ra 19:21:43.6 --dec -15:57:18 --radius 2 --cpulimit 30000".format(
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


parser = argparse.ArgumentParser(description="Some minimal control of data reduction.")
parser.add_argument(
    "--calibration-frame-only",
    action="store_true",
    help="Set to accumulate calibration frames only without reducing light frames.",
)
parser.add_argument(
    "--bias-frame-only",
    action="store_true",
    help="Set to accumulate bias frames only.",
)
parser.add_argument(
    "--dark-frame-only",
    action="store_true",
    help="Set to accumulate dark frames only.",
)
parser.add_argument(
    "--flat-frame-only",
    action="store_true",
    help="Set to accumulate flat frames only.",
)
parser.add_argument(
    "--flatfielding-only",
    action="store_true",
    help="Set to perform field flattening only.",
)
parser.add_argument(
    "--wcs-fit-only",
    action="store_true",
    help="Set to fit wcs only.",
)
parser.add_argument(
    "--build-nightly-stack",
    action="store_true",
    help="(re)build the nightly stack in all filters.",
)
parser.add_argument(
    "--build-nightly-stack-only",
    action="store_true",
    help="(re)build the nightly stack in all filters only.",
)
parser.add_argument(
    "--build-total-stack",
    action="store_true",
    help="(re)build the total stack in all filters.",
)
parser.add_argument(
    "--build-total-stack-only",
    action="store_true",
    help="(re)build the total stack in all filters only.",
)
parser.add_argument(
    "--difference-photometry-only",
    action="store_true",
    help="(re)compte the photometry in all filters only.",
)

parser.add_argument("--folder")
args = parser.parse_args()

ups_sgr_coord = SkyCoord(290.43176441262, -15.95504344758, unit="deg")

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


for folder_i in folder_name:

    # Get all the files
    filelist_all = os.listdir(folder_i)

    # Get all the dark, bias, flat, light frames
    filelist_light_raw = []
    filelist_dark_raw = []
    filelist_bias_raw = []

    filelist_flat_B_raw = []
    filelist_flat_V_raw = []
    filelist_flat_R_raw = []
    filelist_flat_Ha_raw = []

    filelist_shutter_flat_B02_raw = []
    filelist_shutter_flat_B05_raw = []
    filelist_shutter_flat_V_raw = []
    filelist_shutter_flat_R_raw = []

    for filename in filelist_all:
        if (
            filename.startswith("UpsilonSgr")
            & filename.endswith("fts")
            & (not filename.endswith("reduced.fts"))
        ):
            filelist_light_raw.append(filename)
        elif filename.startswith("Dark"):
            filelist_dark_raw.append(filename)
        elif filename.startswith("Bias"):
            filelist_bias_raw.append(filename)
        elif filename.startswith("FF"):
            _name = os.path.splitext(filename)[0].split("-")[-1]
            if _name.upper() == "B":
                filelist_flat_B_raw.append(filename)
            elif _name.upper() == "V":
                filelist_flat_V_raw.append(filename)
            elif _name.upper() == "R":
                filelist_flat_R_raw.append(filename)
            elif _name.upper() == "HA":
                filelist_flat_Ha_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_name))
                print("It is not handled.")
        elif filename.startswith("AutoFlat"):
            _name = os.path.splitext(filename)[0].split("-")[-2]
            if _name.upper() == "B":
                filelist_flat_B_raw.append(filename)
            elif _name.upper() == "V":
                filelist_flat_V_raw.append(filename)
            elif _name.upper() == "R":
                filelist_flat_R_raw.append(filename)
            elif _name.upper() == "HA":
                filelist_flat_Ha_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_name))
                print("It is not handled.")
        elif filename.startswith("SFF"):
            _name = os.path.splitext(filename)[0].split("-")[-1]
            if _name == "B02":
                filelist_shutter_flat_B02_raw.append(filename)
            elif _name == "B05":
                filelist_shutter_flat_B05_raw.append(filename)
            elif _name == "V":
                filelist_shutter_flat_V_raw.append(filename)
            elif _name == "R":
                filelist_shutter_flat_R_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_name))
                print("It is not handled.")
        else:
            pass

    filelist_flat_raw_all = [
        filelist_flat_B_raw,
        filelist_flat_V_raw,
        filelist_flat_R_raw,
        filelist_flat_Ha_raw,
    ]
    filelist_shutter_flat_raw_all = [
        filelist_shutter_flat_B02_raw,
        filelist_shutter_flat_B05_raw,
        filelist_shutter_flat_V_raw,
        filelist_shutter_flat_R_raw,
    ]

    if not (
        args.dark_frame_only
        or args.flat_frame_only
        or args.wcs_fit_only
        or args.flatfielding_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
        or args.difference_photometry_only
    ):

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
                        CCDData(
                            fits_data.data, header=fits_data.header, unit=units.count
                        )
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
                    os.path.join(
                        folder_i, "bias_master_nightly_{}.fits".format(folder_i)
                    ),
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
                    new_bias_master_data.weights = np.array(
                        [n_bias_master, n_bias_nightly]
                    )
                    new_bias_master_data.sigma_clipping()
                    new_bias_master_data_combined = (
                        new_bias_master_data.average_combine()
                    )
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

    if not (
        args.bias_frame_only
        or args.flat_frame_only
        or args.wcs_fit_only
        or args.flatfielding_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
        or args.difference_photometry_only
    ):

        if os.path.exists(bias_master):
            # Load the bias master if exists
            bias_master_fits = fits.open(bias_master, memmap=False)
        else:
            raise ValueError(
                "Bias master frame: {} does not exist.".format(bias_master)
            )

        # Create or add to the master_dark
        # All darks are converted into count per second
        # add the nightly darks to the master if nightly darks are available
        if filelist_dark_raw != []:
            n_dark_nightly = 0
            n_dark_master = 0
            dark_ccddata_list = []
            dark_nightly_frame_name = []
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
                    os.path.join(
                        folder_i, "dark_master_nightly_{}.fits".format(folder_i)
                    ),
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
                    new_dark_master_data.weights = np.array(
                        [n_dark_master, n_dark_nightly]
                    )
                    new_dark_master_data.sigma_clipping()
                    new_dark_master_data_combined = (
                        new_dark_master_data.average_combine()
                    )
                    new_dark_master_fits = fits.PrimaryHDU(
                        new_dark_master_data_combined, dark_master_fits_header
                    )
                    dark_master_fits = None
                    del dark_master_fits
                    for i, filename in enumerate(dark_nightly_frame_name):
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

    if not (
        args.bias_frame_only
        or args.dark_frame_only
        or args.flatfielding_only
        or args.wcs_fit_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
        or args.difference_photometry_only
    ):

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

        # Create or add to the master_flat
        # All flats are converted into count per second
        # add the nightly flats to the master if nightly flats are available
        for filelist_flat_raw in filelist_flat_raw_all:
            if filelist_flat_raw != []:
                _filename_split = os.path.splitext(filelist_flat_raw[0])[0].split("-")
                if _filename_split[0] == "FF":
                    _filter = _filename_split[-1].upper()
                else:
                    _filter = _filename_split[-2].upper()
                print("Building master flats, working with filter: {}.".format(_filter))
                n_flat_nightly = 0
                n_flat_master = 0
                flat_ccddata_list = []
                flat_nightly_frame_name = []
                flat_master_frame_name = []
                # if master flat exists
                if os.path.exists(flat_master[_filter]):
                    # Load the flat master if exists
                    flat_master_fits = fits.open(flat_master[_filter], memmap=False)
                    flat_master_fits_header = copy.deepcopy(flat_master_fits[0].header)
                    # Go through the constitute flat frames name
                    for i in flat_master_fits_header:
                        if i.startswith("FRAME"):
                            flat_master_frame_name.append(flat_master_fits_header[i])
                            n_flat_master += 1
                # Go through the new flat frames name
                for filename in filelist_flat_raw:
                    filename_temp = folder_i + "-" + filename
                    if filename_temp not in flat_master_frame_name:
                        filepath = os.path.join(folder_i, filename)
                        fits_data = fits.open(filepath, memmap=False)[0]
                        flat_ccddata_list.append(
                            CCDData(
                                fits_data.data
                                - dark_master_fits[0].data
                                * float(fits_data.header["EXPTIME"])
                                - bias_master_fits[0].data,
                                header=fits_data.header,
                                unit=units.count,
                            )
                        )
                        n_flat_nightly += 1
                if flat_ccddata_list != []:
                    flat_master_nightly_data = Combiner(
                        flat_ccddata_list, dtype=np.float64
                    )
                    flat_master_nightly_data.minmax_clipping(max_clip=65000)
                    flat_master_nightly_data.sigma_clipping()
                    flat_master_nightly_data_combined = (
                        flat_master_nightly_data.average_combine()
                    )
                    flat_master_nightly_data_combined.data = (
                        flat_master_nightly_data_combined.data
                        / np.nanmean(flat_master_nightly_data_combined.data)
                    )
                    flat_master_nightly_fits = fits.PrimaryHDU(
                        flat_master_nightly_data_combined,
                        fits.Header(),
                    )
                    for i, filename in enumerate(filelist_flat_raw):
                        flat_nightly_frame_name.append(filename)
                        flat_master_nightly_fits.header["FRAME_" + str(i)] = (
                            folder_i + "-" + filename
                        )
                    # Update master flat
                    if os.path.exists(flat_master[_filter]):
                        # weighted average combine of the nightly and total master flat
                        new_flat_master_data = Combiner(
                            [
                                CCDData(
                                    flat_master_fits[0].data,
                                    header=fits_data.header,
                                    unit=units.count,
                                ),
                                CCDData(
                                    flat_master_nightly_data_combined,
                                    header=fits_data.header,
                                    unit=units.count,
                                ),
                            ],
                            dtype=np.float64,
                        )
                        new_flat_master_data.weights = np.array(
                            [n_flat_master, n_flat_nightly]
                        )
                        new_flat_master_data.sigma_clipping()
                        new_flat_master_data_combined = (
                            new_flat_master_data.average_combine()
                        )
                        flat_master_fits.close()
                        flat_master_fits = None
                        del flat_master_fits
                        new_flat_master_fits = fits.PrimaryHDU(
                            new_flat_master_data_combined, flat_master_fits_header
                        )
                        for i, filename in enumerate(flat_nightly_frame_name):
                            filename_temp = folder_i + "-" + filename
                            new_flat_master_fits.header[
                                "FRAME_" + str(i + n_flat_master)
                            ] = filename_temp
                        new_flat_master_fits.writeto(
                            flat_master[_filter], overwrite=True
                        )
                        new_flat_master_fits = None
                        del new_flat_master_fits
                    else:
                        flat_master_nightly_fits.writeto(
                            flat_master[_filter], overwrite=True
                        )
                        flat_master_nightly_fits = None
                        del flat_master_nightly_fits

        # Shutter flats here
        for filelist_flat_raw in filelist_shutter_flat_raw_all:
            if filelist_flat_raw != []:
                _filter = os.path.splitext(filelist_flat_raw[0])[0].split("-")[-1]
                print(
                    "Building shutter flats, working with filter: {}.".format(_filter)
                )
                n_flat_nightly = 0
                n_flat_master = 0
                flat_ccddata_list = []
                flat_nightly_frame_name = []
                flat_master_frame_name = []
                # if master flat exists
                if os.path.exists(shutter_flat_master[_filter]):
                    # Load the flat master if exists
                    flat_master_fits = fits.open(
                        shutter_flat_master[_filter], memmap=False
                    )
                    flat_master_fits_header = copy.deepcopy(flat_master_fits[0].header)
                    # Go through the constitute flat frames name
                    for i in flat_master_fits_header:
                        if i.startswith("FRAME"):
                            flat_master_frame_name.append(flat_master_fits_header[i])
                            n_flat_master += 1
                # Go through the new flat frames name
                for filename in filelist_flat_raw:
                    filename_temp = folder_i + "-" + filename
                    if filename_temp not in flat_master_frame_name:
                        filepath = os.path.join(folder_i, filename)
                        fits_data = fits.open(filepath, memmap=False)[0]
                        flat_ccddata_list.append(
                            CCDData(
                                fits_data.data
                                - dark_master_fits[0].data
                                * float(fits_data.header["EXPTIME"])
                                - bias_master_fits[0].data,
                                header=fits_data.header,
                                unit=units.count,
                            )
                        )
                        n_flat_nightly += 1
                if flat_ccddata_list != []:
                    flat_master_nightly_data = Combiner(
                        flat_ccddata_list, dtype=np.float64
                    )
                    flat_master_nightly_data.minmax_clipping(max_clip=65000)
                    flat_master_nightly_data.sigma_clipping()
                    flat_master_nightly_data_combined = (
                        flat_master_nightly_data.average_combine()
                    )
                    flat_master_nightly_data_combined.data = (
                        flat_master_nightly_data_combined.data
                        / np.nanmean(flat_master_nightly_data_combined.data)
                    )
                    flat_master_nightly_fits = fits.PrimaryHDU(
                        flat_master_nightly_data_combined,
                        fits.Header(),
                    )
                    for i, filename in enumerate(filelist_flat_raw):
                        flat_nightly_frame_name.append(filename)
                        flat_master_nightly_fits.header["FRAME_" + str(i)] = (
                            folder_i + "-" + filename
                        )
                    # Update master flat
                    if os.path.exists(shutter_flat_master[_filter]):
                        # weighted average combine of the nightly and total master flat
                        new_flat_master_data = Combiner(
                            [
                                CCDData(
                                    flat_master_fits[0].data,
                                    header=fits_data.header,
                                    unit=units.count,
                                ),
                                CCDData(
                                    flat_master_nightly_data_combined,
                                    header=fits_data.header,
                                    unit=units.count,
                                ),
                            ],
                            dtype=np.float64,
                        )
                        new_flat_master_data.weights = np.array(
                            [n_flat_master, n_flat_nightly]
                        )
                        new_flat_master_data.sigma_clipping()
                        new_flat_master_data_combined = (
                            new_flat_master_data.average_combine()
                        )
                        flat_master_fits.close()
                        flat_master_fits = None
                        del flat_master_fits
                        new_flat_master_fits = fits.PrimaryHDU(
                            new_flat_master_data_combined, flat_master_fits_header
                        )
                        for i, filename in enumerate(flat_nightly_frame_name):
                            filename_temp = folder_i + "-" + filename
                            new_flat_master_fits.header[
                                "FRAME_" + str(i + n_flat_master)
                            ] = filename_temp
                        new_flat_master_fits.writeto(
                            shutter_flat_master[_filter], overwrite=True
                        )
                        new_flat_master_fits = None
                        del new_flat_master_fits
                    else:
                        flat_master_nightly_fits.writeto(
                            shutter_flat_master[_filter], overwrite=True
                        )
                        flat_master_nightly_fits = None
                        del flat_master_nightly_fits

    if not (
        args.calibration_frame_only
        or args.flat_frame_only
        or args.dark_frame_only
        or args.bias_frame_only
        or args.wcs_fit_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
        or args.difference_photometry_only
    ):

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

        reduced_light_filename_list = []
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
            reduced_light_filename_list.append(outfile_filepath)
            print("Reduced frame is saved to {}.".format(outfile_filepath))

    if not (
        args.calibration_frame_only
        or args.flat_frame_only
        or args.dark_frame_only
        or args.bias_frame_only
        or args.flatfielding_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
        or args.difference_photometry_only
    ):

        # Get all the light frames in different bands
        filelist_light_reduced_B = []
        filelist_light_reduced_V = []
        filelist_light_reduced_R = []
        filelist_light_reduced_Ha = []

        B_combiner_list = []
        V_combiner_list = []
        R_combiner_list = []
        Ha_combiner_list = []

        B_exp_time_list = []
        V_exp_time_list = []
        R_exp_time_list = []
        Ha_exp_time_list = []

        # Get all the files
        filelist_all = os.listdir(folder_i)

        for filename in filelist_all:

            if filename.endswith("-reduced.fts"):

                _filter = os.path.splitext(filename)[0].split("-")[-2]
                if _filter.upper() == "B":
                    filelist_light_reduced_B.append(filename)
                elif _filter.upper() == "V":
                    filelist_light_reduced_V.append(filename)
                elif _filter.upper() == "R":
                    filelist_light_reduced_R.append(filename)
                elif _filter.upper() == "HA":
                    filelist_light_reduced_Ha.append(filename)
                else:
                    print("Unaccounted filters: {}".format(_name))
                    print("It is not handled.")

        filelist_light_reduced_all = (
            filelist_light_reduced_B
            + filelist_light_reduced_V
            + filelist_light_reduced_R
            + filelist_light_reduced_Ha
        )

        reduced_light_filename_list = []
        for filename in filelist_light_reduced_all:
            outfile_filepath = os.path.join(folder_i, filename)
            reduced_light_filename_list.append(outfile_filepath)

        # Do WCS fit on all of the light frames
        wcs_fit(reduced_light_filename_list)

        # Get the filelist of all the WCS fitted light frames
        filelist_wcs_fitted = [
            os.path.splitext(i)[0] + ".new" for i in reduced_light_filename_list
        ]

    if (args.build_nightly_stack) or (args.build_nightly_stack_only):

        # Get all the light frames in different bands
        filelist_light_reduced_B = []
        filelist_light_reduced_V = []
        filelist_light_reduced_R = []
        filelist_light_reduced_Ha = []

        B_combiner_list = []
        V_combiner_list = []
        R_combiner_list = []
        Ha_combiner_list = []

        B_exp_time_list = []
        V_exp_time_list = []
        R_exp_time_list = []
        Ha_exp_time_list = []

        # Get all the files
        filelist_all = os.listdir(folder_i)

        for filename in filelist_all:

            if filename.endswith("new"):

                _filter = os.path.splitext(filename)[0].split("-")[-2]
                if _filter.upper() == "B":
                    filelist_light_reduced_B.append(filename)
                elif _filter.upper() == "V":
                    filelist_light_reduced_V.append(filename)
                elif _filter.upper() == "R":
                    filelist_light_reduced_R.append(filename)
                elif _filter.upper() == "HA":
                    filelist_light_reduced_Ha.append(filename)
                else:
                    print("Unaccounted filters: {}".format(_name))
                    print("It is not handled.")

        filelist_light_reduced_all = (
            filelist_light_reduced_B
            + filelist_light_reduced_V
            + filelist_light_reduced_R
            + filelist_light_reduced_Ha
        )

        if len(filelist_light_reduced_all) == 0:

            print(
                "No WCS fitted frames with extension .new found in {}.".format(folder_i)
            )
            continue

        else:

            print("{} frames found.".format(len(filelist_light_reduced_all)))

        wcs_reference = WCS(
            fits.open(
                os.path.join(folder_i, filelist_light_reduced_all[0]), memmap=False
            )[0].header
        )

        # B band
        if filelist_light_reduced_B == []:

            print("There is not a nightly stack in the B this night.")

        else:

            for filename in filelist_light_reduced_B:

                fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
                wcs = WCS(fits_file[0].header)
                fits_data_reprojected = reproject_adaptive(
                    input_data=fits_file,
                    output_projection=wcs_reference,
                    shape_out=np.shape(fits_file[0].data),
                    return_footprint=False,
                )
                B_combiner_list.append(
                    CCDData(
                        fits_data_reprojected,
                        header=fits_file[0].header,
                        wcs=wcs_reference,
                        unit=units.count,
                    )
                )
                B_exp_time_list.append(fits_file[0].header["EXPTIME"])

            B_combiner = Combiner(B_combiner_list, dtype=np.float64)
            B_combiner.weights = np.array(B_exp_time_list)
            B_combiner.sigma_clipping()
            B_combined_data = B_combiner.average_combine()
            # make the cutout to 30 arcmin by 30 arcmin
            B_combined_cutout = Cutout2D(
                B_combined_data.data,
                ups_sgr_coord,
                30.0 * units.arcmin,
                wcs=wcs_reference,
                mode="partial",
            )
            # put the cutout into a PrimaryHDU
            B_combined_fits = fits.PrimaryHDU(B_combined_cutout.data, fits.Header())
            for i, filename in enumerate(filelist_light_reduced_B):
                B_combined_fits.header["FRAME_" + str(i)] = filename
            B_combined_fits.header.update(B_combined_cutout.wcs.to_header())
            B_combined_fits.header["XPOSURE"] = np.sum(B_exp_time_list)
            B_combined_fits.writeto(
                os.path.join(folder_i, "B_{}_nightly_stack.fits".format(folder_i)),
                overwrite=True,
            )

        # V band
        if filelist_light_reduced_V == []:

            print("There is not a nightly stack in the V this night.")

        else:

            for filename in filelist_light_reduced_V:

                fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
                wcs = WCS(fits_file[0].header)
                fits_data_reprojected = reproject_adaptive(
                    input_data=fits_file,
                    output_projection=wcs_reference,
                    shape_out=np.shape(fits_file[0].data),
                    return_footprint=False,
                )
                V_combiner_list.append(
                    CCDData(
                        fits_data_reprojected,
                        header=fits_file[0].header,
                        wcs=wcs_reference,
                        unit=units.count,
                    )
                )
                V_exp_time_list.append(fits_file[0].header["EXPTIME"])

            V_combiner = Combiner(V_combiner_list, dtype=np.float64)
            V_combiner.weights = np.array(V_exp_time_list)
            V_combiner.sigma_clipping()
            V_combined_data = V_combiner.average_combine()
            # make the cutout to 30 arcmin by 30 arcmin
            V_combined_cutout = Cutout2D(
                V_combined_data.data,
                ups_sgr_coord,
                30.0 * units.arcmin,
                wcs=wcs_reference,
                mode="partial",
            )
            # put the cutout into a PrimaryHDU
            V_combined_fits = fits.PrimaryHDU(V_combined_cutout.data, fits.Header())
            for i, filename in enumerate(filelist_light_reduced_V):
                V_combined_fits.header["FRAME_" + str(i)] = filename
            V_combined_fits.header.update(V_combined_cutout.wcs.to_header())
            V_combined_fits.header["XPOSURE"] = np.sum(V_exp_time_list)
            V_combined_fits.writeto(
                os.path.join(folder_i, "V_{}_nightly_stack.fits".format(folder_i)),
                overwrite=True,
            )

        # R band
        if filelist_light_reduced_R == []:

            print("There is not a nightly stack in the R this night.")

        else:

            for filename in filelist_light_reduced_R:

                fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
                wcs = WCS(fits_file[0].header)
                fits_data_reprojected = reproject_adaptive(
                    input_data=fits_file,
                    output_projection=wcs_reference,
                    shape_out=np.shape(fits_file[0].data),
                    return_footprint=False,
                )
                R_combiner_list.append(
                    CCDData(
                        fits_data_reprojected,
                        header=fits_file[0].header,
                        wcs=wcs_reference,
                        unit=units.count,
                    )
                )
                R_exp_time_list.append(fits_file[0].header["EXPTIME"])

            R_combiner = Combiner(R_combiner_list, dtype=np.float64)
            R_combiner.weights = np.array(R_exp_time_list)
            R_combiner.sigma_clipping()
            R_combined_data = R_combiner.average_combine()
            # make the cutout to 30 arcmin by 30 arcmin
            R_combined_cutout = Cutout2D(
                R_combined_data.data,
                ups_sgr_coord,
                30.0 * units.arcmin,
                wcs=wcs_reference,
                mode="partial",
            )
            # put the cutout into a PrimaryHDU
            R_combined_fits = fits.PrimaryHDU(R_combined_cutout.data, fits.Header())
            for i, filename in enumerate(filelist_light_reduced_R):
                R_combined_fits.header["FRAME_" + str(i)] = filename
            R_combined_fits.header.update(R_combined_cutout.wcs.to_header())
            R_combined_fits.header["XPOSURE"] = np.sum(R_exp_time_list)
            R_combined_fits.writeto(
                os.path.join(folder_i, "R_{}_nightly_stack.fits".format(folder_i)),
                overwrite=True,
            )

        # Ha band
        if filelist_light_reduced_Ha == []:

            print("There is not a nightly stack in the Ha this night.")

        else:

            for filename in filelist_light_reduced_Ha:

                fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
                wcs = WCS(fits_file[0].header)
                fits_data_reprojected = reproject_adaptive(
                    input_data=fits_file,
                    output_projection=wcs_reference,
                    shape_out=np.shape(fits_file[0].data),
                    return_footprint=False,
                )
                Ha_combiner_list.append(
                    CCDData(
                        fits_data_reprojected,
                        header=fits_file[0].header,
                        wcs=wcs_reference,
                        unit=units.count,
                    )
                )
                Ha_exp_time_list.append(fits_file[0].header["EXPTIME"])

            Ha_combiner = Combiner(Ha_combiner_list, dtype=np.float64)
            Ha_combiner.weights = np.array(Ha_exp_time_list)
            Ha_combiner.sigma_clipping()
            Ha_combined_data = Ha_combiner.average_combine()

            # make the cutout to 30 arcmin by 30 arcmin
            Ha_combined_cutout = Cutout2D(
                Ha_combined_data.data,
                ups_sgr_coord,
                30.0 * units.arcmin,
                wcs=wcs_reference,
                mode="partial",
            )
            # put the cutout into a PrimaryHDU
            Ha_combined_fits = fits.PrimaryHDU(Ha_combined_cutout.data, fits.Header())
            for i, filename in enumerate(filelist_light_reduced_Ha):
                Ha_combined_fits.header["FRAME_" + str(i)] = filename
            Ha_combined_fits.header.update(Ha_combined_cutout.wcs.to_header())
            Ha_combined_fits.header["XPOSURE"] = np.sum(Ha_exp_time_list)
            Ha_combined_fits.writeto(
                os.path.join(folder_i, "Ha_{}_nightly_stack.fits".format(folder_i)),
                overwrite=True,
            )


# Generate total stack here
if args.build_nightly_stack or args.build_total_stack_only:

    # Get all the light frames in different bands
    filepathlist_nightly_stack_B = []
    filepathlist_nightly_stack_V = []
    filepathlist_nightly_stack_R = []
    filepathlist_nightly_stack_Ha = []

    B_nightly_combiner_list = []
    V_nightly_combiner_list = []
    R_nightly_combiner_list = []
    Ha_nightly_combiner_list = []

    B_nightly_exp_time_list = []
    V_nightly_exp_time_list = []
    R_nightly_exp_time_list = []
    Ha_nightly_exp_time_list = []

    # Get all the nightly stack files
    for folder_i in folder_name:

        filelist_all = os.listdir(folder_i)

        for filename in filelist_all:

            if filename.endswith("nightly_stack.fits"):

                _filter = filename.split("_")[0]

                if _filter.upper() == "B":

                    filepathlist_nightly_stack_B.append(
                        os.path.join(folder_i, filename)
                    )

                elif _filter.upper() == "V":

                    filepathlist_nightly_stack_V.append(
                        os.path.join(folder_i, filename)
                    )

                elif _filter.upper() == "R":

                    filepathlist_nightly_stack_R.append(
                        os.path.join(folder_i, filename)
                    )

                elif _filter.upper() == "HA":

                    filepathlist_nightly_stack_Ha.append(
                        os.path.join(folder_i, filename)
                    )

                else:

                    print("Unaccounted filters: {}".format(_name))
                    print("It is not handled.")

    # B band
    if filepathlist_nightly_stack_B == []:

        print("There is not any nightly stack in the B filter.")

    else:

        for filepath in filepathlist_nightly_stack_B:

            fits_file = fits.open(filepath, memmap=False)
            wcs = WCS(fits_file[0].header)
            B_nightly_combiner_list.append(
                CCDData(
                    fits_data_reprojected,
                    header=fits_file[0].header,
                    wcs=wcs,
                    unit=units.count,
                )
            )
            B_nightly_exp_time_list.append(float(fits_file[0].header["XPOSURE"]))

        B_combiner = Combiner(B_nightly_combiner_list, dtype=np.float64)
        B_combiner.weights = np.array(B_exp_time_list)
        B_combiner.sigma_clipping()
        B_combined_data = B_combiner.average_combine()

        B_combined_fits = fits.PrimaryHDU(B_combined_data, fits.Header())
        for i, filename in enumerate(filepathlist_nightly_stack_B):
            B_combined_fits.header["FRAME_" + str(i)] = filepath
        B_combined_fits.header.update(wcs.to_header())
        B_combined_fits.header["XPOSURE"] = np.sum(B_nightly_exp_time_list)
        B_combined_fits.writeto(
            "B_total_stack.fits",
            overwrite=True,
        )

    # V band
    if filepathlist_nightly_stack_V == []:

        print("There is not any nightly stack in the V filter.")

    else:

        for filepath in filepathlist_nightly_stack_V:

            fits_file = fits.open(filepath, memmap=False)
            wcs = WCS(fits_file[0].header)
            V_nightly_combiner_list.append(
                CCDData(
                    fits_data_reprojected,
                    header=fits_file[0].header,
                    wcs=wcs,
                    unit=units.count,
                )
            )
            V_nightly_exp_time_list.append(float(fits_file[0].header["XPOSURE"]))

        V_combiner = Combiner(V_nightly_combiner_list, dtype=np.float64)
        V_combiner.weights = np.array(V_exp_time_list)
        V_combiner.sigma_clipping()
        V_combined_data = B_combiner.average_combine()

        V_combined_fits = fits.PrimaryHDU(V_combined_data, fits.Header())
        for i, filename in enumerate(filepathlist_nightly_stack_V):
            V_combined_fits.header["FRAME_" + str(i)] = filepath
        V_combined_fits.header.update(wcs.to_header())
        V_combined_fits.header["XPOSURE"] = np.sum(V_nightly_exp_time_list)
        V_combined_fits.writeto(
            "V_total_stack.fits",
            overwrite=True,
        )

    # R band
    if filepathlist_nightly_stack_R == []:

        print("There is not any nightly stack in the R filter.")

    else:

        for filepath in filepathlist_nightly_stack_R:

            fits_file = fits.open(filepath, memmap=False)
            wcs = WCS(fits_file[0].header)
            R_nightly_combiner_list.append(
                CCDData(
                    fits_data_reprojected,
                    header=fits_file[0].header,
                    wcs=wcs,
                    unit=units.count,
                )
            )
            R_nightly_exp_time_list.append(float(fits_file[0].header["XPOSURE"]))

        R_combiner = Combiner(R_nightly_combiner_list, dtype=np.float64)
        R_combiner.weights = np.array(R_exp_time_list)
        R_combiner.sigma_clipping()
        R_combined_data = R_combiner.average_combine()

        R_combined_fits = fits.PrimaryHDU(R_combined_data, fits.Header())
        for i, filename in enumerate(filepathlist_nightly_stack_R):
            R_combined_fits.header["FRAME_" + str(i)] = filepath
        R_combined_fits.header.update(wcs.to_header())
        R_combined_fits.header["XPOSURE"] = np.sum(R_nightly_exp_time_list)
        R_combined_fits.writeto(
            "R_total_stack.fits",
            overwrite=True,
        )

    # Ha band
    if filepathlist_nightly_stack_V == []:

        print("There is not any nightly stack in the Ha filter.")

    else:

        for filepath in filepathlist_nightly_stack_Ha:

            fits_file = fits.open(filepath, memmap=False)
            wcs = WCS(fits_file[0].header)
            Ha_nightly_combiner_list.append(
                CCDData(
                    fits_data_reprojected,
                    header=fits_file[0].header,
                    wcs=wcs,
                    unit=units.count,
                )
            )
            Ha_nightly_exp_time_list.append(float(fits_file[0].header["XPOSURE"]))

        Ha_combiner = Combiner(Ha_nightly_combiner_list, dtype=np.float64)
        Ha_combiner.weights = np.array(Ha_exp_time_list)
        Ha_combiner.sigma_clipping()
        Ha_combined_data = Ha_combiner.average_combine()

        Ha_combined_fits = fits.PrimaryHDU(Ha_combined_data, fits.Header())
        for i, filename in enumerate(filepathlist_nightly_stack_Ha):
            Ha_combined_fits.header["FRAME_" + str(i)] = filepath
        Ha_combined_fits.header.update(wcs.to_header())
        Ha_combined_fits.header["XPOSURE"] = np.sum(Ha_nightly_exp_time_list)
        Ha_combined_fits.writeto(
            "Ha_total_stack.fits",
            overwrite=True,
        )


# Difference imaging with hotpants here
if (
    not (
        args.calibration_frame_only
        or args.flat_frame_only
        or args.dark_frame_only
        or args.bias_frame_only
        or args.flatfielding_only
        or args.build_nightly_stack_only
        or args.build_total_stack_only
    )
    or args.difference_photometry_only
):

    for filter_i in ["B", "V", "R", "Ha"]:

        data_stacked = fits.open("{}_total_stack.fits".format(filter_i))

        # background subtraction for each frame
        # see also https://photutils.readthedocs.io/en/stable/background.html
        bkg = get_background(
            data_stacked[0].data,
            maxiters=10,
            box_size=(31, 31),
            filter_size=(7, 7),
            create_figure=True,
            output_folder=output_folder,
        )
        data_stacked_bkg_sub = data_stacked[0].data - bkg.background

        # Get the star stamps to build psf
        stars, stars_tbl = get_good_stars(
            data_stacked_bkg_sub,
            threshold=100.0,
            box_size=25,
            npeaks=25,
            edge_size=25,
            output_folder=output_folder,
        )

        # build the psf using the stacked image
        # see also https://photutils.readthedocs.io/en/stable/epsf.html
        epsf, fitted_stars, oversampling_factor = build_psf(
            stars,
            smoothing_kernel="quadratic",
            maxiters=20,
            create_figure=True,
            save_figure=True,
            output_folder=output_folder,
        )

        # Get the FWHM
        # for the stack
        sigma_x_stack, sigma_y_stack = fit_gaussian_for_fwhm(epsf.data, fit_sigma=True)
        sigma_x_stack /= oversampling_factor
        sigma_y_stack /= oversampling_factor

        # Now work on the individual nightly stack
        filepathlist_nightly_stack = []

        # Get all the nightly stack files
        for folder_i in folder_name:

            filelist_all = os.listdir(folder_i)

            for filename in filelist_all:

                if filename.endswith("nightly_stack.fits"):

                    _filter = filename.split("_")[0]

                    if _filter.upper() == filter_i:

                        filepathlist_nightly_stack.append(
                            os.path.join(folder_i, filename)
                        )

                    else:

                        print("Unaccounted filters: {}".format(_name))
                        print("It is not handled.")

        # for each frame, get the sigma (note fit_sigma=Ture, meaning it's returning sigma instead of FWHM)
        sigma_x, sigma_y = get_all_fwhm(
            filepathlist_nightly_stack,
            stars_tbl,
            fit_sigma=True,
            sigma=3.0,
            sigma_lower=3.0,
            sigma_upper=3.0,
            threshold=5000.0,
            box_size=25,
            maxiters=10,
            norm_radius=5.5,
            npeaks=20,
            shift_val=0.01,
            recentering_boxsize=(5, 5),
            recentering_maxiters=10,
            center_accuracy=0.001,
            smoothing_kernel="quadratic",
            output_folder=output_folder,
        )

        # Use the FWHM to generate script for hotpants
        # see also https://github.com/acbecker/hotpants
        sigma_ref = np.sqrt(sigma_x_stack**2.0 + sigma_y_stack**2.0)
        sigma_list = np.sqrt(sigma_x**2.0 + sigma_y**2.0)
        diff_image_script = generate_hotpants_script(
            aligned_file_list[np.argmin(sigma_list)],
            aligned_file_list,
            min(sigma_list),
            sigma_list,
            hotpants="hotpants",
            write_to_file=True,
            filename=os.path.join(output_folder, "diff_image.sh"),
            overwrite=True,
            tu=50000,
            tg=2.4,
            tr=12.0,
            iu=50000,
            ig=2.4,
            ir=12.0,
        )
        # run hotpants
        diff_image_list = run_hotpants(diff_image_script, output_folder=output_folder)

        # Use the FWHM to find the stars in the stacked image
        # see also https://photutils.readthedocs.io/en/stable/detection.html
        source_list = find_star(
            data_stacked_bkg_sub,
            fwhm=sigma_x_stack * 2.355,
            n_threshold=10.0,
            x=len(data_stacked[0].data) / 2.0,
            y=len(data_stacked[0].data) / 2.0,
            radius=300,
            show=True,
            output_folder=output_folder,
        )

        # Use the psf and stars to perdo forced photometry on the differenced images
        # see also https://photutils.readthedocs.io/en/latest/psf.html
        photometry_list = do_photometry(
            diff_image_list,
            source_list,
            sigma_list,
            output_folder=output_folder,
            save_individual=True,
        )

"""
# get lightcurves
source_id, mjd, flux, flux_err, flux_fit = get_lightcurve(
    photometry_list, source_list["id"], plot=True, output_folder=output_folder
)

# plot all lightcurves in the same figure
# plot_lightcurve(mjd, flux, flux_err)

# plot all lightcurves in the separate figure
# plot_lightcurve(mjd, flux, flux_err, same_figure=False)

# Explicitly plot 1 lightcurve
target = 13
target_arg = source_id == target
period = 33.625 / 60.0 / 24.0
period = 1
# scatter((mjd[target_arg] / period) % 1, flux[target_arg], s=1)
# ylim(-5000, 5000)


plot_lightcurve(
    (mjd[target_arg] / period) % 1,
    flux[target_arg],
    flux_err[target_arg],
    source_id=target,
    output_folder=output_folder,
)
"""

# Explicitly plot a few lightcurves
"""
good_stars = [1, 5, 8, 10, 3]
mjd_good_stars = np.array([mjd[i] for i in good_stars])
flux_good_stars = np.array([flux[i] for i in good_stars])
flux_err_good_stars = np.array([flux_err[i] for i in good_stars])
flux_ensemble = ensemble_photometry(flux_good_stars, flux_err_good_stars)
plot_lightcurve(mjd_good_stars,
                flux_good_stars,
                flux_err_good_stars,
                source_id=good_stars,
                output_folder=output_folder)
plot_lightcurve(mjd_good_stars,
                flux_ensemble,
                flux_err_good_stars,
                source_id=good_stars,
                output_folder=output_folder)
"""
