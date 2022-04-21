import argparse
import copy
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner
from reproject import reproject_exact


def wcs_fit(filelist):
    for filename in filelist:
        # WCS fit
        filepath = os.path.join("reduced_image", filename)
        subprocess.call(
            "solve-field {} --ra 19:21:43.6 --dec -15:57:18 --radius 1 --cpulimit 30000".format(
                filepath
            ),
            shell=True,
        )


parser = argparse.ArgumentParser(description="Some minimal control of data reduction.")
parser.add_argument(
    "--calibration-frame-only",
    action="store_true",
    help="Set to accumulate calibration frames only without reducing light frames.",
)
parser.add_argument("--folder")
args = parser.parse_args()

ups_sgr_coord = SkyCoord(290.43176441262, -15.95504344758, unit="deg")

# get the name of the folder that holds the frames
folder_name = args.folder

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
    "Ha": flat_master_Ha,
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

# Get all the files
filelist_all = os.listdir(folder_name)

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
    if filename.startswith("UpsilonSgr") & filename.endswith("fts"):
        filelist_light_raw.append(filename)
    elif filename.startswith("Dark"):
        filelist_dark_raw.append(filename)
    elif filename.startswith("Bias"):
        filelist_bias_raw.append(filename)
    elif filename.startswith("FF"):
        _name = os.path.splitext(filename)[0].split("-")[-1]
        if _name == "B":
            filelist_flat_B_raw.append(filename)
        elif _name == "V":
            filelist_flat_V_raw.append(filename)
        elif _name == "R":
            filelist_flat_R_raw.append(filename)
        elif _name == "Ha":
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
            filepath = os.path.join(folder_name, filename)
            fits_data = fits.open(filepath)[0]
            bias_ccddata_list.append(
                CCDData(fits_data.data, header=fits_data.header, unit=units.count)
            )
            n_bias_nightly += 1
    if bias_ccddata_list != []:
        bias_master_nightly_data = Combiner(
            bias_ccddata_list, dtype=np.float64
        ).average_combine()
        bias_master_nightly_fits = fits.PrimaryHDU(
            bias_master_nightly_data, fits.Header()
        )
        for i, filename in enumerate(filelist_bias_raw):
            bias_master_nightly_fits.header["FRAME_" + str(i)] = filename
        bias_master_nightly_fits.writeto(
            os.path.join(
                folder_name, "bias_master_nightly_{}.fits".format(folder_name)
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
                        bias_master_nightly_data,
                        header=fits_data.header,
                        unit=units.count,
                    ),
                ],
                dtype=np.float64,
            )
            new_bias_master_data.weights = np.array([n_bias_master, n_bias_nightly])
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
    print("No new bias is added from {}.".format(folder_name))


if os.path.exists(bias_master):
    # Load the bias master if exists
    bias_master_fits = fits.open(bias_master)
else:
    raise ValueError("Bias master frame: {} does not exist.".format(bias_master))

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
            filepath = os.path.join(folder_name, filename)
            fits_data = fits.open(filepath)[0]
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
        dark_master_nightly_data = Combiner(
            dark_ccddata_list, dtype=np.float64
        ).average_combine()
        dark_master_nightly_fits = fits.PrimaryHDU(
            dark_master_nightly_data, fits.Header()
        )
        for i, filename in enumerate(filelist_dark_raw):
            dark_master_nightly_fits.header["FRAME_" + str(i)] = filename
        dark_master_nightly_fits.writeto(
            os.path.join(
                folder_name, "dark_master_nightly_{}.fits".format(folder_name)
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
                        dark_master_nightly_data,
                        header=fits_data.header,
                        unit=units.count,
                    ),
                ],
                dtype=np.float64,
            )
            new_dark_master_data.weights = np.array([n_dark_master, n_dark_nightly])
            new_dark_master_data_combined = new_dark_master_data.average_combine()
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
    print("No new dark is added from {}.".format(folder_name))


if os.path.exists(dark_master):
    # Load the dark master if exists
    dark_master_fits = fits.open(dark_master)
else:
    raise ValueError("Dark master frame: {} does not exist.".format(dark_master))


# Create or add to the master_flat
# All flats are converted into count per second
# add the nightly flats to the master if nightly flats are available
for filelist_flat_raw in filelist_flat_raw_all:
    if filelist_flat_raw != []:
        _filter = os.path.splitext(filelist_flat_raw[0])[0].split("-")[-1]
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
            filename_temp = folder_name + "-" + filename
            if filename_temp not in flat_master_frame_name:
                filepath = os.path.join(folder_name, filename)
                fits_data = fits.open(filepath)[0]
                flat_ccddata_list.append(
                    CCDData(
                        fits_data.data
                        - dark_master_fits[0].data * float(fits_data.header["EXPTIME"])
                        - bias_master_fits[0].data,
                        header=fits_data.header,
                        unit=units.count,
                    )
                )
                n_flat_nightly += 1
        if flat_ccddata_list != []:
            flat_master_nightly_data = Combiner(
                flat_ccddata_list, dtype=np.float64
            ).average_combine()
            flat_master_nightly_data /= np.nanmean(flat_master_nightly_data)
            flat_master_nightly_fits = fits.PrimaryHDU(
                flat_master_nightly_data,
                fits.Header(),
            )
            for i, filename in enumerate(filelist_flat_raw):
                flat_nightly_frame_name.append(filename)
                flat_master_nightly_fits.header["FRAME_" + str(i)] = (
                    folder_name + "-" + filename
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
                            flat_master_nightly_data,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                    ],
                    dtype=np.float64,
                )
                new_flat_master_data.weights = np.array([n_flat_master, n_flat_nightly])
                new_flat_master_data_combined = new_flat_master_data.average_combine()
                flat_master_fits.close()
                flat_master_fits = None
                del flat_master_fits
                new_flat_master_fits = fits.PrimaryHDU(
                    new_flat_master_data_combined, flat_master_fits_header
                )
                for i, filename in enumerate(flat_nightly_frame_name):
                    filename_temp = folder_name + "-" + filename
                    new_flat_master_fits.header[
                        "FRAME_" + str(i + n_flat_master)
                    ] = filename_temp
                new_flat_master_fits.writeto(
                    "flat_master_{}.fits".format(_filter), overwrite=True
                )
                new_flat_master_fits = None
                del new_flat_master_fits
            else:
                flat_master_nightly_fits.writeto(
                    "flat_master_{}.fits".format(_filter), overwrite=True
                )
                flat_master_nightly_fits = None
                del flat_master_nightly_fits

# Shutter flats here
for filelist_flat_raw in filelist_shutter_flat_raw_all:
    if filelist_flat_raw != []:
        _filter = os.path.splitext(filelist_flat_raw[0])[0].split("-")[-1]
        print("Building shutter flats, working with filter: {}.".format(_filter))
        n_flat_nightly = 0
        n_flat_master = 0
        flat_ccddata_list = []
        flat_nightly_frame_name = []
        flat_master_frame_name = []
        # if master flat exists
        if os.path.exists(shutter_flat_master[_filter]):
            # Load the flat master if exists
            flat_master_fits = fits.open(shutter_flat_master[_filter], memmap=False)
            flat_master_fits_header = copy.deepcopy(flat_master_fits[0].header)
            # Go through the constitute flat frames name
            for i in flat_master_fits_header:
                if i.startswith("FRAME"):
                    flat_master_frame_name.append(flat_master_fits_header[i])
                    n_flat_master += 1
        # Go through the new flat frames name
        for filename in filelist_flat_raw:
            filename_temp = folder_name + "-" + filename
            if filename_temp not in flat_master_frame_name:
                filepath = os.path.join(folder_name, filename)
                fits_data = fits.open(filepath)[0]
                flat_ccddata_list.append(
                    CCDData(
                        fits_data.data
                        - dark_master_fits[0].data * float(fits_data.header["EXPTIME"])
                        - bias_master_fits[0].data,
                        header=fits_data.header,
                        unit=units.count,
                    )
                )
                n_flat_nightly += 1
        if flat_ccddata_list != []:
            flat_master_nightly_data = Combiner(
                flat_ccddata_list, dtype=np.float64
            ).average_combine()
            flat_master_nightly_data /= np.nanmean(flat_master_nightly_data)
            flat_master_nightly_fits = fits.PrimaryHDU(
                flat_master_nightly_data,
                fits.Header(),
            )
            for i, filename in enumerate(filelist_flat_raw):
                flat_nightly_frame_name.append(filename)
                flat_master_nightly_fits.header["FRAME_" + str(i)] = (
                    folder_name + "-" + filename
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
                            flat_master_nightly_data,
                            header=fits_data.header,
                            unit=units.count,
                        ),
                    ],
                    dtype=np.float64,
                )
                new_flat_master_data.weights = np.array([n_flat_master, n_flat_nightly])
                new_flat_master_data_combined = new_flat_master_data.average_combine()
                flat_master_fits.close()
                flat_master_fits = None
                del flat_master_fits
                new_flat_master_fits = fits.PrimaryHDU(
                    new_flat_master_data_combined, flat_master_fits_header
                )
                for i, filename in enumerate(flat_nightly_frame_name):
                    filename_temp = folder_name + "-" + filename
                    new_flat_master_fits.header[
                        "FRAME_" + str(i + n_flat_master)
                    ] = filename_temp
                new_flat_master_fits.writeto(
                    "shutter_flat_master_{}.fits".format(_filter), overwrite=True
                )
                new_flat_master_fits = None
                del new_flat_master_fits
            else:
                flat_master_nightly_fits.writeto(
                    "shutter_flat_master_{}.fits".format(_filter), overwrite=True
                )
                flat_master_nightly_fits = None
                del flat_master_nightly_fits


if not args.calibration_frame_only:
    reduced_light_filename_list = []
    for filename in filelist_light_raw:
        # Get the filter
        _filter = filename.split("-")[2]
        outfile_name, outfile_extension = os.path.splitext(filename)
        outfile_name += "-reduced"
        # Get the file path
        filepath = os.path.join(folder_name, filename)
        outfile_filepath = os.path.join(folder_name, outfile_name + outfile_extension)
        # Load the light frame and exposure time
        light_fits = fits.open(filepath)
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
        if _filter == "B":
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
        if _filter == "Ha":
            flat_fits = fits.open("flat_master_{}.fits".format(_filter))
        else:
            flat_fits = fits.open("shutter_flat_master_{}.fits".format(_filter))
        light_fits_data = light_fits_data / np.float64(flat_fits[0].data)
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

    # Do WCS fit on all of the light frames
    wcs_fit(reduced_light_filename_list)

    # Get the filelist of all the WCS fitted light frames
    filelist_wcs_fitted = [
        os.path.splitext(i)[0] + ".new" for i in reduced_light_filename_list
    ]

    # Reproject to the total stack
    fits_data = fits.open(filepath)[0]
    wcs = WCS(fits_data.header)
    try:
        cutout_img = Cutout2D(
            data=fits_data.data,
            position=ups_sgr_coord,
            size=2.0 * units.degree,
            wcs=wcs,
            mode="strict",
            copy=True,
        )
        fits_data.header.update(wcs.to_header())
        hdu = fits.PrimaryHDU(cutout_img.data, header=fits_data.header)
        hdu.writeto(
            os.path.join(
                "cutout_image",
                filename.split(".gz")[0].split(".fits")[0].split(".fts")[0]
                + "_reprojected.fits",
            ),
            overwrite=True,
        )
    except Exception as e:
        print(e)
