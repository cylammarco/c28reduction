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

    filelist_flat_raw_all = get_filelist(folder_i, frame_type="flat")
    filelist_shutter_flat_raw_all = get_filelist(folder_i, frame_type="shutter_flat")

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
