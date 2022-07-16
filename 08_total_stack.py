import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner


parser = argparse.ArgumentParser()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


for folder_i in folder_name:

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
                    fits_file[0].data,
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
                    fits_file[0].data,
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
                    fits_file[0].data,
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
                    fits_file[0].data,
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
