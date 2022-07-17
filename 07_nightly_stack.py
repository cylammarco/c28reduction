import argparse
import glob
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner

from get_filelist import get_filelist


ups_sgr_coord = SkyCoord(290.43176441262, -15.95504344758, unit="deg")

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


for folder_i in folder_name:

    B_combiner_list = []
    V_combiner_list = []
    R_combiner_list = []
    Ha_combiner_list = []

    B_exp_time_list = []
    V_exp_time_list = []
    R_exp_time_list = []
    Ha_exp_time_list = []

    filelist_light_reduced_wcs_fitted_reprojected = get_filelist(
        folder_i, frame_type="reprojected"
    )

    (
        filelist_light_reduced_wcs_fitted_reprojected_B,
        filelist_light_reduced_wcs_fitted_reprojected_V,
        filelist_light_reduced_wcs_fitted_reprojected_R,
        filelist_light_reduced_wcs_fitted_reprojected_Ha,
    ) = filelist_light_reduced_wcs_fitted_reprojected

    wcs_reference = WCS(
        fits.open(os.path.join(folder_i, filelist_light_reduced_all[0]), memmap=False)[
            0
        ].header
    )

    # B band
    for filename in filelist_light_reduced_wcs_fitted_reprojected_B:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        B_combiner_list.append(
            CCDData(
                fits_file[0].data,
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
    for filename in filelist_light_reduced_wcs_fitted_reprojected_V:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        V_combiner_list.append(
            CCDData(
                fits_file[0].data,
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
    for filename in filelist_light_reduced_wcs_fitted_reprojected_R:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        R_combiner_list.append(
            CCDData(
                fits_file[0].data,
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
    for filename in filelist_light_reduced_wcs_fitted_reprojected_Ha:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        Ha_combiner_list.append(
            CCDData(
                fits_file[0].data,
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
