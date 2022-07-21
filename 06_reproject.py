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
from reproject import reproject_adaptive

from get_filelist import get_filelist


parser = argparse.ArgumentParser()
parser.add_argument("--folder")
args = parser.parse_args()

ups_sgr_coord = SkyCoord(290.43176441262, -15.95504344758, unit="deg")

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


for folder_i in folder_name:

    print(folder_i)

    filelist_light_reduced_wcs_fitted = get_filelist(folder_i, frame_type="wcs_fitted")

    (
        filelist_light_reduced_wcs_fitted_B,
        filelist_light_reduced_wcs_fitted_V,
        filelist_light_reduced_wcs_fitted_R,
        filelist_light_reduced_wcs_fitted_Ha,
    ) = filelist_light_reduced_wcs_fitted

    wcs_reference_fits = fits.open("skv12573619882825_1.fits", memmap=False)[0]
    wcs_reference = WCS(wcs_reference_fits.header)

    # B band
    for filename in filelist_light_reduced_wcs_fitted_B:

        print(filename)

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=(2160, 2160),
            return_footprint=False,
        )
        # Save the reprojected image
        B_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected, fits_file[0].header)
        B_reprojected_fits.header.update(wcs_reference.to_header())
        B_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # V band
    for filename in filelist_light_reduced_wcs_fitted_V:

        print(filename)

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=(2160, 2160),
            return_footprint=False,
        )
        # Save the reprojected image
        V_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected, fits_file[0].header)
        V_reprojected_fits.header.update(wcs_reference.to_header())
        V_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # R band
    for filename in filelist_light_reduced_wcs_fitted_R:

        print(filename)

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=(2160, 2160),
            return_footprint=False,
        )
        # Save the reprojected image
        R_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected, fits_file[0].header)
        R_reprojected_fits.header.update(wcs_reference.to_header())
        R_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # Ha band
    for filename in filelist_light_reduced_wcs_fitted_Ha:

        print(filename)

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=(2160, 2160),
            return_footprint=False,
        )
        # Save the reprojected image
        Ha_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected, fits_file[0].header)
        Ha_reprojected_fits.header.update(wcs_reference.to_header())
        Ha_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )
