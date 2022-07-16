import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units
from astropy.wcs import WCS
from ccdproc import Combiner
from reproject import reproject_adaptive

from .get_filelist import get_filelist


parser = argparse.ArgumentParser()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


for folder_i in folder_name:

    filelist_light_reduced_wcs_fitted = get_filelist(folder_i, frame_type="wcs_fitted")

    (
        filelist_light_reduced_wcs_fitted_B,
        filelist_light_reduced_wcs_fitted_V,
        filelist_light_reduced_wcs_fitted_R,
        filelist_light_reduced_wcs_fitted_Ha,
    ) = filelist_light_reduced_wcs_fitted

    wcs_reference = WCS(
        fits.open(
            os.path.join(folder_i, filelist_light_reduced_wcs_fitted[0]), memmap=False
        )[0].header
    )

    # B band
    for filename in filelist_light_reduced_wcs_fitted_B:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=np.shape(fits_file[0].data),
            return_footprint=False,
        )
        # Save the reprojected image
        B_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected)
        B_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # V band
    for filename in filelist_light_reduced_wcs_fitted_V:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=np.shape(fits_file[0].data),
            return_footprint=False,
        )
        # Save the reprojected image
        V_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected)
        V_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # R band
    for filename in filelist_light_reduced_wcs_fitted_R:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=np.shape(fits_file[0].data),
            return_footprint=False,
        )
        # Save the reprojected image
        R_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected)
        R_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )

    # Ha band
    for filename in filelist_light_reduced_wcs_fitted_Ha:

        fits_file = fits.open(os.path.join(folder_i, filename), memmap=False)
        wcs = WCS(fits_file[0].header)
        fits_data_reprojected = reproject_adaptive(
            input_data=fits_file,
            output_projection=wcs_reference,
            shape_out=np.shape(fits_file[0].data),
            return_footprint=False,
        )
        # Save the reprojected image
        Ha_reprojected_fits = fits.PrimaryHDU(fits_data_reprojected)
        Ha_reprojected_fits.writeto(
            os.path.join(folder_i, os.path.splitext(filename)[0] + "_reprojected.fits"),
            overwrite=True,
        )
