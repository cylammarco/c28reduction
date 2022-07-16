import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.append("/home/mlam/git/py-hotpants")
from pyhotpants import *


parser = argparse.ArgumentParser()

# get the name of the folder that holds the frames
folder_name = args.folder

if folder_name is None:

    folder_name = [i.split(os.sep)[1] for i in glob.glob(f"./*c28/")]

else:

    folder_name = [folder_name]


for folder_i in folder_name:

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
