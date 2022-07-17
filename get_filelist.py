import argparse
import glob
import os


def get_filelist(folder_name, frame_type):

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

    filelist_light_reduced_B = []
    filelist_light_reduced_V = []
    filelist_light_reduced_R = []
    filelist_light_reduced_Ha = []

    filelist_light_reduced_wcs_fitted_B = []
    filelist_light_reduced_wcs_fitted_V = []
    filelist_light_reduced_wcs_fitted_R = []
    filelist_light_reduced_wcs_fitted_Ha = []

    filelist_light_reduced_wcs_fitted_reprojected_B = []
    filelist_light_reduced_wcs_fitted_reprojected_V = []
    filelist_light_reduced_wcs_fitted_reprojected_R = []
    filelist_light_reduced_wcs_fitted_reprojected_Ha = []

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

            _filter = os.path.splitext(filename)[0].split("-")[-1]
            if _filter.upper() == "B":
                filelist_flat_B_raw.append(filename)
            elif _filter.upper() == "V":
                filelist_flat_V_raw.append(filename)
            elif _filter.upper() == "R":
                filelist_flat_R_raw.append(filename)
            elif _filter.upper() == "HA":
                filelist_flat_Ha_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_filter))
                print("It is not handled.")

        elif filename.startswith("AutoFlat"):

            _filter = os.path.splitext(filename)[0].split("-")[-2]
            if _filter.upper() == "B":
                filelist_flat_B_raw.append(filename)
            elif _filter.upper() == "V":
                filelist_flat_V_raw.append(filename)
            elif _filter.upper() == "R":
                filelist_flat_R_raw.append(filename)
            elif _filter.upper() == "HA":
                filelist_flat_Ha_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_filter))
                print("It is not handled.")

        elif filename.startswith("SFF"):

            _filter = os.path.splitext(filename)[0].split("-")[-1]
            if _filter == "B02":
                filelist_shutter_flat_B02_raw.append(filename)
            elif _filter == "B05":
                filelist_shutter_flat_B05_raw.append(filename)
            elif _filter == "V":
                filelist_shutter_flat_V_raw.append(filename)
            elif _filter == "R":
                filelist_shutter_flat_R_raw.append(filename)
            else:
                print("Unaccounted filters: {}".format(_filter))
                print("It is not handled.")

        elif filename.endswith("-reduced.fts"):

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
                print("Unaccounted filters: {}".format(_filter))
                print("It is not handled.")

        elif filename.endswith("new"):

            _filter = os.path.splitext(filename)[0].split("-")[-2]
            if _filter.upper() == "B":
                filelist_light_reduced_wcs_fitted_B.append(filename)
            elif _filter.upper() == "V":
                filelist_light_reduced_wcs_fitted_V.append(filename)
            elif _filter.upper() == "R":
                filelist_light_reduced_wcs_fitted_R.append(filename)
            elif _filter.upper() == "HA":
                filelist_light_reduced_wcs_fitted_Ha.append(filename)
            else:
                print("Unaccounted filters: {}".format(_filter))
                print("It is not handled.")

        elif filename.endswith("_reprojected.fits"):

            _filter = os.path.splitext(filename)[0].split("-")[-2]
            if _filter.upper() == "B":
                filelist_light_reduced_wcs_fitted_reprojected_B.append(filename)
            elif _filter.upper() == "V":
                filelist_light_reduced_wcs_fitted_reprojected_V.append(filename)
            elif _filter.upper() == "R":
                filelist_light_reduced_wcs_fitted_reprojected_R.append(filename)
            elif _filter.upper() == "HA":
                filelist_light_reduced_wcs_fitted_reprojected_Ha.append(filename)
            else:
                print("Unaccounted filters: {}".format(_filter))
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

    filelist_light_reduced_all = [
        filelist_light_reduced_B,
        filelist_light_reduced_V,
        filelist_light_reduced_R,
        filelist_light_reduced_Ha,
    ]

    filelist_light_reduced_wcs_fitted_all = [
        filelist_light_reduced_wcs_fitted_B,
        filelist_light_reduced_wcs_fitted_V,
        filelist_light_reduced_wcs_fitted_R,
        filelist_light_reduced_wcs_fitted_Ha,
    ]

    filelist_light_reduced_wcs_fitted_reprojected_all = [
        filelist_light_reduced_wcs_fitted_reprojected_B,
        filelist_light_reduced_wcs_fitted_reprojected_V,
        filelist_light_reduced_wcs_fitted_reprojected_R,
        filelist_light_reduced_wcs_fitted_reprojected_Ha,
    ]

    if frame_type == "light":

        return filelist_light_raw

    if frame_type == "dark":

        return filelist_dark_raw

    if frame_type == "bias":

        return filelist_bias_raw

    if frame_type == "flat":

        return filelist_flat_raw_all

    if frame_type == "shutter_flat":

        return filelist_shutter_flat_raw_all

    if frame_type == "reduced":

        return filelist_light_reduced_all

    if frame_type == "wcs_fitted":

        return filelist_light_reduced_wcs_fitted_all

    if frame_type == "reprojected":

        return filelist_light_reduced_wcs_fitted_reprojected_all
