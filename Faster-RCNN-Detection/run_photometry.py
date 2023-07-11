#!/usr/bin/env python

# [START all]

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../shared/')
import argparse
import GalaxyMeasurements
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, ScalarFormatter

from astropy.io import fits
from astropy import coordinates as coords
from astropy import wcs
from astropy.convolution import convolve, Box2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import photutils
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceCatalog, SegmentationImage

from tqdm import tqdm


DATA_PATH = './data/'
FITS_FILE_PATH = '/path/to/FITS/files/'
PREDICTIONS_FILE_PATH = './predictions/'
PHOT_FILE_PATH = './photometry/'

FWHM_MULTIPLIER = 1.125
FWHM_MULTIPLIER_ANNULUS_MIN = 1.5
FWHM_MULTIPLIER_ANNULUS_MAX = 2.5


def main():
    # [START Parse arguments from cmd]
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', dest='prediction_filename', help='Filename of the predictions output.', type=str)
    parser.add_argument('--score', dest='score_threshold', nargs='?', default=0.3, help='Define score threshold.', type=int)

    args = parser.parse_args()

    prediction_filename = os.path.join(PREDICTIONS_FILE_PATH, args.prediction_filename)
    SCORE_THRESHOLD = args.score_threshold
    # [END Parse arguments from cmd]

    # [START Read predictions]
    df = (pd
        .read_parquet(prediction_filename, engine='pyarrow')
        .query('specobjid == specobjid & is_central == False & isin_galaxy == True')
        .assign(specobjid = lambda df_: df_.specobjid.astype(int))
    )
    # [END Read predictions]

    # [START Merging adjacent clumps and masking with galaxy]
    results = []
    
    for specobjid in tqdm(df['specobjid'].unique()):
        
        fits_files = {band: fits.open(FITS_FILE_PATH + str(specobjid)[-2:] + '/' + str(specobjid) + '_Projected_{}.fits'.format(band)) for band in 'ugriz'}
        
        _df = df[(df['specobjid']==specobjid)]
    
        # use r-band FITS image for galaxy masking
        data = fits_files['r']['PRIMARY'].data.copy() #.byteswap(inplace=True).newbyteorder()
        kernel_size = _df['px_FITS_fwhm_r'].iloc[0]
        box_size = int(data.shape[0] / (2 * kernel_size))
    
        mask, seg_map, cat = GalaxyMeasurements.galaxy_mask(
            data=data, 
            box_size=box_size, 
            filter_size=3, 
            kernel_size=kernel_size, 
            detection_threshold=.5
        )
        
        if len(_df) > 0:
            
            # px-coords for the circular aperture
            positions = [tuple(r) for r in _df[['px_FITS_centre_x', 'px_FITS_centre_y']].to_numpy().tolist()]
             
            phot = {}
            
            for i, (band, f_file) in enumerate(fits_files.items()):
                
                fits_image = f_file['PRIMARY'].data
                
                fwhm = _df['px_FITS_fwhm_'+band].iloc[0] # or use only the r-band FWHM??
                error = _df['err_fit_m_'+band].iloc[0] * fits_image + _df['err_fit_b_'+band].iloc[0]
            
                # masking clumps
                masked_clumps = GalaxyMeasurements.clump_mask(
                    px_x=_df['px_FITS_centre_x'],
                    px_y=_df['px_FITS_centre_y'],
                    radius=fwhm*FWHM_MULTIPLIER,
                    img_size=fits_image.shape,
                    # debug=True
                )
                # setting masked pixel to nan for background substraction
                masked_fits_image = fits_image.copy()
                masked_fits_image[masked_clumps] = np.nan
                masked_fits_image[~mask] = np.nan
            
                aperture = photutils.aperture.CircularAperture(positions, r=fwhm*FWHM_MULTIPLIER)  
                annulus_aperture = photutils.aperture.CircularAnnulus(positions, r_in=fwhm*FWHM_MULTIPLIER_ANNULUS_MIN, r_out=fwhm*FWHM_MULTIPLIER_ANNULUS_MAX)
                
                phot_table = photutils.aperture.aperture_photometry(fits_image, aperture, error=error)
                aperstats = photutils.aperture.ApertureStats(fits_image, annulus_aperture)
                aperture_area = aperture.area_overlap(fits_image)
            
                # and with clumps masked:
                aperstats_mask = photutils.aperture.ApertureStats(masked_fits_image, annulus_aperture)
            
                phot_table['total_bkg'] = aperstats.median * aperture_area
                phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['total_bkg']
                phot_table['total_bkg_masked'] = aperstats_mask.median * aperture_area
                phot_table['aperture_sum_bkgsub_masked'] = phot_table['aperture_sum'] - phot_table['total_bkg_masked']
                
                phot['aperture_sum_'+band] = np.array(phot_table['aperture_sum'])
                phot['aperture_sum_err_'+band] = np.array(phot_table['aperture_sum_err'])
                phot['total_bkg_'+band] = np.array(phot_table['total_bkg'])
                phot['total_bkg_masked_'+band] = np.array(phot_table['total_bkg_masked'])
                phot['aperture_sum_bkgsub_'+band] = np.array(phot_table['aperture_sum_bkgsub'])
                phot['aperture_sum_bkgsub_masked_'+band] = np.array(phot_table['aperture_sum_bkgsub_masked'])
            
                # add aperture correction
                phot['aperture_correction_gauss_'+band] = GalaxyMeasurements.get_aperture_correction(psf=fwhm, aper=fwhm*FWHM_MULTIPLIER, func='gauss')
                phot['aperture_correction_moffat_'+band] = GalaxyMeasurements.get_aperture_correction(psf=fwhm, aper=fwhm*FWHM_MULTIPLIER, func='moffat')
        
            results.append([
                specobjid, _df['clump_id'].values,
                _df['petroFlux_u'].values,
                _df['labels'].values, _df['scores'].values, _df['px_FITS_fwhm_r'].iloc[0],
                _df['px_FITS_centre_x'].values, _df['px_FITS_centre_y'].values, _df['clump_centre_ra'].values,  _df['clump_centre_dec'].values,
                phot['aperture_sum_u'], phot['aperture_sum_err_u'], phot['total_bkg_u'], phot['total_bkg_masked_u'], phot['aperture_sum_bkgsub_u'], phot['aperture_sum_bkgsub_masked_u'], phot['aperture_correction_gauss_u'], phot['aperture_correction_moffat_u'],
                phot['aperture_sum_g'], phot['aperture_sum_err_g'], phot['total_bkg_g'], phot['total_bkg_masked_g'], phot['aperture_sum_bkgsub_g'], phot['aperture_sum_bkgsub_masked_g'], phot['aperture_correction_gauss_g'], phot['aperture_correction_moffat_g'],
                phot['aperture_sum_r'], phot['aperture_sum_err_r'], phot['total_bkg_r'], phot['total_bkg_masked_r'], phot['aperture_sum_bkgsub_r'], phot['aperture_sum_bkgsub_masked_r'], phot['aperture_correction_gauss_r'], phot['aperture_correction_moffat_r'],
                phot['aperture_sum_i'], phot['aperture_sum_err_i'], phot['total_bkg_i'], phot['total_bkg_masked_i'], phot['aperture_sum_bkgsub_i'], phot['aperture_sum_bkgsub_masked_i'], phot['aperture_correction_gauss_i'], phot['aperture_correction_moffat_i'],
                phot['aperture_sum_z'], phot['aperture_sum_err_z'], phot['total_bkg_z'], phot['total_bkg_masked_z'], phot['aperture_sum_bkgsub_z'], phot['aperture_sum_bkgsub_masked_z'], phot['aperture_correction_gauss_z'], phot['aperture_correction_moffat_z'],
            ])
        
        # close FITS
        for i, (band, f_file) in enumerate(fits_files.items()):
            f_file.close()
        
    df_results = (
        pd.DataFrame(
            results, 
            columns=[
                'specobjid', 'clump_id', 'petroFlux_u',
                'labels', 'scores', 'px_FITS_fwhm_r', 
                'px_FITS_centre_x', 'px_FITS_centre_y', 'deg_FITS_centre_ra', 'deg_FITS_centre_dec',
                'aperture_sum_u', 'aperture_sum_err_u', 'total_bkg_u', 'total_bkg_masked_u', 'aperture_sum_bkgsub_u', 'aperture_sum_bkgsub_masked_u', 'aperture_correction_gauss_u', 'aperture_correction_moffat_u',
                'aperture_sum_g', 'aperture_sum_err_g', 'total_bkg_g', 'total_bkg_masked_g', 'aperture_sum_bkgsub_g', 'aperture_sum_bkgsub_masked_g', 'aperture_correction_gauss_g', 'aperture_correction_moffat_g',
                'aperture_sum_r', 'aperture_sum_err_r', 'total_bkg_r', 'total_bkg_masked_r', 'aperture_sum_bkgsub_r', 'aperture_sum_bkgsub_masked_r', 'aperture_correction_gauss_r', 'aperture_correction_moffat_r',
                'aperture_sum_i', 'aperture_sum_err_i', 'total_bkg_i', 'total_bkg_masked_i', 'aperture_sum_bkgsub_i', 'aperture_sum_bkgsub_masked_i', 'aperture_correction_gauss_i', 'aperture_correction_moffat_i',
                'aperture_sum_z', 'aperture_sum_err_z', 'total_bkg_z', 'total_bkg_masked_z', 'aperture_sum_bkgsub_z', 'aperture_sum_bkgsub_masked_z', 'aperture_correction_gauss_z', 'aperture_correction_moffat_z',
            ]
        )
        .explode([
            'clump_id', 'petroFlux_u', 'labels', 'scores', 'px_FITS_centre_x', 'px_FITS_centre_y', 'deg_FITS_centre_ra', 'deg_FITS_centre_dec',
            'aperture_sum_u', 'aperture_sum_err_u', 'total_bkg_u', 'total_bkg_masked_u', 'aperture_sum_bkgsub_u', 'aperture_sum_bkgsub_masked_u',
            'aperture_sum_g', 'aperture_sum_err_g', 'total_bkg_g', 'total_bkg_masked_g', 'aperture_sum_bkgsub_g', 'aperture_sum_bkgsub_masked_g',
            'aperture_sum_r', 'aperture_sum_err_r', 'total_bkg_r', 'total_bkg_masked_r', 'aperture_sum_bkgsub_r', 'aperture_sum_bkgsub_masked_r',
            'aperture_sum_i', 'aperture_sum_err_i', 'total_bkg_i', 'total_bkg_masked_i', 'aperture_sum_bkgsub_i', 'aperture_sum_bkgsub_masked_i',
            'aperture_sum_z', 'aperture_sum_err_z', 'total_bkg_z', 'total_bkg_masked_z', 'aperture_sum_bkgsub_z', 'aperture_sum_bkgsub_masked_z',
        ])
    )
    # [END Merging adjacent clumps and masking with galaxy]

    # [START Write output]
    result_file_name = os.path.splitext(args.prediction_filename)[0] + '_phot_score{}.gzip'.format(SCORE_THRESHOLD)
    df_results.to_parquet(result_file_name , compression='gzip')
    print('Results written to {}'.format(result_file_name))
    # [END Write output]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)

# [END all]
