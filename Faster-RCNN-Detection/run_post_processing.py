#!/usr/bin/env python

# [START all]

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../shared/')
import argparse
import GalaxyMeasurements
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union

from astropy.io import fits
from astropy import wcs
from astropy import coordinates as coords
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


def main():
    # [START Parse arguments from cmd]
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', dest='prediction_filename', help='Filename of the predictions output.', type=str)
    parser.add_argument('--score', dest='score_threshold', nargs='?', default=0.3, help='Define score threshold.', type=int)

    args = parser.parse_args()

    prediction_filename = os.path.join(PREDICTIONS_FILE_PATH, args.prediction_filename)
    score_threshold = args.score_threshold
    # [END Parse arguments from cmd]
    
    # [START Read galaxy data and FITS metadata]
    cols = [
        'specobjid', 
        'arcsec_per_side', 'pix_per_arcsec', 'pix_per_side',
        'petroFlux_u',
        'psffwhm_u', 'psffwhm_g', 'psffwhm_r', 'psffwhm_i', 'psffwhm_z',
        'err_fit_m_u', 'err_fit_m_g', 'err_fit_m_r', 'err_fit_m_i', 'err_fit_m_z', 
        'err_fit_b_u', 'err_fit_b_g', 'err_fit_b_r', 'err_fit_b_i', 'err_fit_b_z'
    ]
    df_meta = (pd
        .read_csv(DATA_PATH + 'cutout_data_enhanced.csv')
        [cols]
        .rename(columns={
            'arcsec_per_side': 'asec_FITS_side', 
            'pix_per_arcsec': 'px_FITS_asec',
            'pix_per_side': 'px_FITS_side'
        })
        .assign(petroFlux_u = lambda df_: df_.petroFlux_u * 3.631e-06)
        .assign(specobjid = lambda df_: df_.specobjid.astype(int))
        .drop_duplicates()
    )
    # [END Read galaxy data and FITS metadata]

    # [START Read predictions]
    df = (pd
        .read_parquet(prediction_filename, engine='pyarrow')
        .query('scores >= @score_threshold')
        .dropna()
        .rename(columns={
            'local_ids': 'specobjid',
            'px_x1': 'px_GZ2_x1',
            'px_x2': 'px_GZ2_x2',
            'px_y1': 'px_GZ2_y1',
            'px_y2': 'px_GZ2_y2',
        })
        .assign(px_GZ2_centre_x = lambda df_: (df_['px_GZ2_x1'] + df_['px_GZ2_x2']) / 2.0 )
        .assign(px_GZ2_centre_y = lambda df_: (df_['px_GZ2_y1'] + df_['px_GZ2_y2']) / 2.0 )
        .assign(px_GZ2_x1_normed = lambda df_: df_['px_GZ2_x1'] / 400)
        .assign(px_GZ2_x2_normed = lambda df_: df_['px_GZ2_x2'] / 400)
        .assign(px_GZ2_y1_normed = lambda df_: df_['px_GZ2_y1'] / 400)
        .assign(px_GZ2_y2_normed = lambda df_: df_['px_GZ2_y2'] / 400)
        .assign(is_central = lambda df_: 
                (
                    np.abs(0.5*(df_['px_GZ2_x2_normed'] + df_['px_GZ2_x1_normed']) - 0.5) < 0.02
                    ) & (
                    np.abs(0.5*(df_['px_GZ2_y2_normed'] + df_['px_GZ2_y1_normed']) - 0.5) < 0.02
                )
            )
        .merge(df_meta, how='left', on='specobjid')
        .assign(px_FITS_side = lambda df_: np.rint(df_.asec_FITS_side * df_.px_FITS_asec))
        .assign(px_FITS_centre_x = lambda df_: df_.px_GZ2_centre_x / (400. / df_.px_FITS_side))
        .assign(px_FITS_centre_y = lambda df_: df_.px_GZ2_centre_y / (400. / df_.px_FITS_side))
        .assign(px_FITS_x1 = lambda df_: df_.px_GZ2_x1_normed * df_.px_FITS_side)
        .assign(px_FITS_x2 = lambda df_: df_.px_GZ2_x2_normed * df_.px_FITS_side)
        .assign(px_FITS_y1 = lambda df_: df_.px_GZ2_y1_normed * df_.px_FITS_side)
        .assign(px_FITS_y2 = lambda df_: df_.px_GZ2_y2_normed * df_.px_FITS_side)
        .assign(px_GZ2_fwhm = lambda df_: (df_.psffwhm_r * df_.px_FITS_asec * (400. / df_.px_FITS_side)))
        .assign(px_FITS_fwhm_u = lambda df_: (df_.psffwhm_u * df_.px_FITS_asec))
        .assign(px_FITS_fwhm_g = lambda df_: (df_.psffwhm_g * df_.px_FITS_asec))
        .assign(px_FITS_fwhm_r = lambda df_: (df_.psffwhm_r * df_.px_FITS_asec))
        .assign(px_FITS_fwhm_i = lambda df_: (df_.psffwhm_i * df_.px_FITS_asec))
        .assign(px_FITS_fwhm_z = lambda df_: (df_.psffwhm_z * df_.px_FITS_asec))
        .assign(specobjid = lambda df_: df_.specobjid.astype(int))
    )

    # assign clump_id
    df['clump_id'] = df.groupby(['run', 'model_name', 'specobjid'])['px_FITS_centre_x'].rank(method='first').astype(int)
    # [END Read predictions]

    # [START Merging adjacent clumps and masking with galaxy]
    # define columns for merging
    first_cols = [
        'run', 'model_name', 'specobjid', 'clump_id',
        'asec_FITS_side', 'px_FITS_asec', 'px_FITS_side',
        'petroFlux_u', 
        'psffwhm_u', 'psffwhm_g', 'psffwhm_r', 'psffwhm_i', 'psffwhm_z',
        'err_fit_m_u', 'err_fit_m_g', 'err_fit_m_r', 'err_fit_m_i', 'err_fit_m_z', 
        'err_fit_b_u', 'err_fit_b_g', 'err_fit_b_r', 'err_fit_b_i', 'err_fit_b_z',
        'px_GZ2_fwhm', 
        'px_FITS_fwhm_u', 'px_FITS_fwhm_g', 'px_FITS_fwhm_r', 'px_FITS_fwhm_i', 'px_FITS_fwhm_z',
        'px_FITS_x1', 'px_FITS_y1', 'px_FITS_x2', 'px_FITS_y2',
        'px_FITS_centre_x', 'px_FITS_centre_y',
        'px_GZ2_x1', 'px_GZ2_x2', 'px_GZ2_y1', 'px_GZ2_y2',
        'px_GZ2_centre_x', 'px_GZ2_centre_y',
        'px_GZ2_x1_normed', 'px_GZ2_y1_normed', 'px_GZ2_x2_normed', 'px_GZ2_y2_normed',
    ]
    
    max_cols = [
        'labels',
        'scores',
        'is_central',
        'isin_galaxy',
    ]

    model_list = df['model_name'].unique().tolist()

    results = []
    
    for specobjid in tqdm(df['specobjid'].unique()):
        f_file = fits.open(FITS_FILE_PATH + str(specobjid)[-2:] + '/' + str(specobjid) + '_Projected_r.fits')
        
        for model_name in model_list:
            _df = df[(df['specobjid']==specobjid) & (df['model_name']==model_name)]
    
            if len(_df) > 0:
                # use r-band FITS image for galaxy masking
                data = f_file['PRIMARY'].data.copy() #.byteswap(inplace=True).newbyteorder()
                kernel_size = _df['px_FITS_fwhm_r'].iloc[0]
                box_size = int(data.shape[0] / (2 * kernel_size))
            
                mask, seg_map, cat = GalaxyMeasurements.galaxy_mask(
                    data=data, 
                    box_size=box_size, 
                    filter_size=3, 
                    kernel_size=kernel_size, 
                    detection_threshold=.5,
                )
            
                # apply mask, switch x-y to y-x because of the shape of the array
                _df = _df.assign(isin_galaxy = lambda __df: mask[__df.px_FITS_centre_y.astype(int), __df.px_FITS_centre_x.astype(int)])
                
                if len(_df[_df['isin_galaxy']]) > 0:
                    # merge adjacent clumps into single clump and aggregating the df
                    _df = GalaxyMeasurements.merge_clumps_df(
                        _df[_df['isin_galaxy']], 
                        'px_FITS_centre_x', 
                        'px_FITS_centre_y', 
                        'px_FITS_fwhm_u', 
                        1., 
                        first_cols=first_cols, 
                        max_cols=max_cols
                    )
                    
                    # Re-project coords
                    w = wcs.WCS(f_file['PRIMARY'].header)
                    df1 = pd.DataFrame(
                        w.wcs_pix2world(np.array(_df[['px_x', 'px_y']].to_numpy(), np.float_), 1),
                        dtype=float, 
                        columns=['clump_centre_ra', 'clump_centre_dec']
                    )
            
                    __df = pd.concat([_df.reset_index(), df1], axis=1)
            
                    results.append(__df)
        
        f_file.close()
    
    # concatinating the results
    df_results = pd.concat(results)
    # [END Merging adjacent clumps and masking with galaxy]

    # [START Write output]
    result_file_name = os.path.splitext(args.prediction_filename)[0] + '_post.gzip'
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
