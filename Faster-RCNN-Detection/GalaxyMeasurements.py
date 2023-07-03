import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union

from astropy.io import fits
from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import photutils
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog


def merge_clumps(coords, fwhm, close_by_threshold):
    """
    If two clumps are extremely close, merges them into one 
    and averages the position of two clumps.
    
    Args:
      coords: list of [x, y]
      fwhm: radius
      close_by_threshold: multiplicator for radius
      
    Returns:
      Returns array with new [x, y]
    """
    
    pts = [geometry.Point((p[0], p[1])) for p in coords]
    unioned_buffered_poly = unary_union([p.buffer(fwhm * close_by_threshold) for p in pts])
    
    if unioned_buffered_poly.geom_type == 'MultiPolygon':
        return np.array([[u.centroid.x, u.centroid.y] for u in unioned_buffered_poly])
    elif unioned_buffered_poly.geom_type == 'Polygon':
        return np.array([[unioned_buffered_poly.centroid.x, unioned_buffered_poly.centroid.y]])


def merge_clumps_df(df, coordx, coordy, fwhm, close_by_threshold, first_cols, max_cols):
    """
    If two clumps are extremely close, merges them into one 
    and averages the position of two clumps.
    If one clump is "odd" and the other "normal", assigns the clumps the class "odd"
    
    Creates a geopandas dataframe from the input-df, creates buffers around the centroids,
    dissolves the geometries and aggregates by geometry.
    
    Args:
      df: dataframe as input
      coordx, coordy: name of the cols in df to be used as x,y coordinates
      fwhm: radius
      close_by_threshold: multiplicator for radius
      cols: columns of the dataframe df to return
      
    Returns:
      Returns df with new [x, y], aggregated columns
    """
    
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[coordx], df[coordy]).buffer(0.5 * df[fwhm].iloc[0] * close_by_threshold)
    )

    agg_col_geo = {'geometry': 'first'}
    agg_first_cols = dict.fromkeys(first_cols, 'first')
    agg_max_cols = dict.fromkeys(max_cols, 'max')

    agg_col = agg_col_geo | agg_first_cols | agg_max_cols

    intersects = (
        gdf[['geometry']]
        .dissolve()
        .explode(index_parts=True)
        .sjoin(gdf, how='inner', predicate='intersects')
        .reset_index()
        .groupby('level_1')
        .agg(agg_col)
        .set_geometry('geometry')
        .assign(px_x = lambda df_: df_.centroid.x)
        .assign(px_y = lambda df_: df_.centroid.y)
        .reset_index()
    )
    
    return intersects # [['px_x', 'px_y', label, fwhm]]


def galaxy_mask(data, box_size=50, filter_size=3, kernel_size=3, detection_threshold=1.5, debug=False):
    """
    Takes a FITS-image and masks the galaxy.
    The mask can then be applied to the clump centroids.
    
    Args:
      data: image-data as array from FITS
      box_size: size in px of the box for background substraction
      filter_size: size in px for the background filter
      kernel_size: size in px of the kernel for convolve, should be FWHM
      detection_threshold: multiplier for a detection threshold image using the background RMS image (sigma per px)
      
    Returns:
      Returns mask-array
    """

    bkg_estimator = MedianBackground()

    bkg = Background2D(
        data, 
        (box_size, box_size), 
        filter_size=(filter_size, filter_size), 
        bkg_estimator=bkg_estimator
    )
    data -= bkg.background

    threshold = detection_threshold * bkg.background_rms

    kernel = make_2dgaussian_kernel(kernel_size, size=5)
    convolved_data = convolve(data, kernel)

    segment_map = detect_sources(convolved_data, threshold, npixels=10)

    # keep only most central segmentation mask, presumably the main galaxy
    border = int(segment_map.shape[0]/2 - 1)
    segment_map.remove_border_labels(border_width=border, partial_overlap=False, relabel=True)

    # plot segmentation map
    if debug:
        print(segment_map)
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(data, cmap='Greys_r', norm=norm)
        plt.imshow(segment_map.outline_segments(mask_background=False), alpha=0.3)
    
    # create mask (array)
    footprint = circular_footprint(radius=np.floor(kernel_size))
    mask = segment_map.make_source_mask(footprint=footprint)

    # create catalogue
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

    return mask, segment_map, cat


def clump_mask(px_x, px_y, radius, img_size, debug=False):
    """
    Takes centroid coordinates and FWHM-radius of clumps in px
    and creates a mask for background substraction.
    
    Args:
      px_x, px_y: clump centroids coords in px
      radius: radius of the circular mask to be applied, e.g. FWHM
      img_size: shape of the FITS-image
      
    Returns:
      Returns mask-array
    """

    mask = np.zeros(img_size).astype(bool)
    y0, x0 = np.indices(img_size)

    for x, y in zip(px_x, px_y):
        distance = np.sqrt((x-x0)**2 + (y-y0)**2)
        mask[distance <= radius] = True

    # plot segmentation map
    if debug:
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(mask, cmap='Greys_r', norm=norm)
    
    return mask


def gauss2d(x, y, sig, x0=0, y0=0):
    return 0.5/sig**2/np.pi * np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sig**2)


def fwhm_to_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2*np.log(2.)))


def Moffat2D_xy(gamma, alpha, normed=True):
    if normed:
        norm = 1 / Moffat2D_integral(r=1e10,gamma=gamma,alpha=alpha)
    else:
        norm = 1.0

    return lambda x, y: norm * (1 + ((x**2+y**2)/gamma**2))**(-alpha)


def Moffat2D_integral(r, gamma, alpha):
    return np.pi * (gamma**2 - (gamma**2)**alpha * (gamma**2 + r**2)**(1-alpha)) / (alpha - 1)


def reff_from_gamma(gamma, alpha):
    norm = 1 / Moffat2D_integral(r=1e5, gamma=gamma, alpha=alpha)
    return np.sqrt(gamma**(-2*alpha/(1-alpha)) * (gamma**2 + (1-alpha)/(2*np.pi*norm))**(1/(1-alpha)) - gamma**2)


def fwhm_to_gamma(fwhm, alpha):
    reff_gauss = fwhm / 2
    return scipy.optimize.brentq(lambda g: reff_from_gamma(g, alpha=alpha) - reff_gauss, 0.1, 10)
    # return fwhm / 2 / np.sqrt(2**(1/alpha) - 1)


def get_aperture_correction(psf, aper, func='gauss'):
    dx = dy = 0.01
    x = np.arange(-10, 10, dx)
    y = np.arange(-10, 10, dy)

    yy, xx = np.meshgrid(y, x, sparse=True)

    if func=='gauss':
        g_see = gauss2d(xx, yy, sig=fwhm_to_sigma(psf))

    elif func=='moffat':
        alpha = 2.5
        MoffatFunc = Moffat2D_xy(gamma=fwhm_to_gamma(psf,alpha), alpha=alpha)
        g_see = MoffatFunc(xx,yy)

    else:
        raise Exception("Invalid function type.")

    cond_aper = np.sqrt(xx**2 + yy**2) <= aper
    f_see = np.sum(g_see[cond_aper]) * dx * dy

    aperture_correction = 1/f_see

    return aperture_correction


