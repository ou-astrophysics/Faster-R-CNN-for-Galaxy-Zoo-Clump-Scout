# Clump catalogue
Faster-R-CNN clump detections using Zoobot as the feature extraction backbone on the full set of Galaxy Zoo 2 galaxies observed by SDSS.

The csv-file `FRCNN_Zoobot_SDSS_GZ2_detections.csv.zip` has the following columns and contains all clumps with an estimated clump/galaxy u-band flux ratio of >3%.

| Column name | Description |
| ----------- | ----------- |
 `specobjid`  |  SDSS spec object ID |
 `dr7objid`  |  SDSS DR7 object ID |
 `clump_id`  |  Clump index |
 `clump_label_id`  |  Clump label ID (1 or 2) |
 `clump_label_name`  |  Clump label name |
 `clump_score`  |  Detection score for the clump |
 `clump_centre_ra`  |  Clump centroid RA in degrees | 
 `clump_centre_dec`  |  Clump centroid dec in degrees | 
 `clump_centre_px_x`  |  Clump centroid X in PNG-coutout pixels |
 `clump_centre_px_y`  |  Clump centroid Y in PNG-coutout pixels |
 `clump_px_x1`  |  Clump bounding box X1 in PNG-coutout pixels |
 `clump_px_x2`  |  Clump bounding box X2 in PNG-coutout pixels |
 `clump_px_y1`  |  Clump bounding box Y1 in PNG-coutout pixels |
 `clump_px_y2`  |  Clump bounding box Y2 in PNG-coutout pixels |
 `FWHM_r_px`  |  r-band PSF-FWHM in PNG-coutout pixels |
 `clump_flux_u`  |  Clump u-band flux in Jy |
 `clump_flux_g`  |  Clump g-band flux in Jy |
 `clump_flux_r`  |  Clump r-band flux in Jy |
 `clump_flux_i`  |  Clump i-band flux in Jy |
 `clump_flux_z`  |  Clump z-band flux in Jy |
 `clump_flux_err_u`  |  Clump u-band flux error in Jy |
 `clump_flux_err_g`  |  Clump g-band flux error in Jy |
 `clump_flux_err_r`  |  Clump r-band flux error in Jy |
 `clump_flux_err_i`  |  Clump i-band flux error in Jy |
 `clump_flux_err_z`  |  Clump z-band flux error in Jy |
 `clump_mag_u`  |  Clump u-band magnitude (AB-mag) |
 `clump_mag_g`  |  Clump g-band magnitude (AB-mag) |
 `clump_mag_r`  |  Clump r-band magnitude (AB-mag) |
 `clump_mag_i`  |  Clump i-band magnitude (AB-mag) |
 `clump_mag_z`  |  Clump z-band magnitude (AB-mag) |
 `clump_ext_mag_u`  |  Clump u-band extinction (E(B-V), AB-mag) |
 `clump_ext_mag_g`  |  Clump g-band extinction (E(B-V), AB-mag) |
 `clump_ext_mag_r`  |  Clump r-band extinction (E(B-V), AB-mag) |
 `clump_ext_mag_i`  |  Clump i-band extinction (E(B-V), AB-mag) |
 `clump_ext_mag_z`  |  Clump z-band extinction (E(B-V), AB-mag) |
 `clump_mag_corr_u`  |  Clump corrected u-band magnitude (AB-mag) |
 `clump_mag_corr_g`  |  Clump corrected g-band magnitude (AB-mag) |
 `clump_mag_corr_r`  |  Clump corrected r-band magnitude (AB-mag) |
 `clump_mag_corr_i`  |  Clump corrected i-band magnitude (AB-mag) |
 `clump_mag_corr_z`  |  Clump corrected z-band magnitude (AB-mag) |
 `clump_mag_corr_u_g`  |  Clump colour (u-g) |
 `clump_mag_corr_g_r`  |  Clump colour (g-r) |
 `clump_mag_corr_r_i`  |  Clump colour (r-i) |
 `clump_mag_corr_i_z`  |  Clump colour (i-z) |
 `clump_flux_ratio`  |  Est. clump/galaxy near-UV flux ratio (u-band) |
 `is_clump_3pct`  |  Flag (True/False) if clump/galaxy flux ratio is >3% |
 `is_clump_8pct`  |  Flag (True/False) if clump/galaxy flux ratio is >8% |
 `galaxy_ra`  |  Host galaxy RA in degrees | 
 `galaxy_dec`  |  Host galaxy dec in degrees | 
 `galaxy_z`  |  Host galaxy redshift | 
 `galaxy_mag_u`  |  Host galaxy u-band magnitude (AB-mag) |
 `galaxy_mag_g`  |  Host galaxy g-band magnitude (AB-mag) |
 `galaxy_mag_r`  |  Host galaxy r-band magnitude (AB-mag) |
 `galaxy_mag_i`  |  Host galaxy i-band magnitude (AB-mag) |
 `galaxy_mag_z`  |  Host galaxy z-band magnitude (AB-mag) |
 `galaxy_mag_err_u`  |  Host galaxy u-band magnitude error (AB-mag) |
 `galaxy_mag_err_g`  |  Host galaxy g-band magnitude error (AB-mag) |
 `galaxy_mag_err_r`  |  Host galaxy r-band magnitude error (AB-mag) |
 `galaxy_mag_err_i`  |  Host galaxy i-band magnitude error (AB-mag) |
 `galaxy_mag_err_z`  |  Host galaxy z-band magnitude error (AB-mag) |
 `galaxy_flux_u`  |  Host galaxy u-band flux in Jy |
 `galaxy_flux_g`  |  Host galaxy g-band flux in Jy |
 `galaxy_flux_r`  |  Host galaxy r-band flux in Jy |
 `galaxy_flux_i`  |  Host galaxy i-band flux in Jy |
 `galaxy_flux_z`  |  Host galaxy z-band flux in Jy |
 `galaxy_expAB_r`  |  Host galaxy axis ratio from SDSS |
 `galaxy_expRad_r`  |  Host galaxy exponential fit scale radius from SDSS |
 `galaxy_lmass`  |  Host galaxy log mass in MSun |
 `galaxy_lssfr`  |  Host galaxy log specific SFR |
 `galaxy_mag_corr_u`  |  Host galaxy corrected u-band magnitude (AB-mag) |
 `galaxy_mag_corr_g`  |  Host galaxy corrected g-band magnitude (AB-mag) |
 `galaxy_mag_corr_r`  |  Host galaxy corrected r-band magnitude (AB-mag) |
 `galaxy_mag_corr_i`  |  Host galaxy corrected i-band magnitude (AB-mag) |
 `galaxy_mag_corr_z`  |  Host galaxy corrected z-band magnitude (AB-mag) |
