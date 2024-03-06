# Import necessary libraries for data manipulation, neuroimaging data processing, and machine learning

import os
import numpy as np
import nibabel as nib
import pandas as pd
import joblib
from nilearn import datasets as ds
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean
from nilearn import datasets as ds
from sklearn.cross_decomposition import PLSCanonical 
import seaborn as sns
from scipy.stats import scoreatpercentile
from scipy.stats import pearsonr

DECONF = True  # Flag for deconfounding, not used in the snippet provided
TAR_ANA = 'AM'  # Target analysis set to Amygdala

# Load UKBiobank data
ukbb = pd.read_csv('/Users/dblab/Desktop/Project Amygdala/Amygdala/ukb40500_cut_merged.csv/ukb40500_cut_merged.csv', low_memory=True)

# Load descriptions dictionary for region volumes
descr_dict = joblib.load('/Users/dblab/Desktop/Project Amygdala/Amygdala/descr_dict')

# Process and filter columns for specific regions based on the FSL atlas, excluding Diederichsen cerebellar atlas
ukbb_HO20 = ukbb.loc[:, '25782-2.0':'25892-2.0']  # Timepoint 2 data
ukbb_HO20 = ukbb_HO20.iloc[:, ~ukbb_HO20.columns.str.contains('-3.0')]
ukbb_HO30 = ukbb.loc[:, '25782-3.0':'25892-3.0']  # Timepoint 3 data
ukbb_HO30 = ukbb_HO30.iloc[:, ~ukbb_HO30.columns.str.contains('-2.0')]

# Rename columns using descriptions for easier understanding
HO_vol_names = np.array([descr_dict[c]['descr'].split('Volume of grey matter in ')[1] for c in ukbb_HO20.columns])
ukbb_HO20.columns = HO_vol_names
ukbb_HO30.columns = HO_vol_names

# Load demographic and clinical information
eid = ukbb['eid']
age_T2 = ukbb.loc[:, '21003-2.0']  # Age at Timepoint 2
age_T3 = ukbb.loc[:, '21003-3.0']  # Age at Timepoint 3
sex = ukbb.loc[:, '31-0.0']  # Sex

# Function to impute missing values non-parametrically
def NonparametricImpute(input_vars):
    nan_inds = np.where(np.isnan(input_vars))[0]
    pres_inds = np.where(~np.isnan(input_vars))[0]
    rs = np.random.RandomState(0)
    rs.shuffle(pres_inds)
    input_vars[nan_inds] = input_vars[pres_inds[:len(nan_inds)]]
    return input_vars

# Load amygdala subregion measurements for first timepoint 
COLS_NAMES = []
COLS_IDS = []
for fname in ['/Users/dblab/Desktop/Project Amygdala/Amygdala/subcortical_labels_%s.txt' % TAR_ANA]:
    with open(fname) as f:
        lines=f.readlines()
        for line in lines:
            a, b = line.split('\t')[0], line.strip().split('\t')[1]
            COLS_IDS.append(a + '-2.0')
            COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
dfS20 = ukbb.loc[:, COLS_IDS]
dfS20.columns = np.array([str(c.encode("ascii")) for c in COLS_NAMES])
dfS20 = ukbb.loc[:, COLS_IDS]
dfS20.columns = np.array([str(c.encode("ascii")) for c in COLS_NAMES])


# Load amygdala subregion measurements for second timepoint 
TAR_ANA = 'AM'
COLS_NAMES = []
COLS_IDS = []
for fname in ['/Users/dblab/Desktop/Project Amygdala/Amygdala/subcortical_labels_%s.txt' % TAR_ANA]:
    with open(fname) as f:
        lines=f.readlines()
        f.close()
        for line in lines:
            a = line[:line.find('\t')]
            b = line[line.find('\t') + 1:].rsplit('\n')[0]
            COLS_IDS.append(a + '-3.0')
            COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
sub_dict = {COLS_IDS[i_col] : COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}


dfS30 = ukbb.loc[:, COLS_IDS]
dfS30.columns = np.array([str(c.encode("ascii")) for c in COLS_NAMES])


# remove columns with excessive missingness in 2nd time point
subs_keep = dfS30.isna().sum(1) == 0
dfS20 = dfS20.loc[subs_keep]
dfS30 = dfS30.loc[subs_keep]
ukbb_HO20 = ukbb_HO20.loc[subs_keep]
ukbb_HO30 = ukbb_HO30.loc[subs_keep]
eid = eid.loc[subs_keep]
age_T2 = age_T2[subs_keep]
age_T3 = age_T3[subs_keep]
sex = sex[subs_keep]
ukbb_2tp = ukbb[subs_keep]

# Standardize (z-score) the datasets for both time points
S_scaler = StandardScaler()
FS_AM20 = dfS20.values
FS_AM20_ss = S_scaler.fit_transform(FS_AM20)
FS_AM30 = dfS30.values
FS_AM30_ss = S_scaler.transform(FS_AM30)

HO_scaler = StandardScaler()
FS_HO20 = ukbb_HO20.values
FS_HO20_ss = HO_scaler.fit_transform(FS_HO20)
FS_HO30 = ukbb_HO30.values
FS_HO30_ss = HO_scaler.transform(FS_HO30)

# remove the amygdala from HO atlas space
idx_nonAM = ~ukbb_HO20.columns.str.contains('Amygdala')
FS_HO20 = FS_HO20[:, idx_nonAM]
FS_HO20_ss = FS_HO20_ss[:, idx_nonAM]
FS_HO30 = FS_HO30[:, idx_nonAM]
FS_HO30_ss = FS_HO30_ss[:, idx_nonAM]
ukbb_HO20 = ukbb_HO20.loc[:, idx_nonAM]
ukbb_HO30 = ukbb_HO30.loc[:, idx_nonAM]

# Impute missing values using a Nonparametric method (not provided here, ensure implementation is available)
FS_AM20_ss = NonparametricImpute(FS_AM20_ss)
FS_AM30_ss = NonparametricImpute(FS_AM30_ss)
FS_HO20_ss = NonparametricImpute(FS_HO20_ss)
FS_HO30_ss = NonparametricImpute(FS_HO30_ss)

# Ensure there are no missing values after imputation
assert np.all(~np.isnan(FS_AM20_ss))
assert np.all(~np.isnan(FS_AM30_ss))
assert np.all(~np.isnan(FS_HO20_ss))
assert np.all(~np.isnan(FS_HO30_ss))

# Remove columns related to whole-HC measures
keep_col_inds = ~dfS30.columns.str.contains('Whole')  # remove 6 whle-HC measures
dfS20 = dfS20.loc[:, keep_col_inds]
dfS30 = dfS30.loc[:, keep_col_inds]
FS_AM20 = FS_AM20[:, keep_col_inds]
FS_AM20_ss = FS_AM20_ss[:, keep_col_inds]
FS_AM30 = FS_AM30[:, keep_col_inds]
FS_AM30_ss = FS_AM30_ss[:, keep_col_inds]
COLS_NAMES = COLS_NAMES[keep_col_inds]

# Deconfound brain structural measures if required
if DECONF:
    # Additional behavior data for deconfounding
    beh = ukbb_2tp

    age = StandardScaler().fit_transform(beh['21022-0.0'].values[:, np.newaxis])  # Age at recruitment
    age2 = age ** 2
    sex = np.array(pd.get_dummies(beh['31-0.0']).values, dtype=np.int)  # Sex
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    head_motion_rest = np.nan_to_num(beh['25741-2.0'].values)  # Mean rfMRI head motion
    head_motion_task = np.nan_to_num(beh['25742-2.0'].values)  # Mean tfMRI head motion

    # added during previous paper revisions
    head_size = np.nan_to_num(beh['25006-2.0'].values)  # Volume of gray matter
    body_mass = np.nan_to_num(beh['21001-0.0'].values)  # BMI

    # motivated by Elliott et al., 2018
    # exact location of the head and the radio-frequency receiver coil in the scanner
    head_pos_x = np.nan_to_num(beh['25756-2.0'].values)  
    head_pos_y = np.nan_to_num(beh['25757-2.0'].values)
    head_pos_z = np.nan_to_num(beh['25758-2.0'].values)
    head_pos_table = np.nan_to_num(beh['25759-2.0'].values) 
    scan_site_dummies = pd.get_dummies(beh['54-2.0']).values

    #Ensure no Nans
    assert np.any(np.isnan(head_motion_rest)) == False
    assert np.any(np.isnan(head_motion_task)) == False
    assert np.any(np.isnan(head_size)) == False
    assert np.any(np.isnan(body_mass)) == False

    print('Deconfounding brain structural measures space!')

    #construct confounding matrix
    conf_mat = np.hstack([
        # age, age2, sex, sex_x_age, sex_x_age2,
        np.atleast_2d(head_motion_rest).T, np.atleast_2d(head_motion_task).T,
        np.atleast_2d(head_size).T, np.atleast_2d(body_mass).T,

        np.atleast_2d(head_pos_x).T, np.atleast_2d(head_pos_y).T,
        np.atleast_2d(head_pos_z).T, np.atleast_2d(head_pos_table).T,
        np.atleast_2d(scan_site_dummies)
        ])

    # Deconfounding using nilearn's clean function
    FS_AM20_ss = clean(FS_AM20_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_AM30_ss = clean(FS_AM30_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_HO20_ss = clean(FS_HO20_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_HO30_ss = clean(FS_HO30_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    
#Get the change in gray matter volume in the amygdala
dfAM20minus30 = pd.DataFrame(
	FS_AM20_ss - FS_AM30_ss, columns=dfS20.columns)

#Get the change in gray matter volume in the (sub)cortex
dfHO20minus30 = pd.DataFrame(
	FS_HO20_ss - FS_HO30_ss, columns=ukbb_HO20.columns)

#Difference in volume between second and third timepoints of the left subregions of the amygdala
dfAM20minus30_L = pd.DataFrame(dfAM20minus30.filter(regex='left').values, columns=dfAM20minus30.loc[:,['left' in i for i in dfAM20minus30.columns]].columns)

#Difference in volume between second and third timepoints of the right subregions of the amygdala
dfAM20minus30_R = pd.DataFrame(dfAM20minus30.filter(regex='right').values, columns=dfAM20minus30.loc[:,['right' in i for i in dfAM20minus30.columns]].columns)

#Difference between second and third timepoints
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical
pls = PLSCanonical(n_components=n_comps)
pls.fit(dfAM20minus30, dfHO20minus30)
r2 = pls.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`

est = pls
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])

print(actual_Rs)
# [0.19654273, 0.24062467, 0.21549389, 0.19554875, 0.21232852, 0.25243018, 0.21673239, 0.22932024]

#Analysis of the differences of the left subregions of the amygdala and the cortex between second and third time points
#PLS Canonical
pls_left= PLSCanonical(n_components=n_comps)
pls_left.fit(dfAM20minus30_L, dfHO20minus30)
r2 = pls_left.score(dfAM20minus30_L, dfHO20minus30)  # coefficient of determination :math:`R^2`

est = pls_left
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
print(actual_Rs)
# [0.17969291 0.22206758 0.24594784 0.19278412 0.20976133]


#Analysis of the differences of the right subregions of the amygdala and the cortex between second and third time points
#PLS Canonical
pls_right = PLSCanonical(n_components=n_comps)
pls_right.fit(dfAM20minus30_R, dfHO20minus30)
r2 = pls_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`

est = pls_right
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])

print(actual_Rs)
# [0.15075179 0.21492126 0.21132531 0.22200016 0.20072646]

#Fetch the Harvard Oxford Atlases
HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)

OUT_DIR = 'AMsub2cortex/dfAM20minus30/pls'
#Difference between second and third timepoints

# Loop over each component in the PLS (Partial Least Squares) analysis or similar
for i_mode in range(n_comps):
    # Initialize an array to store the Spatial Effect Size (SES) data in brain space,
    # matching the shape of the cortical atlas maps
    SES_in_brain_data = np.zeros((HO_atlas_cort.maps.shape)) 
    dfX = dfHO20minus30  # DataFrame containing the brain data differences between timepoints

    # Loop over each feature (brain region) in the DataFrame
    for i_feat in range(dfX.shape[-1]):
        cur_feat_name = dfX.columns[i_feat].split(' (')[0]  # Extract the brain region name
        
        # Special case handling for brain stem or renaming conventions
        if 'Stem' in cur_feat_name:
            pass  # Skip the brain stem or handle it separately
        else:
            # Prefix region names with hemisphere based on the column name
            cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

        # Adjust naming conventions for compatibility with atlas labels
        if 'Ventral Striatum' in cur_feat_name:
            cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

        b_found_roi = False  # Flag to check if the region of interest (ROI) was found in the atlas

        # Loop over cortical labels in the atlas to find a match with the current feature
        for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
            if cur_feat_name in cort_label:
                # If matched, find all voxels corresponding to this cortical label
                b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
                n_roi_vox = np.sum(b_roi_mask)  # Count the voxels in the ROI
                print(f'Found: {cort_label} ({n_roi_vox} voxels)')
                # Assign the PLS loading for this feature to all voxels in the ROI
                SES_in_brain_data[b_roi_mask] = pls.y_loadings_[i_feat, i_mode]
                b_found_roi = True

        # Repeat the process for subcortical labels if the ROI wasn't found in cortical labels
        for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
            if cur_feat_name in cort_label:
                b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
                n_roi_vox = np.sum(b_roi_mask)
                print(f'Found: {cort_label} ({n_roi_vox} voxels)')
                SES_in_brain_data[b_roi_mask] = pls.y_loadings_[i_feat, i_mode]
                b_found_roi = True

        # If the ROI was not found in either cortical or subcortical labels, print a message
        if not b_found_roi:
            print(f'NOT Found: {cur_feat_name} !!!')

    # After processing all features for the current component, save the SES data as a NIfTI image
    SES_name = f'mode{i_mode + 1}'
    SES_in_brain_nii = nib.Nifti1Image(SES_in_brain_data, HO_atlas_cort.maps.affine)
    # Define the output directory and file name, and save the NIfTI image
    SES_in_brain_nii.to_filename(OUT_DIR + '/' + SES_name + '_coef.nii.gz')

#Left and Right subregions of the amygdala seperately and cortex regions difference between second and third time points
OUT_DIR_4 = 'AMsub2cortex/dfAM20minus30_L/pls' 
OUT_DIR_6 = 'AMsub2cortex/dfAM20minus30_R/pls'

# Loop over each component in the PLS analysis or similar
for i_mode in range(n_comps):
    # Initialize two arrays to store the Spatial Effect Size (SES) data in brain space for left and right hemispheres
    SES_in_brain_data_L = np.zeros((HO_atlas_cort.maps.shape)) 
    SES_in_brain_data_R = np.zeros((HO_atlas_cort.maps.shape))
    dfX = dfHO20minus30  # DataFrame with the differences in brain data between two time points

    # Loop over each feature (brain region) in the DataFrame
    for i_feat in range(dfX.shape[-1]):
        cur_feat_name = dfX.columns[i_feat].split(' (')[0]  # Extract the brain region name
        
        # Skip processing for the brain stem or apply specific handling
        if 'Stem' in cur_feat_name:
            pass
        else:
            # Adjust the feature name to include hemisphere information based on the column name
            cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

        # Adjust naming conventions for compatibility with atlas labels (specific case handling)
        if 'Ventral Striatum' in cur_feat_name:
            cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

        b_found_roi = False  # Flag to check if the region of interest (ROI) was found in the atlas

        # Loop over cortical labels in the atlas to find a match with the current feature
        for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
            if cur_feat_name in cort_label:
                # If matched, find all voxels corresponding to this cortical label
                b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
                n_roi_vox = np.sum(b_roi_mask)  # Count the voxels in the ROI
                print(f'Found: {cort_label} ({n_roi_vox} voxels)')
                # Assign the PLS loading for this feature to all voxels in the ROI for left and right SES data
                SES_in_brain_data_L[b_roi_mask] = pls_left.y_loadings_[i_feat, i_mode]
                SES_in_brain_data_R[b_roi_mask] = pls_right.y_loadings_[i_feat, i_mode]
                b_found_roi = True

        # Repeat the process for subcortical labels if the ROI wasn't found in cortical labels
        for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
            if cur_feat_name in cort_label:
                b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
                n_roi_vox = np.sum(b_roi_mask)
                print(f'Found: {cort_label} ({n_roi_vox} voxels)')
                SES_in_brain_data_L[b_roi_mask] = pls_left.y_loadings_[i_feat, i_mode]
                SES_in_brain_data_R[b_roi_mask] = pls_right.y_loadings_[i_feat, i_mode]
                b_found_roi = True

        # If the ROI was not found in either cortical or subcortical labels, print a message
        if not b_found_roi:
            print(f'NOT Found: {cur_feat_name} !!!')

    # After processing all features for the current component, save the SES data as NIfTI images for both hemispheres
    SES_name = f'mode{i_mode + 1}'
    
    # Save left hemisphere SES data as a NIfTI file
    SES_in_brain_nii_L = nib.Nifti1Image(SES_in_brain_data_L, HO_atlas_cort.maps.affine)
    SES_in_brain_nii_L.to_filename(OUT_DIR_4 + '/' + SES_name + '_left_amygdala_coef.nii.gz')
    
    # Save right hemisphere SES data as a NIfTI file
    SES_in_brain_nii_R = nib.Nifti1Image(SES_in_brain_data_R, HO_atlas_cort.maps.affine)
    SES_in_brain_nii_R.to_filename(OUT_DIR_6 + '/' + SES_name + '_right_amygdala_coef.nii.gz')

#Difference between second and third timepoints PLS
# Define a suffix for file naming, useful for specifying conditions or parameters
SUFFIX = ''
# Iterate over each component in the PLS analysis
for counter, i_comp in enumerate(range(n_comps)):
    n_rois = pls.x_loadings_.shape[0]  # Number of ROIs/features in the PLS model
    X_AM_weights = pls.x_loadings_[:, i_comp]  # Extract weights (loadings) for the current component

    # Create a figure for the heatmap
    f = plt.figure(figsize=(9, 6), dpi=600)  # Define figure size and resolution

    # Initialize an array to store the component weights for visualization
    X_comp_weights = np.zeros((n_rois, 1))
    X_comp_weights[:, 0] = X_AM_weights  # Assign the weights to the array

    # Create a DataFrame for the weights, indexing by modified column names from dfAM20minus30
    dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30.columns.str.replace('b', '')).str.replace("'",''), columns=[''])

    # Save the weights to CSV and Excel files for external analysis and sharing
    dfdata.to_csv('%s/pls_AM_topcomp%i%s_style_.csv' % (OUT_DIR, counter + 1, SUFFIX))
    dfdata.to_excel('%s/pls_AM_topcomp%i%s_style_.xls' % (OUT_DIR, counter + 1, SUFFIX))

    # Generate a heatmap of the component weights using seaborn
    ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                     cbar_kws={'shrink': 0.5},  # Customization for the colorbar
                     square=True,
                     cmap=plt.cm.RdBu_r, center=0)  # Use diverging color map centered at 0

    # Set font sizes for the heatmap labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    # Workaround for a matplotlib bug that cuts off the top and bottom of the heatmap
    b, t = plt.ylim()  # Get the current bottom and top limits
    b += 0.5  # Adjust the bottom limit
    t -= 0.5  # Adjust the top limit
    plt.ylim(b, t)  # Apply the new limits

    plt.tight_layout()  # Adjust subplots to fit into the figure area

    # Save the heatmap to PNG and PDF files for high-quality visualizations
    plt.savefig('%s/pls_AM_topcomp%i%s_style_.png' % (OUT_DIR, counter + 1, SUFFIX), DPI=200)
    plt.savefig('%s/pls_AM_topcomp%i%s_style_.pdf' % (OUT_DIR, counter + 1, SUFFIX))


#Left Amygdala with cortex PLS
# Iterate over each component from the PLS analysis specific to the left amygdala
for counter, i_comp in enumerate(range(n_comps)):
    n_rois = pls_left.x_loadings_.shape[0]  # Number of regions of interest (ROIs) or features
    X_AM_weights = pls_left.x_loadings_[:, i_comp]  # Extract weights for the current component

    # Create a figure for the heatmap visualization
    f = plt.figure(figsize=(10, 7))

    # Initialize an array to store the component weights for visualization
    X_comp_weights = np.zeros((n_rois, 1))
    X_comp_weights[:, 0] = X_AM_weights

    # Prepare DataFrame of the weights, with modified column names for readability
    dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30_L.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
    
    # Save the component weights to CSV and Excel files
    dfdata.to_csv('%s/left_AM_topcomp%i%s_style_.csv' % (OUT_DIR_4, counter + 1, SUFFIX))
    dfdata.to_excel('%s/left_AM_topcomp%i%s_style_.xls' % (OUT_DIR_4, counter + 1, SUFFIX))

    # Generate and customize the heatmap using seaborn
    ax = sns.heatmap(dfdata, cbar=True, linewidths=.75, cbar_kws={'shrink': 0.5}, square=True, cmap=plt.cm.RdBu_r, center=0)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    # Adjust the heatmap display to ensure no cutoff at top/bottom
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.tight_layout()

    # Save the heatmap as high-resolution PNG and PDF
    plt.savefig('%s/left_AM_topcomp%i%s_style_.png' % (OUT_DIR_4, counter + 1, SUFFIX), DPI=600)
    plt.savefig('%s/left_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_4, counter + 1, SUFFIX))


#Right Amygdala with cortex PLS
# Similar loop for the right amygdala, generating heatmaps for each PLS componen
for counter, i_comp in enumerate(range(n_comps)):
  # The process repeats as above, with specifics adjusted for the right amygdala
  # This includes loading the PLS model for the right amygdala, preparing the data,
  # generating heatmaps, and saving the results to files designated for right amygdala analysis
  n_rois = pls_right.x_loadings_.shape[0]
  X_AM_weights = pls_right.x_loadings_[:, i_comp]

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30_R.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/right_AM_topcomp%i%s_style_.csv' % (OUT_DIR_6, counter + 1, SUFFIX))
  dfdata.to_excel('%s/right_AM_topcomp%i%s_style_.xls' % (OUT_DIR_6, counter + 1, SUFFIX))

  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  plt.tight_layout()

  plt.savefig('%s/right_AM_topcomp%i%s_style_.png' % (OUT_DIR_6, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/right_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_6, counter + 1, SUFFIX))

#########################################################################

#Hemispheric  Difference Analysis

# Number of bootstrap permutations and components to keep
n_BS_perm = 100
n_keep = 8 
# Initialize lists to store bootstrap results
BS_diff = []
list_l_x, list_l_y, list_r_x, list_r_y = [], [], [], []

# Bootstrap permutation loop
for i_BS in range(n_BS_perm):
    print(i_BS)

    # Generate a random sample of indices for bootstrap
    bs_rs = np.random.RandomState(i_BS)
    bs_sample_inds = bs_rs.randint(0, len(dfAM20minus30_L), len(dfAM20minus30_L))

    # Create bootstrap samples for left and right amygdala and cortex data
    bs_X_train_l = dfAM20minus30_L.iloc[bs_sample_inds, :]
    bs_X_train_r = dfAM20minus30_R.iloc[bs_sample_inds, :]
    bs_Y_train = dfHO20minus30.iloc[bs_sample_inds, :] 
    
    # Fit PLS models for left and right amygdala with bootstrap samples
    est_l = PLSCanonical(n_components=n_keep, scale=False)
    est_l.fit(bs_X_train_l, bs_Y_train)
    list_l_x.append(est_l.x_loadings_)
    list_l_y.append(est_l.y_loadings_)

    est_r = PLSCanonical(n_components=n_keep, scale=False)
    est_r.fit(bs_X_train_r, bs_Y_train)
    list_r_x.append(est_r.x_loadings_)
    list_r_y.append(est_r.y_loadings_)

# Save bootstrap results for further analysis
today_stamp = '201010'  # Current date stamp for file naming
np.save('BS_dump_list_full_l_x' + today_stamp, np.array(list_l_x))
np.save('BS_dump_list_full_l_y' + today_stamp, np.array(list_l_y))
np.save('BS_dump_list_full_r_x' + today_stamp, np.array(list_r_x))
np.save('BS_dump_list_full_r_y' + today_stamp, np.array(list_r_y))


# AM side
# Calculate the differences between left and right amygdala component loadings
it_diffs_x = np.zeros((n_BS_perm, n_keep, pls_left.x_loadings_.shape[0]))

# Loop through each permutation and component
for i_bs in range(n_BS_perm):
    for i_org_comp in range(n_keep):
        # Calculate correlations between original and bootstrap loadings for left and right
        l_rhos_x, r_rhos_x = np.zeros((n_keep)), np.zeros((n_keep))
        for i_bs_comp in range(n_keep):
            l_rhos_x[i_bs_comp], _ = pearsonr(pls_left.x_loadings_[:, i_org_comp], list_l_x[i_bs][:, i_bs_comp])
            r_rhos_x[i_bs_comp], _ = pearsonr(pls_right.x_loadings_[:, i_org_comp], list_r_x[i_bs][:, i_bs_comp])

        # Identify and adjust the most correlated component
        good_comp_l_x_ind = np.argmax(np.abs(l_rhos_x))
        good_comp_l_x = list_l_x[i_bs, :, good_comp_l_x_ind] * np.sign(l_rhos_x[good_comp_l_x_ind])
        good_comp_r_x_ind = np.argmax(np.abs(r_rhos_x))
        good_comp_r_x = list_r_x[i_bs, :, good_comp_r_x_ind] * np.sign(r_rhos_x[good_comp_r_x_ind])

        # Store the difference between adjusted components
        it_diffs_x[i_bs, i_org_comp] = good_comp_l_x - good_comp_r_x

# Determine significant differences based on a threshold
THRESH = 10  # Percentile threshold for significance
rel_mask_x = np.zeros((n_keep, pls_left.x_loadings_.shape[0]))
for i_comp in range(n_keep):
    lower_th = scoreatpercentile(it_diffs_x[:, i_comp, :], THRESH, axis=0)
    upper_th = scoreatpercentile(it_diffs_x[:, i_comp, :], 100 - THRESH, axis=0)
    rel_mask_x[i_comp] = ((lower_th < 0) & (upper_th < 0)) | ((lower_th > 0) & (upper_th > 0))
    # Print components with significant differences
    if np.sum(rel_mask_x[i_comp]) > 0:
        print('Amygdala component %i: %i hits' % (i_comp + 1, np.sum(rel_mask_x[i_comp])))
        print(list(dfAM20minus30_L.columns[np.array(rel_mask_x[i_comp], dtype=bool)].str.replace('(left hemisphere)', '')))
np.save('rel_mask_x_full_THRESH%i' % THRESH, rel_mask_x)



# HO side
# Initialize an array to store the differences in component loadings between hemispheres for cortex data
it_diffs_y = np.zeros((n_BS_perm, n_keep, pls_left.y_loadings_.shape[0]))

# Iterate through bootstrap permutations
for i_bs in range(n_BS_perm):
    for i_org_comp in range(n_keep):
        # Calculate correlation coefficients between the original and bootstrap samples for left and right hemispheres
        l_rhos_y = np.zeros(n_keep)
        r_rhos_y = np.zeros(n_keep)
        for i_bs_comp in range(n_keep):
            l_rhos_y[i_bs_comp], _ = pearsonr(pls_left.y_loadings_[:, i_org_comp], list_l_y[i_bs][:, i_bs_comp])
            r_rhos_y[i_bs_comp], _ = pearsonr(pls_right.y_loadings_[:, i_org_comp], list_r_y[i_bs][:, i_bs_comp])

        # Identify the component from the bootstrap samples that best matches the original component
        good_comp_l_y_ind = np.argmax(np.abs(l_rhos_y))
        good_comp_l_y = list_l_y[i_bs, :, good_comp_l_y_ind] * np.sign(l_rhos_y[good_comp_l_y_ind])
        good_comp_r_y_ind = np.argmax(np.abs(r_rhos_y))
        good_comp_r_y = list_r_y[i_bs, :, good_comp_r_y_ind] * np.sign(r_rhos_y[good_comp_r_y_ind])

        # Calculate the difference between the matched components for left and right hemispheres
        it_diffs_y[i_bs, i_org_comp] = good_comp_l_y - good_comp_r_y

# Assess significance of the differences using a threshold
rel_mask_y = np.zeros((n_keep, pls_left.y_loadings_.shape[0]))
THRESH = 10
for i_comp in range(n_keep):
    lower_th = scoreatpercentile(it_diffs_y[:, i_comp, :], THRESH, axis=0)
    upper_th = scoreatpercentile(it_diffs_y[:, i_comp, :], 100 - THRESH, axis=0)
    rel_mask_y[i_comp] = ((lower_th < 0) & (upper_th < 0)) | ((lower_th > 0) & (upper_th > 0))
    n_hits = np.sum(rel_mask_y[i_comp])
    if n_hits > 0:
        print(f'Cortex component {i_comp + 1}: {n_hits} hits')
        print(list(dfHO20minus30.columns[np.array(rel_mask_y[i_comp], dtype=bool)]))
np.save('rel_mask_y_full_THRESH%i' % THRESH, rel_mask_y)

# dump Cortex results in nifti format
# Load the significant mask and iterate through each component to create NIfTI images
rel_mask_y = np.load('rel_mask_y_full_THRESH%i.npy' % THRESH)
for i_comp in range(n_keep):
    out_nii = np.zeros(HO_atlas_cort.maps.shape)  # Initialize output NIfTI image
    comp_HO_weights = pls_left.y_loadings_[:, i_comp] - pls_right.y_loadings_[:, i_comp]
    comp_HO_weights[rel_mask_y[i_comp] == 0] = 0  # Zero out non-significant weights
    for i_feat in range(dfX.shape[-1]):  # Iterate over features to map weights to ROIs
        cur_feat_name = dfX.columns[i_feat].split(' (')[0]
        # Adjust feature names for specific cases
        if 'Stem' in cur_feat_name:
            pass
        else:
            cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name
        if 'Ventral Striatum' in cur_feat_name:
            cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

        b_found_roi = False
        # Map component weights to corresponding ROIs in the atlas
        for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
            if cur_feat_name in cort_label:
                b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label 
                print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

                out_nii[b_roi_mask] = comp_HO_weights[i_feat]
    
                b_found_roi = True
        # Map component weights to corresponding ROIs in the atlas
        for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
            if cur_feat_name in cort_label:
                b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label 
                n_roi_vox = np.sum(b_roi_mask)
                print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

                out_nii[b_roi_mask] = comp_HO_weights[i_feat]
                
                b_found_roi = True
    

    print('Comp %i: dumping %i region weights.' % (
        (i_comp + 1), np.sum(comp_HO_weights != 0)))
    
    SES_name = f'mode{i_mode + 1}'
    
    out_nii= nib.Nifti1Image(out_nii, HO_atlas_cort.maps.affine)
    out_nii.to_filename('AMsub2cortex/Hemispheric Difference Analysis/AMsub2HOcomp%i_%ihits_AMLeft_vs_AMRight.nii.gz' % ((i_comp + 1, int(n_hits))))

OUT_DIR_7 = 'AMsub2cortex/Hemispheric Difference Analysis' 

# Iterate over components to visualize and save differences in amygdala components
for counter, i_comp in enumerate(range(n_comps)): 
  n_rois = pls_right.x_loadings_.shape[0]
  X_AM_weights = pls_left.x_loadings_[:, i_comp] - pls_right.x_loadings_[:, i_comp]

  # Create a figure for the heatmap visualization
  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights
  
  # Prepare DataFrame of the weights, with modified column names for readability
  dfdata = pd.DataFrame(X_AM_weights, index=((dfAM20minus30_L.columns.str.replace('b', '')).str.replace("'",'')).str.replace('(left hemisphere)', 'Group Difference'), columns=[''])
  # Save the component weights to CSV and Excel files
  dfdata.to_csv('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.csv' % (OUT_DIR_7, counter + 1, SUFFIX))
  dfdata.to_excel('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.xls' % (OUT_DIR_7, counter + 1, SUFFIX))
  
  # Generate and customize the heatmap using seaborn
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                    cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                    square=True,
                    cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)

  # Adjust the heatmap display to ensure no cutoff at top/bottom
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  plt.tight_layout()
  plt.savefig('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_7, counter + 1, SUFFIX))

#Histogram
OUT_DIR_8 = 'AMsub2cortex/Amygdala Subregion Histograms' 

for name in dfAM20minus30.columns:
    # Clean column names for labeling
    dfAM20minus30_temp = dfAM20minus30.rename(columns = {(dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]) : ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')})
    hist = plt.figure()
    # Generate and save histogram
    dfAM20minus30_temp.hist(column = ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",''), bins=30)
    plt.savefig('%s/%s_Histogram.pdf' % (OUT_DIR_8, ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')), DPI=600)
    
    



####################################################################################

# Expression Level for each participant
#Fit Transform
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical

n_comps = 8  # Number of components to consider in the PLS analysis

# Perform PLS Canonical Analysis to explore the relationship between amygdala and cortex volume changes
pls_phe = PLSCanonical(n_components=n_comps)
X_m, Y_m = pls_phe.fit_transform(dfAM20minus30, dfHO20minus30)  # Fit and transform the data

# Initialize a DataFrame to store the expression levels for each participant
express = pd.DataFrame(columns=['index', 'x', 'y'])
express['index'] = age_T3.index
express.set_index("index", inplace=True)



#Age/Sex

# Extract and filter sex information for participants
sex = ukbb.loc[:, '31-0.0':'31-0.0'] 
sex = sex[subs_keep]  # Filter based on previously determined indices

# Group participants by age to identify those with the same age (and potentially sex)
age = age_T3[age_T3.duplicated(keep=False)]
age = age.groupby(age.columns.tolist()).apply(lambda x: tuple(x.index)).tolist()

# Create a DataFrame to differentiate between male and female participants with the same age
same_sex_age = pd.DataFrame(columns=['male', 'female'])

temp_m = []
temp_f = []
for i in range(len(age)):
    tuple_m = ()
    tuple_f = ()
    for index in age[i]:
        if int(sex.loc[index]) == 1:
            tuple_m = tuple_m + (index,)
        else:
            tuple_f = tuple_f + (index,)  
    temp_m.append(tuple_m)
    temp_f.append(tuple_f)

same_sex_age['male'] = temp_m
same_sex_age['female'] = temp_f
    
OUT_DIR_9 = 'AMsub2cortex/Age_Sex/Amygdala' 

# Loop over the number of components to analyze expression levels separately for male and female participants

for counter, mode in enumerate(range(n_comps)): 
    express['x'] = X_m[:, mode]  # Amygdala expression levels
    express['y'] = Y_m[:, mode]  # Cortex expression levels
    
    # Initialize lists to store median expression levels and thresholds for confidence intervals
    median_xm, median_xf = [], []
    lower_th_m, lower_th_f = [], []
    upper_th_m, upper_th_f = [], []

    temp_list_age = []

    # Analyze expression levels for male participants
    # This involves calculating median expression levels and confidence intervals
    
    for tuples in same_sex_age['male']:
        if len(tuples)>6:
            temp_list_x = []
            for index in tuples:
                temp_list_x.append(express.loc[index,'x'])
                temp_list_age.append(age_T3.loc[index,'21003-3.0'])
            median_xm.append(np.median(temp_list_x))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_x), len(temp_list_x))
                temp_list_x_train = [temp_list_x[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_x_train)))
            lower_th_m.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_m.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_m = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
            
    temp_list_age = []

    # Similar steps are followed for female participants
                             
    for tuples in same_sex_age['female']:
        if len(tuples)>10:
            temp_list_x = []
            for index in tuples:
                temp_list_x.append(express.loc[index,'x'])
                temp_list_age.append(age_T3.loc[index,'21003-3.0'])
            median_xf.append(np.median(temp_list_x))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_x), len(temp_list_x))
                temp_list_x_train = [temp_list_x[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_x_train)))
            lower_th_f.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_f.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_f = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
        
    # Plotting the results with error bars and linear fits to visualize trends

    plt.errorbar(age_m, median_xm, yerr=[lower_th_m,upper_th_m], fmt='o', color = 'dodgerblue', mec='black')
    plt.errorbar(age_f, median_xf, yerr=[lower_th_f,upper_th_f], fmt='o', color = 'purple', mec='black')

    plt.legend(["Male", "Female"], loc='lower left')
    
    m, b = np.polyfit(age_m, median_xm, 1)
    mf, bf = np.polyfit(age_f, median_xf, 1)
    plt.plot(age_m, m*np.array(age_m) + b, 'dodgerblue', mec = 'black')
    plt.plot(age_f, mf*np.array(age_f) + bf, 'purple', mec = 'black')
    
    plt.xlabel("Age (years)")
    plt.ylabel("Subject Expression of Amygdala Structual Plasticity (+/-)")
    
   
    plt.savefig('%s/Age_Sex_mode_%i.pdf' % (OUT_DIR_9, counter + 1), bbox_inches='tight')
    plt.show()

OUT_DIR_10 = 'AMsub2cortex/Age_Sex/Brain' 


# Similar steps are followed to analyze cortex expression levels
# The process involves fitting PLS models, extracting expression levels,
# and analyzing these levels in relation to age and sex, specifically for the cortex

for counter,mode  in enumerate(range(n_comps)):

    express['x'] = X_m[:,mode] # Amygdala expression levels
    express['y'] = Y_m[:,mode] # Cortex expression levels

    # Initialize lists to store median expression levels and thresholds for confidence intervals
    median_ym, median_yf = [], []
    lower_th_m, lower_th_f = [], []
    upper_th_m, upper_th_f = [], []

    temp_list_age = []

    # Analyze expression levels for male participants
    # This involves calculating median expression levels and confidence intervals

    for tuples in same_sex_age['male']:
        if len(tuples)>10:
            temp_list_y = []
            for index in tuples:
                temp_list_y.append(express.loc[index,'y'])
                temp_list_age.append(age_T3.loc[index,'21003-3.0'])
            median_ym.append(np.median(temp_list_y))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_y), len(temp_list_y))
                temp_list_y_train = [temp_list_y[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_y_train)))
            lower_th_m.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_m.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_m = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
            
    temp_list_age = []

    # Similar steps are followed for female participants
                             
    for tuples in same_sex_age['female']:
        if len(tuples)>6:
            temp_list_y = []
            for index in tuples:
                temp_list_y.append(express.loc[index,'x'])
                temp_list_age.append(age_T3.loc[index,'21003-3.0'])
            median_yf.append(np.median(temp_list_y))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_y), len(temp_list_y))
                temp_list_y_train = [temp_list_y[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_y_train)))
            lower_th_f.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_f.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_f = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
        
    # Plotting the results with error bars and linear fits to visualize trends

    plt.errorbar(age_m, median_ym, yerr=[lower_th_m,upper_th_m], fmt='o', color = 'dodgerblue', mec='black')
    plt.errorbar(age_f, median_yf, yerr=[lower_th_f,upper_th_f], fmt='o', color = 'purple', mec='black')
    
    plt.legend(["Male", "Female"], loc='lower left')
    
    m, b = np.polyfit(age_m, median_ym, 1)
    mf, bf = np.polyfit(age_f, median_yf, 1)
    plt.plot(age_m, m*np.array(age_m) + b, 'dodgerblue', mec = 'black')
    plt.plot(age_f, mf*np.array(age_f) + bf, 'purple', mec = 'black')
    
    plt.xlabel("Age (years)")
    plt.ylabel("Subject Expression of Brain Structual Plasticity (+/-)")
    
   
    plt.savefig('%s/Age_Sex_mode_%i.pdf' % (OUT_DIR_10, counter + 1), bbox_inches='tight')
    plt.show()


##############################################################################

#Age/Sex analysis 

#PheWAS with the 1414 participants at the first time point

pls_phe_20_1414 = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe_20_1414.fit_transform(
    pd.DataFrame(FS_AM20_ss, columns=dfS20.columns), pd.DataFrame(FS_HO20_ss, columns=ukbb_HO20.columns))
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T2.index)
express['index'] = age_T2.index
express.set_index("index", inplace=True)

sex = ukbb.loc[:, '31-0.0':'31-0.0'] 
sex = sex[subs_keep]

age = age_T2[age_T2.duplicated(keep=False)]
age = age.groupby(age.columns.tolist()).apply(lambda x: tuple(x.index)).tolist()

same_sex_age = pd.DataFrame(columns = ['male', 'female'])

temp_m = []
temp_f = []
for i in range(len(age)):
    tuple_m = ()
    tuple_f = ()
    for index in age[i]:
        if int(sex.loc[index]) == 1:
            tuple_m = tuple_m + (index,)
        else:
            tuple_f = tuple_f + (index,)  
    temp_m.append(tuple_m)
    temp_f.append(tuple_f)

same_sex_age['male'] = temp_m
same_sex_age['female'] = temp_f


#PheWAS with the 1414 participants at the first time point

OUT_DIR_10 = 'AMsub2cortex/Age_Sex/Brain/First Time Point' 

for counter,mode  in enumerate(range(n_comps)): 
    express['x'] = X_m[:,mode]
    express['y'] = Y_m[:,mode]
    
    median_ym = []
    median_yf = []
    
    lower_th_m = []
    lower_th_f = []
    upper_th_m = [] 
    upper_th_f = []
    
    temp_list_age = []
    
    for tuples in same_sex_age['male']:
        if len(tuples)>10:
            temp_list_y = []
            for index in tuples:
                temp_list_y.append(express.loc[index,'y'])
                temp_list_age.append(age_T2.loc[index,'21003-2.0'])
            median_ym.append(np.median(temp_list_y))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_y), len(temp_list_y))
                temp_list_y_train = [temp_list_y[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_y_train)))
            lower_th_m.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_m.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_m = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
            
    temp_list_age = []
                             
    for tuples in same_sex_age['female']:
        if len(tuples)>10:
            temp_list_y = []
            for index in tuples:
                temp_list_y.append(express.loc[index,'x'])
                temp_list_age.append(age_T2.loc[index,'21003-2.0'])
            median_yf.append(np.median(temp_list_y))
            it_diffs = []
            for i in range(100):
                bs_rs = np.random.RandomState(100)
                bs_sample_inds = bs_rs.randint(0, len(temp_list_y), len(temp_list_y))
                temp_list_y_train = [temp_list_y[i] for i in bs_sample_inds]
                it_diffs.append(np.median(np.array(temp_list_y_train)))
            lower_th_f.append(scoreatpercentile(it_diffs, 5, axis=0))
            upper_th_f.append(scoreatpercentile(it_diffs, 100 - 5, axis=0))
            
    temp_set = set()
    age_f = [x for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]
        
    
    plt.errorbar(age_m, median_ym, yerr=[lower_th_m,upper_th_m], fmt='o', color = 'dodgerblue', mec='black')
    plt.errorbar(age_f, median_yf, yerr=[lower_th_f,upper_th_f], fmt='o', color = 'purple', mec='black')
    
    plt.legend(["Male", "Female"], loc='lower left')
    
    m, b = np.polyfit(age_m, median_ym, 1)
    mf, bf = np.polyfit(age_f, median_yf, 1)
    plt.plot(age_m, m*np.array(age_m) + b, 'dodgerblue', mec = 'black')
    plt.plot(age_f, mf*np.array(age_f) + bf, 'purple', mec = 'black')
    
    plt.xlabel("Age (years)")
    plt.ylabel("Subject Expression of Brain Structual Plasticity (+/-)")
    
   
    plt.savefig('%s/Age_Sex_mode_%i.pdf' % (OUT_DIR_10, counter + 1), bbox_inches='tight')
    plt.show()


#####################################################################


# #PheWAS
# Expression Level for each participant used for the phewas analysis

#Fit Transform
n_comps = 8

#PheWAS with the 1414 participants at the first time point

pls_phe_20_1414 = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe_20_1414.fit_transform(
    pd.DataFrame(FS_AM20_ss, columns=dfS20.columns), pd.DataFrame(FS_HO20_ss, columns=ukbb_HO20.columns))
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T2.index)
express['index'] = age_T2.index
express.set_index("index", inplace=True)

eid = ukbb['eid']
eid = eid.loc[subs_keep]

#Download the amygdala per-participant expression levels 
amygdala_expression_df_20_1414 = pd.DataFrame({'eid': eid.values,
     'Comp 1' : X_m[:, 0],
     'Comp 2' : X_m[:, 1],
     'Comp 3' : X_m[:, 2],
     'Comp 4' : X_m[:, 3],
     'Comp 5' : X_m[:, 4],
     'Comp 6' : X_m[:, 5],
     'Comp 7' : X_m[:, 6],
     'Comp 8' : X_m[:, 7],
    })

amygdala_expression_df_20_1414.to_csv('amygdala_expression_df_20_1414.csv')

#Download the brain per-participant expression levels 
brain_expression_df_20_1414 = pd.DataFrame({'eid': eid.values,
     'Comp 1' : Y_m[:, 0],
     'Comp 2' : Y_m[:, 1],
     'Comp 3' : Y_m[:, 2],
     'Comp 4' : Y_m[:, 3],
     'Comp 5' : Y_m[:, 4],
     'Comp 6' : Y_m[:, 5],
     'Comp 7' : Y_m[:, 6],
     'Comp 8' : Y_m[:, 7],
    })

brain_expression_df_20_1414.to_csv('brain_expression_df_20_1414.csv')


#########################################################################
#MEDIAN RANKED CHANGE

# Both Sexes, 6 age groups MEDIAN

#Ranked Change
OUT_DIR_11 = 'AMsub2cortex/Ranked Change'

#Age/Sex
# Clean column names for labeling
for name in dfAM20minus30.columns:
    dfAM20minus30.rename(columns = {(dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]) : ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')}, inplace=True)

# Extract and filter sex information for participants
sex = ukbb.loc[:, '31-0.0':'31-0.0'] 
sex = sex[subs_keep]

# Group participants by age to identify those with the same age (and potentially sex)
age = age_T3[age_T3.duplicated(keep=False)]
age = age_T3.groupby(age_T3.columns.tolist()).apply(lambda x: tuple(x.index)).tolist()

# Create a DataFrame to differentiate between male and female participants with the same age
same_sex_age = pd.DataFrame(columns = ['male', 'female'])

temp_m = []
temp_f = []
for i in range(len(age)):
    tuple_m = ()
    tuple_f = ()
    for index in age[i]:
        if int(sex.loc[index]) == 1:
            tuple_m = tuple_m + (index,)
        else:
            tuple_f = tuple_f + (index,)  
    temp_m.append(tuple_m)
    temp_f.append(tuple_f)

same_sex_age['male'] = temp_m
same_sex_age['female'] = temp_f

# Calculate the difference in gray matter volume between time points for the amygdala and cortex

#Get the change in gray matter volume in the amygdala
dfAM30minus20 = pd.DataFrame(
	FS_AM30_ss - FS_AM20_ss, columns=dfAM20minus30.columns)
dfAM30minus20['age'] = age_T3.index
dfAM30minus20.set_index("age", inplace=True)

#Get the change in gray matter volume in the (sub)cortex
dfHO30minus20 = pd.DataFrame(
	FS_HO30_ss - FS_HO20_ss, columns=ukbb_HO20.columns)
dfHO30minus20['age'] = age_T3.index
dfHO30minus20.set_index("age", inplace=True)

form = dfAM20minus30.columns.tolist()

# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

median_list_55 = []
median_list_60 = []
median_list_65 = []
median_list_70 = []
median_list_75 = []
median_list_81 = []
    
# Calculate median changes in volume for each subregion and age range 
# (Code highlights the different age ranges analyzed, to be explicit)
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_55 = []
    temp_list_60 = []
    temp_list_65 = []
    temp_list_70 = []
    temp_list_75 = []
    temp_list_81 = []
    
    for tuples in age:
            for index in tuples:
                if age_T3.loc[tuples[0],'21003-3.0'] <55:
                    temp_list_55.append(dfAM30minus20.loc[index,name])
                elif 54 < age_T3.loc[tuples[0],'21003-3.0'] <60:
                    temp_list_60.append(dfAM30minus20.loc[index,name])
                elif 59 < age_T3.loc[tuples[0],'21003-3.0'] <65:
                    temp_list_65.append(dfAM30minus20.loc[index,name])
                elif 64 < age_T3.loc[tuples[0],'21003-3.0'] <70:
                    temp_list_70.append(dfAM30minus20.loc[index,name])
                elif 69 < age_T3.loc[tuples[0],'21003-3.0'] <75:
                    temp_list_75.append(dfAM30minus20.loc[index,name])
                elif 74 < age_T3.loc[tuples[0],'21003-3.0'] <82:
                    temp_list_81.append(dfAM30minus20.loc[index,name])
    median_list_55.append(np.median(temp_list_55))
    median_list_60.append(np.median(temp_list_60))
    median_list_65.append(np.median(temp_list_65))
    median_list_70.append(np.median(temp_list_70))
    median_list_75.append(np.median(temp_list_75))
    median_list_81.append(np.median(temp_list_81))
    median_list_55_sorted = [(median_list_55[i]) for i in np.argsort(median_list_55).tolist()]
    median_list_60_sorted = [(median_list_60[i]) for i in np.argsort(median_list_60).tolist()]
    median_list_65_sorted = [(median_list_65[i]) for i in np.argsort(median_list_65).tolist()]
    median_list_70_sorted = [(median_list_70[i]) for i in np.argsort(median_list_70).tolist()]
    median_list_75_sorted = [(median_list_75[i]) for i in np.argsort(median_list_75).tolist()]
    median_list_81_sorted = [(median_list_81[i]) for i in np.argsort(median_list_81).tolist()]
    
temp_age_lists = [median_list_55, median_list_60, median_list_65, median_list_70, median_list_75, median_list_81]

# Prepare DataFrames for visualization, one per age range, with subregions as rows

form = age_list     
dfAM30minus20_median_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_median_sorted_by_subregion.append(df)

  
for indx,age_range in enumerate(age_list):
    dfAM30minus20_median_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_median_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_median_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_median_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_median_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_median_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_median_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_median_sorted_by_subregion[indx][age_range.columns.values[0]])]
    
X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_median_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_median_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_median_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_median_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
# Visualize the ranked changes as a heatmap

age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Median Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Median Ranked Change 6 age ranges.xls' % (OUT_DIR_11))

ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'label': 'Median of the Difference in Gray Matter Volume','shrink': 0.25}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
plt.xlabel('Volume Loss <                                                                                                           > Volume Gain', fontsize=20)
plt.ylabel('Age Bracket', fontsize=20)

# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

plt.tight_layout()
plt.savefig('%s/Median Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))

#######################################################################

#Significant Component Analysis

# Setup for permutation testing
n_comps = 8
n_keep = 8
n_permutations = 1000
cur_X = np.array(dfAM20minus30)
cur_Y = np.array(dfHO20minus30)
perm_rs = np.random.RandomState(0)  # Random state 
perm_Rs = [] # Store permutation Pearson Rs
perm_scores = [] # Store permutation scores
n_except = 0 # Counter for exceptions

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical
pls = PLSCanonical(n_components=n_comps)
pls.fit(dfAM20minus30, dfHO20minus30)
r2 = pls.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`

est = pls
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
print(actual_Rs)
# [0.19654273, 0.24062467, 0.21549389, 0.19554875, 0.21232852, 0.25243018, 0.21673239, 0.22932024]

# Perform permutation testing
for i_iter in range(n_permutations):
    print(i_iter + 1)

    # Shuffle the Y matrix for permutation
    perm_rs.shuffle(cur_Y)

    try:
        perm_cca = PLSCanonical(n_components=n_keep, scale=False)  
        perm_cca.fit(cur_X, cur_Y)

         # Calculate Pearson correlation for the permuted data
        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        cur_score = perm_cca.score(cur_X, cur_Y)
        print(np.sort(perm_R)[::-1][:10])
        print(cur_score)
        perm_Rs.append(perm_R)
        perm_scores.append(cur_score)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))  # Append zeros in case of an exception
perm_Rs = np.array(perm_Rs)

pvals = []
for i_coef in range(n_keep):  # COMP-WISE comparison to permutation results !!!
    cur_pval = (np.sum(perm_Rs[:, i_coef] > actual_Rs[i_coef])) / n_permutations
    pvals.append(cur_pval)
    
# =========Change of gray matter volume between the two time points============ 
# Now - [0.004 !!!, 0.004 !!!, 0.025!!, 0.326, 0.12, 0.0!!!, 0.038!!, 0.005!!!]
# With 10000 permutations [0.0035!!!, 0.0048!!!, 0.0237!!, 0.3163, 0.1227, 0.0002!!!, 0.0432!!, 0.0036!!!]
# 2 CCs are significant at p<0.05
# 4 CCs are significant at p<0.01
# =============================================================================

# pvals = np.array(pvals)[inds_max_to_min]
# print(pvals)
# print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
# print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
# print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))
# [0.023 !!! 0.042 !!! 0.14  0.043 0.251 0.207 0.555 0.506 0.777 0.212]
# 3 CCs are significant at p<0.05
# 0 CCs are significant at p<0.01
