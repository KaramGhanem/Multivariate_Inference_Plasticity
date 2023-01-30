"""

ABC


TODO:
-add other diffusion MRI modalities



"""



import os
import os.path as op
import numpy as np
import nibabel as nib
import glob
import pandas as pd
import joblib
from nilearn import datasets as ds
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.image import resample_img, index_img, concat_imgs, math_img
import sklearn
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.decomposition import PCA


DECONF = True
# n_rois = 100
# n_nets = 7

TAR_ANA = 'AM'

# if 'ukbb' not in locals():
print('Fresh load of UKB databased...(go get some coffee !)')
ukbb = pd.read_csv('/Users/dblab/Desktop/Project Amygdala/Amygdala/ukb40500_cut_merged.csv/ukb40500_cut_merged.csv', low_memory=True)
# 8min 57s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each) 

# schaefer = ds.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=n_nets, resolution_mm=1, data_dir=None,
#                                                       base_url=None, resume=True, verbose=1)
# atlas = schaefer.maps
# atlas_nii = nib.load(atlas)
# atlas_masker = NiftiLabelsMasker(labels_img=atlas)
# atlas_masker.fit()
# from sklearn.preprocessing import OrdinalEncoder

# ROI_netnames = np.array(schaefer.labels, dtype=str)

# ROI_netnames7 = np.array([str(n).split('H_')[1].split('_')[0] for n in ROI_netnames])
# # ROI_netnames7[ROI_netnames == 'VisCent'] = 'Vis'
# # ROI_netnames7[ROI_netnames == 'VisPeri'] = 'Vis'
# # ROI_netnames7[ROI_netnames == 'TempPar'] = 'Default'
# net7_encoder = OrdinalEncoder(dtype=np.int)
# ROI_netnums7 = net7_encoder.fit_transform(ROI_netnames7[:, None])[:, 0]


# ROI_netnames = np.array([str(n).split('ks_')[1]for n in ROI_netnames])

# load region volumes
descr_dict = joblib.load('/Users/dblab/Desktop/Project Amygdala/Amygdala/descr_dict')

# ukbb_sMRI = ukbb.loc[:, '25782-2.0':'25920-2.0']  # FSL atlas including Diederichsen cerebellar atlas
ukbb_HO20 = ukbb.loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas
ukbb_HO20 = ukbb_HO20.iloc[:, ~ukbb_HO20.columns.str.contains('-3.0')]
ukbb_HO30 = ukbb.loc[:, '25782-3.0':'25892-3.0']  # FSL atlas without Diederichsen cerebellar atlas
ukbb_HO30 = ukbb_HO30.iloc[:, ~ukbb_HO30.columns.str.contains('-2.0')]

HO_vol_names = np.array([descr_dict[c]['descr'].split('Volume of grey matter in ')[1]
    for c in ukbb_HO20.columns])
ukbb_HO20.columns = HO_vol_names
ukbb_HO30.columns = HO_vol_names

# =============================================================================
eid = ukbb['eid']
# ukbb_HO30 = ukbb.loc[:, '21003-2.0':'21003-2.0'] 

age_T2 = ukbb.loc[:, '21003-2.0':'21003-2.0'] 
age_T3 = ukbb.loc[:, '21003-3.0':'21003-3.0'] 

sex = ukbb.loc[:, '31-0.0':'31-0.0'] 


# # Extract Wait Time between two time points
# import datetime
# date_format = "%Y-%m-%dT%H:%M:%S"
# v1datestr = ukbb['21862-2.0'].values
# v2datestr = ukbb['21862-3.0'].values

# v1date = []
# v2date = []
# for t in v1datestr:
#     if str(t) != 'nan':
#         v1date += [datetime.datetime.strptime(t, date_format)]
# for t in v2datestr:
#     if str(t) != 'nan':
#         v2date += [datetime.datetime.strptime(t, date_format)]

# waittime_days = np.zeros(len(v2date))
# for i in range(len(v2date)):
#                 waittime_days[i] = np.asarray([((v2date[i] - v1date[i]).days)/365])
# waittime_SS = StandardScaler()
# waittime_days = waittime_SS.fit_transform(waittime_days.reshape(-1,1))

# =============================================================================


def NonparametricImpute(input_vars):
	nan_inds = np.where(np.isnan(input_vars))[0]
	pres_inds = np.where(~np.isnan(input_vars))[0]
	rs = np.random.RandomState(0)
	rs.shuffle(pres_inds)
	input_vars[nan_inds] = input_vars[pres_inds[:len(nan_inds)]]
	return input_vars


# load AM subregion measurements
TAR_ANA = 'AM'
COLS_NAMES = []
COLS_IDS = []
# for fname in ['ukbbids_smoking_brain.txt', 'ukbbids_social_brain.txt', 'ukbbids_demographic.txt']:
# for fname in ['subcortical_labels_HC.txt']:
for fname in ['/Users/dblab/Desktop/Project Amygdala/Amygdala/subcortical_labels_%s.txt' % TAR_ANA]:
# for fname in ['ukbbids_social_brain.txt']:
    with open(fname) as f:
        lines=f.readlines()
        f.close()
        for line in lines:
            # if "(R)" in line:
            #     COLS_NAMES.append(line.split('\t'))
            a = line[:line.find('\t')]
            b = line[line.find('\t') + 1:].rsplit('\n')[0]
            COLS_IDS.append(a + '-2.0')
            COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
sub_dict = {COLS_IDS[i_col] : COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}


dfS20 = ukbb.loc[:, COLS_IDS]
# dfS.columns = COLS_NAMES
dfS20.columns = np.array([str(c.encode("ascii")) for c in COLS_NAMES])


# first follow-up measurements
TAR_ANA = 'AM'
COLS_NAMES = []
COLS_IDS = []
# for fname in ['ukbbids_smoking_brain.txt', 'ukbbids_social_brain.txt', 'ukbbids_demographic.txt']:
# for fname in ['subcortical_labels_HC.txt']:
for fname in ['/Users/dblab/Desktop/Project Amygdala/Amygdala/subcortical_labels_%s.txt' % TAR_ANA]:
# for fname in ['ukbbids_social_brain.txt']:
    with open(fname) as f:
        lines=f.readlines()
        f.close()
        for line in lines:
            # if "(R)" in line:
            #     COLS_NAMES.append(line.split('\t'))
            a = line[:line.find('\t')]
            b = line[line.find('\t') + 1:].rsplit('\n')[0]
            COLS_IDS.append(a + '-3.0')
            COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
sub_dict = {COLS_IDS[i_col] : COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}


dfS30 = ukbb.loc[:, COLS_IDS]
# dfS.columns = COLS_NAMES
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

FS_AM20_ss = NonparametricImpute(FS_AM20_ss)
FS_AM30_ss = NonparametricImpute(FS_AM30_ss)
FS_HO20_ss = NonparametricImpute(FS_HO20_ss)
FS_HO30_ss = NonparametricImpute(FS_HO30_ss)

assert np.all(~np.isnan(FS_AM20_ss))
assert np.all(~np.isnan(FS_AM30_ss))
assert np.all(~np.isnan(FS_HO20_ss))
assert np.all(~np.isnan(FS_HO30_ss))

keep_col_inds = ~dfS30.columns.str.contains('Whole')  # remove 6 whle-HC measures
dfS20 = dfS20.loc[:, keep_col_inds]
dfS30 = dfS30.loc[:, keep_col_inds]
FS_AM20 = FS_AM20[:, keep_col_inds]
FS_AM20_ss = FS_AM20_ss[:, keep_col_inds]
FS_AM30 = FS_AM30[:, keep_col_inds]
FS_AM30_ss = FS_AM30_ss[:, keep_col_inds]
COLS_NAMES = COLS_NAMES[keep_col_inds]


if DECONF == True:
    from nilearn.signal import clean

    beh = ukbb_2tp

    age = StandardScaler().fit_transform(beh['21022-0.0'].values[:, np.newaxis])  # Age at recruitment
    age2 = age ** 2
    sex = np.array(pd.get_dummies(beh['31-0.0']).values, dtype=np.int)  # Sex
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    head_motion_rest = np.nan_to_num(beh['25741-2.0'].values)  # Mean rfMRI head motion
    head_motion_task = np.nan_to_num(beh['25742-2.0'].values)  # Mean tfMRI head motion

    # added during previous paper revisions
    head_size = np.nan_to_num(beh['25006-2.0'].values)  # Volume of grey matter
    body_mass = np.nan_to_num(beh['21001-0.0'].values)  # BMI

    # motivated by Elliott et al., 2018
    head_pos_x = np.nan_to_num(beh['25756-2.0'].values)  # exact location of the head and the radio-frequency receiver coil in the scanner
    head_pos_y = np.nan_to_num(beh['25757-2.0'].values)
    head_pos_z = np.nan_to_num(beh['25758-2.0'].values)
    head_pos_table = np.nan_to_num(beh['25759-2.0'].values)
    scan_site_dummies = pd.get_dummies(beh['54-2.0']).values

    assert np.any(np.isnan(head_motion_rest)) == False
    assert np.any(np.isnan(head_motion_task)) == False
    assert np.any(np.isnan(head_size)) == False
    assert np.any(np.isnan(body_mass)) == False

    print('Deconfounding brain structural measures space!')
    conf_mat = np.hstack([
        # age, age2, sex, sex_x_age, sex_x_age2,
        np.atleast_2d(head_motion_rest).T, np.atleast_2d(head_motion_task).T,
        np.atleast_2d(head_size).T, np.atleast_2d(body_mass).T,

        np.atleast_2d(head_pos_x).T, np.atleast_2d(head_pos_y).T,
        np.atleast_2d(head_pos_z).T, np.atleast_2d(head_pos_table).T,
        np.atleast_2d(scan_site_dummies)
        ])

    FS_AM20_ss = clean(FS_AM20_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_AM30_ss = clean(FS_AM30_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_HO20_ss = clean(FS_HO20_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_HO30_ss = clean(FS_HO30_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    
dfAM20minus30 = pd.DataFrame(
	FS_AM20_ss - FS_AM30_ss, columns=dfS20.columns)
# print(dfAM20minus30.columns)
# dfAM20minus30 = dfAM20minus30.iloc[:, :-2]  # remove whole amygdala measures
print(dfAM20minus30.columns)

dfHO20minus30 = pd.DataFrame(
	FS_HO20_ss - FS_HO30_ss, columns=ukbb_HO20.columns)
print(dfHO20minus30.columns)
    
#difference between the volumes of the left amygdala subregions and the volumes of the right amygdala subregions at AM20(instance 2)
FS_AM20_ss_df = pd.DataFrame(
	FS_AM20_ss, columns=dfS20.columns)

FS_AM20_ss_L_minus_R = pd.DataFrame(FS_AM20_ss_df.filter(regex='left').values - FS_AM20_ss_df.filter(regex='right').values, columns=dfS20.loc[:,['left' in i for i in dfS20.columns]].columns.str.replace('left hemisphere', 'Left hemishphere minus Right hemisphere'))
print(FS_AM20_ss_L_minus_R.columns)

#difference between the volumes of the left cortex regions and the volumes of the cortex amygdala regions at HO20(instance 2)
FS_HO20_ss_df = pd.DataFrame(
	FS_HO20_ss, columns=ukbb_HO20.columns)

FS_HO20_ss_L_minus_R = pd.DataFrame(FS_HO20_ss_df.filter(regex='left').values - FS_HO20_ss_df.filter(regex='right').values, columns=ukbb_HO20.loc[:,['left' in i for i in ukbb_HO20.columns]].columns.str.replace('left', 'Left hemishphere minus Right hemisphere'))
#add brainstem to cortex data frame - doesnt belong to any hemisphere
FS_HO20_ss_L_minus_R.insert(FS_HO20_ss_L_minus_R.shape[-1],ukbb_HO20.columns[-1],FS_HO20_ss_df.values[:,-1])
print(FS_HO20_ss_L_minus_R.columns)

#Difference in volume between second and third timepoints of the left subregions of the amygdala
dfAM20minus30_L = pd.DataFrame(dfAM20minus30.filter(regex='left').values, columns=dfAM20minus30.loc[:,['left' in i for i in dfAM20minus30.columns]].columns)
#Difference in volume between second and third timepoints of the right subregions of the amygdala
dfAM20minus30_R = pd.DataFrame(dfAM20minus30.filter(regex='right').values, columns=dfAM20minus30.loc[:,['right' in i for i in dfAM20minus30.columns]].columns)

#Difference in volume between second and third timepoints of the left hemisphere of the cortex
dfHO20minus30_L = pd.DataFrame(dfHO20minus30.filter(regex='left').values, columns=dfHO20minus30.loc[:,['left' in i for i in dfHO20minus30.columns]].columns)
dfHO20minus30_L.insert(dfHO20minus30_L.shape[-1],ukbb_HO20.columns[-1],FS_HO20_ss_df.values[:,-1])
#Difference in volume between second and third timepoints of the right hemisphere of the cortex
dfHO20minus30_R = pd.DataFrame(dfHO20minus30.filter(regex='right').values, columns=dfHO20minus30.loc[:,['right' in i for i in dfHO20minus30.columns]].columns)
dfHO20minus30_R.insert(dfHO20minus30_R.shape[-1],ukbb_HO20.columns[-1],FS_HO20_ss_df.values[:,-1])

# STOP1  # done - loading the data

#Difference between second and third timepoints

from sklearn.cross_decomposition import PLSCanonical 
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical
pls = PLSCanonical(n_components=n_comps)
pls.fit(dfAM20minus30, dfHO20minus30)
r2 = pls.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca = CCA(n_components=n_comps)
# cca.fit(dfAM20minus30, dfHO20minus30)
# r2 = cca.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`


from scipy.stats import pearsonr

est = pls
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.19654273, 0.24062467, 0.21549389, 0.19554875, 0.21232852, 0.25243018, 0.21673239, 0.22932024]

# est = cca
# actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
#     zip(est.x_scores_.T, est.y_scores_.T)])
# # inds_max_to_min = np.argsort(actual_Rs)[::-1]
# # actual_Rs_sorted = actual_Rs[inds_max_to_min]
# print(actual_Rs)
# [0.44242675 0.38695511 0.37666435 0.36153381 0.3343162 ]



#Analysis of the differences in brain volumes between the left and right amygdalas and left and right hemispheres at the second interval timepoint
#PLS Canonical
pls_20 = PLSCanonical(n_components=n_comps)
pls_20.fit(FS_AM20_ss_L_minus_R, FS_HO20_ss_L_minus_R)
r2 = pls_20.score(FS_AM20_ss_L_minus_R, FS_HO20_ss_L_minus_R)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_20 = CCA(n_components=n_comps)
# cca_20.fit(FS_AM20_ss_L_minus_R, FS_HO20_ss_L_minus_R)
# r2 = cca_20.score(FS_AM20_ss_L_minus_R, FS_HO20_ss_L_minus_R)  # coefficient of determination :math:`R^2`

est = pls_20
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.3537046  0.24147282 0.20476178 0.1884665  0.18726884]

# est = cca_20
# actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
#     zip(est.x_scores_.T, est.y_scores_.T)])
# # inds_max_to_min = np.argsort(actual_Rs)[::-1]
# # actual_Rs_sorted = actual_Rs[inds_max_to_min]
# print(actual_Rs)
# # [0.41829755 0.30374785 0.2407011  0.22281107 0.20710428]




#Analysis of the differences of the left subregions of the amygdala and the cortex between second and third time points
#PLS Canonical
pls_left= PLSCanonical(n_components=n_comps)
pls_left.fit(dfAM20minus30_L, dfHO20minus30)
r2 = pls_left.score(dfAM20minus30_L, dfHO20minus30)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_left = CCA(n_components=n_comps)
# cca_left.fit(dfAM20minus30_L, dfHO20minus30)
# r2 = cca_left.score(dfAM20minus30_L, dfHO20minus30)  # coefficient of determination :math:`R^2`

est = pls_left
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.17969291 0.22206758 0.24594784 0.19278412 0.20976133]

# est = cca_left
# actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
#     zip(est.x_scores_.T, est.y_scores_.T)])
# # inds_max_to_min = np.argsort(actual_Rs)[::-1]
# # actual_Rs_sorted = actual_Rs[inds_max_to_min]
# print(actual_Rs)
# # [0.40945256 0.36340931 0.34345261 0.320587 0.29771575]




#Analysis of the differences of the right subregions of the amygdala and the cortex between second and third time points
#PLS Canonical
pls_right = PLSCanonical(n_components=n_comps)
pls_right.fit(dfAM20minus30_R, dfHO20minus30)
r2 = pls_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_right = CCA(n_components=n_comps)
# cca_right.fit(dfAM20minus30_R, dfHO20minus30)
# r2 = cca_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`


est = pls_right
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.15075179 0.21492126 0.21132531 0.22200016 0.20072646]

# est = cca_right
# actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
#     zip(est.x_scores_.T, est.y_scores_.T)])
# # inds_max_to_min = np.argsort(actual_Rs)[::-1]
# # actual_Rs_sorted = actual_Rs[inds_max_to_min]
# print(actual_Rs)
# # [0.4022312  0.35064107 0.33434872 0.28970732 0.28798194]

#Analysis of the differences of the right cortex hemisphere and the amygdala between second and third time points
#PLS Canonical
pls_right_cortex = PLSCanonical(n_components=n_comps)
pls_right_cortex.fit(dfAM20minus30, dfHO20minus30_R)
r2 = pls_right_cortex.score(dfAM20minus30, dfHO20minus30_R)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_right = CCA(n_components=n_comps)
# cca_right.fit(dfAM20minus30_R, dfHO20minus30)
# r2 = cca_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`


est = pls_right_cortex
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.19209712 0.20697045 0.19683422 0.15030597 0.22179664 0.19932314 0.18261532 0.19546436]

#Analysis of the differences of the left cortex hemisphere and the amygdala between second and third time points
#PLS Canonical
pls_left_cortex = PLSCanonical(n_components=n_comps)
pls_left_cortex.fit(dfAM20minus30, dfHO20minus30_L)
r2 = pls_left_cortex.score(dfAM20minus30, dfHO20minus30_L)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_right = CCA(n_components=n_comps)
# cca_right.fit(dfAM20minus30_R, dfHO20minus30)
# r2 = cca_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`


est = pls_left_cortex
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.19608387 0.22596609 0.14694996 0.17865733 0.19423987 0.19670539 0.15735294 0.1650333 ]

#Analysis of the gray matter volume of the cortex and the amygdala at the first time point
#PLS Canonical
pls_first= PLSCanonical(n_components=n_comps)
pls_first.fit(pd.DataFrame(FS_AM20_ss, columns=dfS20.columns), pd.DataFrame(FS_HO20_ss , columns=ukbb_HO20.columns))
r2 = pls_first.score(pd.DataFrame(FS_AM20_ss, columns=dfS20.columns),  pd.DataFrame(FS_HO20_ss , columns=ukbb_HO20.columns))  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca_right = CCA(n_components=n_comps)
# cca_right.fit(dfAM20minus30_R, dfHO20minus30)
# r2 = cca_right.score(dfAM20minus30_R, dfHO20minus30)  # coefficient of determination :math:`R^2`


est = pls_first
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.19608387 0.22596609 0.14694996 0.17865733 0.19423987 0.19670539 0.15735294 0.1650333 ]



from nilearn import datasets as ds
HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)


OUT_DIR = 'AMsub2cortex/dfAM20minus30/pls'
# OUT_DIR_1 = 'AMsub2cortex/dfAM20minus30/cca'
#Difference between second and third timepoints
for i_mode in range(n_comps):
	SES_in_brain_data = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca = np.zeros((HO_atlas_cort.maps.shape)) 
	dfX = dfHO20minus30

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data[b_roi_mask] = pls.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca[b_roi_mask] = cca.y_loadings_[i_feat, i_mode]

				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data[b_roi_mask] = pls.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca[b_roi_mask] = cca.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'

	SES_in_brain_nii = nib.Nifti1Image(SES_in_brain_data, HO_atlas_cort.maps.affine)
	SES_in_brain_nii.to_filename(OUT_DIR + '/' + SES_name + '_coef.nii.gz')
    
# 	SES_in_brain_nii_cca = nib.Nifti1Image(SES_in_brain_data_cca, HO_atlas_cort.maps.affine)
# 	SES_in_brain_nii_cca.to_filename(OUT_DIR_1 + '/' + SES_name + '_cca_coef.nii.gz')


#Differences in brain volumes between the left and right amygdalas and left and right hemispheres at the second interval timepoint  
OUT_DIR_2 = 'AMsub2cortex/AM20LminusR/pls'  
# OUT_DIR_3 = 'AMsub2cortex/AM20LminusR/cca'  
for i_mode in range(n_comps):
	SES_in_brain_data_20 = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_20 = np.zeros((HO_atlas_cort.maps.shape)) 
	dfX = FS_HO20_ss_L_minus_R 

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_20[b_roi_mask] = pls_20.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]

				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_20[b_roi_mask] = pls_20.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'
    
	SES_in_brain_nii_20 = nib.Nifti1Image(SES_in_brain_data_20, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_20.to_filename(OUT_DIR_2 + '/' + SES_name + '_20_coef.nii.gz')
    
# 	SES_in_brain_nii_cca_20 = nib.Nifti1Image(SES_in_brain_data_cca_20, HO_atlas_cort.maps.affine)
# 	SES_in_brain_nii_cca_20.to_filename(OUT_DIR_3 + '/' + SES_name + '_cca_20_coef.nii.gz')

#Left and Right subregions of the amygdala seperately and cortex regions difference between second and third time points
OUT_DIR_4 = 'AMsub2cortex/dfAM20minus30_L/pls' 
# OUT_DIR_5 = 'AMsub2cortex/dfAM20minus30_L/cca' 
OUT_DIR_6 = 'AMsub2cortex/dfAM20minus30_R/pls'
# OUT_DIR_7 = 'AMsub2cortex/dfAM20minus30_R/cca'  
for i_mode in range(n_comps):
	SES_in_brain_data_L = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_L = np.zeros((HO_atlas_cort.maps.shape)) 
    
	SES_in_brain_data_R = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_R = np.zeros((HO_atlas_cort.maps.shape)) 
    
	dfX = dfHO20minus30

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_L[b_roi_mask] = pls_left.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_L[b_roi_mask] = cca_left.y_loadings_[i_feat, i_mode]

				SES_in_brain_data_R[b_roi_mask] = pls_right.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_R[b_roi_mask] = cca_right.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_L[b_roi_mask] = pls_left.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_L[b_roi_mask] = cca_left.y_loadings_[i_feat, i_mode]
                
				SES_in_brain_data_R[b_roi_mask] = pls_right.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_R[b_roi_mask] = cca_right.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'
    
	SES_in_brain_nii_L = nib.Nifti1Image(SES_in_brain_data_L, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_L.to_filename(OUT_DIR_4 + '/' + SES_name + '_left_amygdala_coef.nii.gz')
    
# 	SES_in_brain_nii_L_cca = nib.Nifti1Image(SES_in_brain_data_cca_L, HO_atlas_cort.maps.affine)
# 	SES_in_brain_nii_L_cca.to_filename(OUT_DIR_5 + '/' + SES_name + '_cca_left_amygdala_coef.nii.gz')
    
	SES_in_brain_nii_R = nib.Nifti1Image(SES_in_brain_data_R, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_R.to_filename(OUT_DIR_6 + '/' + SES_name + '_right_amygdala_coef.nii.gz')
    
# 	SES_in_brain_nii_R_cca = nib.Nifti1Image(SES_in_brain_data_cca_R, HO_atlas_cort.maps.affine)
# 	SES_in_brain_nii_R_cca.to_filename(OUT_DIR_7 + '/' + SES_name + '_cca_right_amygdala_coef.nii.gz')
    
OUT_DIR_cortex_right = 'AMsub2cortex/dfHO20minus30_R' 
for i_mode in range(n_comps):
	SES_in_brain_data_cortex_right = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_20 = np.zeros((HO_atlas_cort.maps.shape)) 
	dfX = dfHO20minus30_R 

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex_right[b_roi_mask] = pls_right_cortex.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]

				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex_right[b_roi_mask] = pls_right_cortex.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'
    
	SES_in_brain_nii_cortex_right = nib.Nifti1Image(SES_in_brain_data_cortex_right, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_cortex_right.to_filename(OUT_DIR_cortex_right + '/' + SES_name + '_right_hemisphere_coef.nii.gz')

OUT_DIR_cortex_left = 'AMsub2cortex/dfHO20minus30_L' 
for i_mode in range(n_comps):
	SES_in_brain_data_cortex_left = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_20 = np.zeros((HO_atlas_cort.maps.shape)) 
	dfX = dfHO20minus30_L

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex_left[b_roi_mask] = pls_left_cortex.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]

				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex_left[b_roi_mask] = pls_left_cortex.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'
    
	SES_in_brain_nii_cortex_left = nib.Nifti1Image(SES_in_brain_data_cortex_left, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_cortex_left.to_filename(OUT_DIR_cortex_left + '/' + SES_name + '_left_hemisphere_coef.nii.gz')
    
OUT_DIR = 'AMsub2cortex/AM20' 
for i_mode in range(n_comps):
	SES_in_brain_data_cortex = np.zeros((HO_atlas_cort.maps.shape)) 
# 	SES_in_brain_data_cca_20 = np.zeros((HO_atlas_cort.maps.shape)) 
	dfX = pd.DataFrame(FS_HO20_ss , columns=ukbb_HO20.columns)

	for i_feat in range(dfX.shape[-1]):
		cur_feat_name = dfX.columns[i_feat].split(' (')[0]
		if 'Stem' in cur_feat_name:
			pass
		else:
			cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
		if 'Ventral Striatum' in cur_feat_name:
			cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

		b_found_roi = False
		for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex[b_roi_mask] = pls_first.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]

				b_found_roi = True

		for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
			if cur_feat_name in cort_label:
				b_roi_mask = HO_atlas_sub.maps.get_data() == i_cort_label
				n_roi_vox = np.sum(b_roi_mask)
				print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

				# SES_in_brain_data[b_roi_mask] = pls.x_loadings_[i_feat, i_mode]
				SES_in_brain_data_cortex[b_roi_mask] = pls_first.y_loadings_[i_feat, i_mode]
# 				SES_in_brain_data_cca_20[b_roi_mask] = cca_20.y_loadings_[i_feat, i_mode]
                
				b_found_roi = True

		if not b_found_roi:
			print('NOT Found: %s !!!' % (cur_feat_name))


	# store results
	SES_name = f'mode{i_mode + 1}'
    
	SES_in_brain_nii_cortex = nib.Nifti1Image(SES_in_brain_data_cortex, HO_atlas_cort.maps.affine)
	SES_in_brain_nii_cortex.to_filename(OUT_DIR + '/' + SES_name + '_first_time_point_coef.nii.gz')
    
import seaborn as sns

#Difference between second and third timepoints PLS
SUFFIX = ''
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls.x_loadings_.shape[0]
  X_AM_weights = pls.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(9, 6), dpi = 600)
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/pls_AM_topcomp%i%s_style_.csv' % (OUT_DIR, counter + 1, SUFFIX))
  dfdata.to_excel('%s/pls_AM_topcomp%i%s_style_.xls' % (OUT_DIR, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  # pyplot.figure(figsize=(15, 15))
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/pls_AM_topcomp%i%s_style_.png' % (OUT_DIR, counter + 1, SUFFIX), DPI=200)
  plt.savefig('%s/pls_AM_topcomp%i%s_style_.pdf' % (OUT_DIR, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

#second time point PLS
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_20.x_loadings_.shape[0]
  X_AM_weights = pls_20.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(FS_AM20_ss_L_minus_R.columns.str.replace('b', '')).str.replace("'",''), columns=['']) 
  dfdata.to_csv('%s/pls_20_AM_topcomp%i%s_style_.csv' % (OUT_DIR_2, counter + 1, SUFFIX))
  dfdata.to_excel('%s/pls_20_AM_topcomp%i%s_style_.xls' % (OUT_DIR_2, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/pls_20_AM_topcomp%i%s_style_.png' % (OUT_DIR_2, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/pls_20_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_2, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))
  
#Left Amygdala with cortex PLS
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_left.x_loadings_.shape[0]
  X_AM_weights = pls_left.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30_L.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/left_AM_topcomp%i%s_style_.csv' % (OUT_DIR_4, counter + 1, SUFFIX))
  dfdata.to_excel('%s/left_AM_topcomp%i%s_style_.xls' % (OUT_DIR_4, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/left_AM_topcomp%i%s_style_.png' % (OUT_DIR_4, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/left_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_4, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


#Right Amygdala with cortex PLS
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_right.x_loadings_.shape[0]
  X_AM_weights = pls_right.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30_R.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/right_AM_topcomp%i%s_style_.csv' % (OUT_DIR_6, counter + 1, SUFFIX))
  dfdata.to_excel('%s/right_AM_topcomp%i%s_style_.xls' % (OUT_DIR_6, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/right_AM_topcomp%i%s_style_.png' % (OUT_DIR_6, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/right_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_6, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

#Right Hemisphere with amygdala PLS
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_right_cortex.x_loadings_.shape[0]
  X_AM_weights = pls_right_cortex.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/right_Hemisphere_topcomp%i%s_style_.csv' % (OUT_DIR_cortex_right, counter + 1, SUFFIX))
  dfdata.to_excel('%s/right_Hemisphere_topcomp%i%s_style_.xls' % (OUT_DIR_cortex_right, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/right_Hemisphere_topcomp%i%s_style_.png' % (OUT_DIR_cortex_right, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/right_Hemisphere_topcomp%i%s_style_.pdf' % (OUT_DIR_cortex_right, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


#Left Hemisphere with amygdala PLS
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_left_cortex.x_loadings_.shape[0]
  X_AM_weights = pls_left_cortex.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/left_Hemisphere_topcomp%i%s_style_.csv' % (OUT_DIR_cortex_left, counter + 1, SUFFIX))
  dfdata.to_excel('%s/left_Hemisphere_topcomp%i%s_style_.xls' % (OUT_DIR_cortex_left, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  plt.savefig('%s/left_Hemisphere_topcomp%i%s_style_.png' % (OUT_DIR_cortex_left, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/left_Hemisphere_topcomp%i%s_style_.pdf' % (OUT_DIR_cortex_left, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))
  
#PLS at the first time point
for counter, i_comp in enumerate(range(n_comps)):
  # plt.figure(figsize=(14, 14))
  n_rois = pls_first.x_loadings_.shape[0]
  X_AM_weights = pls_first.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=(dfAM20minus30.columns.str.replace('b', '')).str.replace("'",''), columns=[''])
  dfdata.to_csv('%s/first_timepoint_topcomp%i%s_style_.csv' % (OUT_DIR, counter + 1, SUFFIX))
  dfdata.to_excel('%s/first_timepoint_topcomp%i%s_style_.xls' % (OUT_DIR, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                   cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                   square=True,
                   cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  # plt.savefig('%s/first_timepoint_topcomp%i%s_style_.png' % (OUT_DIR, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/first_timepoint_topcomp%i%s_style_.pdf' % (OUT_DIR, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


n_BS_perm = 100
n_keep = 8 
BS_diff = []
list_l_x = []
list_l_y = []
list_r_x = []
list_r_y = []

for i_BS in range(n_BS_perm):
    print(i_BS)

    bs_rs = np.random.RandomState(i_BS)
    bs_sample_inds = bs_rs.randint(0, len(dfAM20minus30_L), len(dfAM20minus30_L)) #left and right amygdala arrays have the same length

    bs_X_train_l = dfAM20minus30_L.iloc[bs_sample_inds, :]
    bs_X_train_r = dfAM20minus30_R.iloc[bs_sample_inds, :]
    bs_Y_train = dfHO20minus30.iloc[bs_sample_inds, :] 
    
    est_l = PLSCanonical(n_components=n_keep, scale=False)
    est_l.fit(bs_X_train_l, bs_Y_train)
    list_l_x.append(est_l.x_loadings_)
    list_l_y.append(est_l.y_loadings_)

    est_r = PLSCanonical(n_components=n_keep, scale=False)
    est_r.fit(bs_X_train_r, bs_Y_train)
    list_r_x.append(est_r.x_loadings_)
    list_r_y.append(est_r.y_loadings_)



today_stamp = '201010'
list_l_x = np.array(list_l_x)
np.save('BS_dump_list_full_l_x' + today_stamp, list_l_x)
list_l_y = np.array(list_l_y)
np.save('BS_dump_list_full_l_y' + today_stamp, list_l_y)
list_r_x = np.array(list_r_x)
np.save('BS_dump_list_full_r_x' + today_stamp, list_r_x)
list_r_y = np.array(list_r_y)
np.save('BS_dump_list_full_r_y' + today_stamp, list_r_y)

# AM side
it_diffs_x = np.zeros((n_BS_perm, n_keep, pls_left.x_loadings_.shape[0])) #pls_left and pls_right have the same shape 

for i_bs in range(n_BS_perm):
    for i_org_comp in range(n_keep):
        l_rhos_x = np.zeros((n_keep))
        for i_bs_comp in range(n_keep):
            l_rhos_x[i_bs_comp], _ = pearsonr(
                pls_left.x_loadings_[:, i_org_comp],
                list_l_x[i_bs][:, i_bs_comp])

        r_rhos_x = np.zeros((n_keep))
        for i_bs_comp in range(n_keep):
            r_rhos_x[i_bs_comp], _ = pearsonr(
                pls_right.x_loadings_[:, i_org_comp],
                list_r_x[i_bs][:, i_bs_comp])

        good_comp_l_x_ind = np.argmax(np.abs(l_rhos_x))
        good_comp_l_x = list_l_x[i_bs, :, good_comp_l_x_ind] * np.sign(l_rhos_x[good_comp_l_x_ind])

        good_comp_r_x_ind = np.argmax(np.abs(r_rhos_x))
        good_comp_r_x = list_r_x[i_bs, :, good_comp_r_x_ind] * np.sign(r_rhos_x[good_comp_r_x_ind])

        it_diffs_x[i_bs, i_org_comp] = good_comp_l_x - good_comp_r_x



from scipy.stats import scoreatpercentile

rel_mask_x = np.zeros((n_keep, pls_left.x_loadings_.shape[0])) #pls_left and pls_right have the same shape
THRESH = 10
for i_comp in range(n_keep):
    lower_th = scoreatpercentile(it_diffs_x[:, i_comp, :], THRESH, axis=0)
    upper_th = scoreatpercentile(it_diffs_x[:, i_comp, :], 100 - THRESH, axis=0)

    rel_mask_x[i_comp] = ((lower_th < 0) & (upper_th < 0)) | ((lower_th > 0) & (upper_th > 0))
    n_hits = np.sum(rel_mask_x[i_comp])
    if n_hits > 0:
        print('-' * 80)
        print('Amygdala component %i: %i hits' % ((i_comp + 1), n_hits))
        print(list(dfAM20minus30_L.columns[np.array(rel_mask_x[i_comp], dtype=bool)].str.replace('(left hemisphere)', ''))) #Distinguishing between left and right amygdala removed in columns
np.save('rel_mask_x_full_THRESH%i' % THRESH, rel_mask_x)


# HO side
it_diffs_y = np.zeros((n_BS_perm, n_keep, pls_left.y_loadings_.shape[0]))  #pls_left and pls_right have the same shape

for i_bs in range(n_BS_perm):
    for i_org_comp in range(n_keep):
        l_rhos_y = np.zeros((n_keep))
        for i_bs_comp in range(n_keep):
            l_rhos_y[i_bs_comp], _ = pearsonr(
                pls_left.y_loadings_[:, i_org_comp],
                list_l_y[i_bs][:, i_bs_comp])

        r_rhos_y = np.zeros((n_keep))
        for i_bs_comp in range(n_keep):
            r_rhos_y[i_bs_comp], _ = pearsonr(
                pls_right.y_loadings_[:, i_org_comp],
                list_r_y[i_bs][:, i_bs_comp])

        good_comp_l_y_ind = np.argmax(np.abs(l_rhos_y))
        good_comp_l_y = list_l_y[i_bs, :, good_comp_l_y_ind] * np.sign(l_rhos_y[good_comp_l_y_ind])

        good_comp_r_y_ind = np.argmax(np.abs(r_rhos_y))
        good_comp_r_y = list_r_y[i_bs, :, good_comp_r_y_ind] * np.sign(r_rhos_y[good_comp_r_y_ind])

        it_diffs_y[i_bs, i_org_comp] = good_comp_l_y - good_comp_r_y



from scipy.stats import scoreatpercentile

rel_mask_y = np.zeros((n_keep, pls_left.y_loadings_.shape[0])) #pls_left and pls_right have the same shape
THRESH = 10
for i_comp in range(n_keep):
    lower_th = scoreatpercentile(it_diffs_y[:, i_comp, :], THRESH, axis=0)
    upper_th = scoreatpercentile(it_diffs_y[:, i_comp, :], 100 - THRESH, axis=0)

    rel_mask_y[i_comp] = ((lower_th < 0) & (upper_th < 0)) | ((lower_th > 0) & (upper_th > 0))
    n_hits = np.sum(rel_mask_y[i_comp])
    if n_hits > 0:
        print('-' * 80)
        print('Cortex component %i: %i hits' % ((i_comp + 1), n_hits))
        print(list(dfHO20minus30.columns[np.array(rel_mask_y[i_comp], dtype=bool)]))
np.save('rel_mask_y_full_THRESH%i' % THRESH, rel_mask_y)

# dump Cortex results in nifti format
rel_mask_y = np.load('rel_mask_y_full_THRESH%i.npy' % THRESH)
for i_comp in range(n_keep):
    out_nii = np.zeros((HO_atlas_cort.maps.shape))
    dfX = dfHO20minus30
    comp_HO_weights = pls_left.y_loadings_[:, i_comp] - pls_right.y_loadings_[:, i_comp]
    comp_HO_weights[rel_mask_y[i_comp] == 0] = 0 
    n_hits = np.sum(rel_mask_y[i_comp])
    for i_feat in range(dfX.shape[-1]):
        cur_feat_name = dfX.columns[i_feat].split(' (')[0]
        if 'Stem' in cur_feat_name:
            pass
        else:
            cur_feat_name = ('Right ' if 'right' in dfX.columns[i_feat] else 'Left ') + cur_feat_name

		# HACK
        if 'Ventral Striatum' in cur_feat_name:
            cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')

        b_found_roi = False
        for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
            if cur_feat_name in cort_label:
                b_roi_mask = HO_atlas_cort.maps.get_data() == i_cort_label 
                print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))

                out_nii[b_roi_mask] = comp_HO_weights[i_feat]
    
                b_found_roi = True
    
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
# rel_mask_x = np.load('rel_mask_x_full_THRESH%i.npy' % THRESH)
for counter, i_comp in enumerate(range(n_comps)): 
  # plt.figure(figsize=(14, 14))
  n_rois = pls_right.x_loadings_.shape[0]
  X_AM_weights = pls_left.x_loadings_[:, i_comp] - pls_right.x_loadings_[:, i_comp]
  # tril_inds = np.tril_indices(n=n_rois, k=-1)

  f = plt.figure(figsize=(10, 7))
  X_comp_weights = np.zeros((n_rois, 1))
  X_comp_weights[:, 0] = X_AM_weights

  # dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
  dfdata = pd.DataFrame(X_AM_weights, index=((dfAM20minus30_L.columns.str.replace('b', '')).str.replace("'",'')).str.replace('(left hemisphere)', 'Group Difference'), columns=[''])
  dfdata.to_csv('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.csv' % (OUT_DIR_7, counter + 1, SUFFIX))
  dfdata.to_excel('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.xls' % (OUT_DIR_7, counter + 1, SUFFIX))


  # sns.set(font_scale=0.8)
  ax = sns.heatmap(dfdata, cbar=True, linewidths=.75,
                    cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                    square=True,
                    cmap=plt.cm.RdBu_r, center=0)

  ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
  ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
  # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
  # sns.despine(top=True, right=True, left=False, bottom=False)
  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values

  # ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
  #   cmap=plt.cm.RdBu_r, center=0)
  # plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
  # plt.colorbar()
  plt.tight_layout()

  # plt.savefig('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.png' % (OUT_DIR_7, counter + 1, SUFFIX), DPI=600)
  plt.savefig('%s/Hemispheric_Difference_AM_topcomp%i%s_style_.pdf' % (OUT_DIR_7, counter + 1, SUFFIX))
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
  # plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

#Histogram
OUT_DIR_8 = 'AMsub2cortex/Amygdala Subregion Histograms' 

for name in dfAM20minus30.columns:
    dfAM20minus30_temp = dfAM20minus30.rename(columns = {(dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]) : ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')})
    hist = plt.figure()
    dfAM20minus30_temp.hist(column = ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",''), bins=30)
    # plt.savefig('%s/%s_Histogram.png' % (OUT_DIR_8, ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')), DPI=600)
    plt.savefig('%s/%s_Histogram.pdf' % (OUT_DIR_8, ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')), DPI=600)
    
    
# Expression Level for each participant

#Fit Transform
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical
pls_phe= PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe.fit_transform(dfAM20minus30, dfHO20minus30)
express =  pd.DataFrame(columns=  ['index', 'x', 'y'])
express['index'] = age_T3.index
express.set_index("index", inplace=True)

#Age/Sex
sex = ukbb.loc[:, '31-0.0':'31-0.0'] 
sex = sex[subs_keep]

age = age_T3[age_T3.duplicated(keep=False)]
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
    
OUT_DIR_9 = 'AMsub2cortex/Age_Sex/Amygdala' 

for counter,mode  in enumerate(range(n_comps)): 
    express['x'] = X_m[:,mode]
    express['y'] = Y_m[:,mode]
    
    median_xm = []
    median_xf = []
    
    lower_th_m = []
    lower_th_f = []
    upper_th_m = [] 
    upper_th_f = []
    
    temp_list_age = []
    
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
    # plt.savefig('%s/Age_Sex_mode_%i.png' % (OUT_DIR_9, counter + 1), bbox_inches='tight')
    plt.show()

OUT_DIR_10 = 'AMsub2cortex/Age_Sex/Brain' 

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
    # plt.savefig('%s/Age_Sex_mode_%i.png' % (OUT_DIR_10, counter + 1), bbox_inches='tight')
    plt.show()

#Ranked Change
OUT_DIR_11 = 'AMsub2cortex/Ranked Change'

#Age/Sex
for name in dfAM20minus30.columns:
    dfAM20minus30.rename(columns = {(dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]) : ((dfAM20minus30.columns[dfAM20minus30.columns.get_loc(name)]).replace('b', '')).replace("'",'')}, inplace=True)

sex = ukbb.loc[:, '31-0.0':'31-0.0'] 
sex = sex[subs_keep]

age = age_T3[age_T3.duplicated(keep=False)]
age = age_T3.groupby(age_T3.columns.tolist()).apply(lambda x: tuple(x.index)).tolist()

# temp_list_age_2 = []
# for tuples in same_sex_age['male']:
#     for i in tuples:
#         temp_list_age_2.append([age_T2.loc[i,'21003-2.0'],age_T3.loc[i,'21003-3.0']] )


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

dfAM30minus20 = pd.DataFrame(
	FS_AM30_ss - FS_AM20_ss, columns=dfAM20minus30.columns)
dfAM30minus20['age'] = age_T3.index
dfAM30minus20.set_index("age", inplace=True)


dfHO30minus20 = pd.DataFrame(
	FS_HO30_ss - FS_HO20_ss, columns=ukbb_HO20.columns)
dfHO30minus20['age'] = age_T3.index
dfHO30minus20.set_index("age", inplace=True)

# male
# Create list of dataframes with each dataframe being a subregion with all the age ranges
form = dfAM20minus30.columns.tolist()
dfAM30minus20_mean_sorted_m = list()
dfAM30minus20_mean_unsorted_m = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    dfAM30minus20_mean_sorted_m.append(df)
    dfAM30minus20_mean_unsorted_m.append(df)
   
# Create age range axis
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_age_2 = []
    temp_list_age_3 = []
    for tuples in same_sex_age['male']:
        temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
        temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
        age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]

# Create list of dataframes for each age range to argsort the subregions within the same age range
form = age_unsorted
dfAM30minus20_mean_m_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_m_sorted_by_subregion.append(df)
    
# Filling up the lists of dataframes with their respective data
for idx,name in enumerate(dfAM30minus20.columns):
    mean_xm = []
    temp_list_age_2 = []
    temp_list_age_3 = []
  
    for tuples in same_sex_age['male']:
            temp_list_x = []
            for index in tuples:
                temp_list_x.append(dfAM30minus20.loc[index,name])
            temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
            temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
            mean_xm.append(np.mean(temp_list_x))
            mean_xm_sorted = [(mean_xm[i]) for i in np.argsort(mean_xm).tolist()]
            age_sorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in np.argsort(mean_xm).tolist()]
            age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]
    
    dfAM30minus20_mean_sorted_m[idx][name] = mean_xm_sorted
    dfAM30minus20_mean_sorted_m[idx]['age'] = age_sorted 
    dfAM30minus20_mean_unsorted_m[idx]['age'] = age_unsorted
    dfAM30minus20_mean_unsorted_m[idx][name] = mean_xm
    dfAM30minus20_mean_unsorted_m[idx].set_index("age", inplace=True)

temp_age_lists = []
for age_1 in age_unsorted:
    temp_age_values = []
    for i in range(len(dfAM30minus20_mean_unsorted_m)):
        temp_age_values.append(dfAM30minus20_mean_unsorted_m[i].loc[age_1].values[0])
    temp_age_lists.append(temp_age_values)
            
for indx,age_range in enumerate(age_unsorted):
    dfAM30minus20_mean_m_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_m_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_m_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_m_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_m_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_m_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_m_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_m_sorted_by_subregion[indx][age_range.columns.values[0]])]

X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_m_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_m_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_m_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_m_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_unsorted)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Male Ranked Change.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Male Ranked Change.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.2}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# 'lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
# textstr = 'lAb = left Accessory-Basal-Nucleus\nrAb = right Accessory-Basal-Nucleus\nlAAAleft Anterior-amygdaloid-area\n'
# plt.text(2000, 2000, textstr, fontsize=14)
# plt.yticks(rotation = 45)

plt.savefig('%s/Male Ranked Change.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


    
# temp_set = set()
# age_m_2 = [int(x) for x in temp_list_age_2 if x not in temp_set and (temp_set.add(x) or True)]  
# temp_set = set()
# age_m_3 = [int(x) for x in temp_list_age_3 if x not in temp_set and (temp_set.add(x) or True)]

# dfAM30minus20_mean_sorted_m['age'] = age_m_3
# dfAM30minus20_mean_sorted_m.set_index("age", inplace=True)

# female
# Create list of dataframes with each dataframe being a subregion with all the age ranges
form = dfAM20minus30.columns.tolist()
dfAM30minus20_mean_sorted_f = list()
dfAM30minus20_mean_unsorted_f = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    dfAM30minus20_mean_sorted_f.append(df)
    dfAM30minus20_mean_unsorted_f.append(df)
   
# Create age range axis
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_age_2 = []
    temp_list_age_3 = []
    for tuples in same_sex_age['female']:
        temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
        temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
        age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]

# Create list of dataframes for each age range to argsort the subregions within the same age range
form = age_unsorted
dfAM30minus20_mean_f_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_f_sorted_by_subregion.append(df)
    
# Filling up the lists of dataframes with their respective data
for idx,name in enumerate(dfAM30minus20.columns):
    mean_xf = []
    temp_list_age_2 = []
    temp_list_age_3 = []
  
    for tuples in same_sex_age['female']:
            temp_list_x = []
            for index in tuples:
                temp_list_x.append(dfAM30minus20.loc[index,name])
            temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
            temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
            mean_xf.append(np.mean(temp_list_x))
            mean_xf_sorted = [(mean_xf[i]) for i in np.argsort(mean_xf).tolist()]
            age_sorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in np.argsort(mean_xf).tolist()] # T1 has a +/- 1 
            age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]
    
    dfAM30minus20_mean_sorted_f[idx][name] = mean_xf_sorted
    dfAM30minus20_mean_sorted_f[idx]['age'] = age_sorted 
    dfAM30minus20_mean_unsorted_f[idx]['age'] = age_unsorted
    dfAM30minus20_mean_unsorted_f[idx][name] = mean_xf
    dfAM30minus20_mean_unsorted_f[idx].set_index("age", inplace=True)

temp_age_lists = []
for age_1 in age_unsorted:
    temp_age_values = []
    for i in range(len(dfAM30minus20_mean_unsorted_f)):
        temp_age_values.append(dfAM30minus20_mean_unsorted_f[i].loc[age_1].values[0])
    temp_age_lists.append(temp_age_values)
            
for indx,age_range in enumerate(age_unsorted):
    dfAM30minus20_mean_f_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_f_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_f_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_f_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_f_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_f_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_f_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_f_sorted_by_subregion[indx][age_range.columns.values[0]])]

X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_f_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_f_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_f_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_f_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_unsorted)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Female Ranked Change.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Female Ranked Change.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Female Ranked Change.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


    
    
# temp_set = set()
# age_f = [int(x) for x in temp_list_age if x not in temp_set and (temp_set.add(x) or True)]

# dfAM30minus20_mean_sorted_f['age'] = age_m
# dfAM30minus20_mean_sorted_f.set_index("age", inplace=True)

# all genders

form = dfAM20minus30.columns.tolist()
dfAM30minus20_mean_sorted = list()
dfAM30minus20_mean_unsorted = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    dfAM30minus20_mean_sorted.append(df)
    dfAM30minus20_mean_unsorted.append(df)
   
# Create age range axis
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_age_2 = []
    temp_list_age_3 = []
    for tuples in age:
        temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
        temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
        age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]

# Create list of dataframes for each age range to argsort the subregions within the same age range
form = age_unsorted
dfAM30minus20_mean_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion.append(df)
    
# Filling up the lists of dataframes with their respective data
for idx,name in enumerate(dfAM30minus20.columns):
    mean_x = []
    temp_list_age_2 = []
    temp_list_age_3 = []
  
    for tuples in age:
            temp_list_x = []
            for index in tuples:
                temp_list_x.append(dfAM30minus20.loc[index,name])
            temp_list_age_2.append(age_T2.loc[tuples[0],'21003-2.0'])
            temp_list_age_3.append(age_T3.loc[tuples[0],'21003-3.0'])
            mean_x.append(np.mean(temp_list_x))
            mean_x_sorted = [(mean_x[i]) for i in np.argsort(mean_x).tolist()]
            age_sorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in np.argsort(mean_x).tolist()]
            age_unsorted = [str(int(temp_list_age_2[i])) + '-' + str(int(temp_list_age_3[i])) for i in range(len(temp_list_age_3))]
    
    dfAM30minus20_mean_sorted[idx][name] = mean_x_sorted
    dfAM30minus20_mean_sorted[idx]['age'] = age_sorted 
    dfAM30minus20_mean_unsorted[idx]['age'] = age_unsorted
    dfAM30minus20_mean_unsorted[idx][name] = mean_x
    dfAM30minus20_mean_unsorted[idx].set_index("age", inplace=True)

temp_age_lists = []
for age_1 in age_unsorted:
    temp_age_values = []
    for i in range(len(dfAM30minus20_mean_unsorted)):
        temp_age_values.append(dfAM30minus20_mean_unsorted[i].loc[age_1].values[0])
    temp_age_lists.append(temp_age_values)
            
for indx,age_range in enumerate(age_unsorted):
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]])]

X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_unsorted)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Ranked Change.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Ranked Change.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Ranked Change.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

# Male, 6 age ranges
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

mean_list_55 = []
mean_list_60 = []
mean_list_65 = []
mean_list_70 = []
mean_list_75 = []
mean_list_81 = []
    
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_55 = []
    temp_list_60 = []
    temp_list_65 = []
    temp_list_70 = []
    temp_list_75 = []
    temp_list_81 = []
    
    for tuples in same_sex_age['male']:
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
    mean_list_55.append(np.mean(temp_list_55))
    mean_list_60.append(np.mean(temp_list_60))
    mean_list_65.append(np.mean(temp_list_65))
    mean_list_70.append(np.mean(temp_list_70))
    mean_list_75.append(np.mean(temp_list_75))
    mean_list_81.append(np.mean(temp_list_81))
    mean_list_55_sorted = [(mean_list_55[i]) for i in np.argsort(mean_list_55).tolist()]
    mean_list_60_sorted = [(mean_list_60[i]) for i in np.argsort(mean_list_60).tolist()]
    mean_list_65_sorted = [(mean_list_65[i]) for i in np.argsort(mean_list_65).tolist()]
    mean_list_70_sorted = [(mean_list_70[i]) for i in np.argsort(mean_list_70).tolist()]
    mean_list_75_sorted = [(mean_list_75[i]) for i in np.argsort(mean_list_75).tolist()]
    mean_list_81_sorted = [(mean_list_81[i]) for i in np.argsort(mean_list_81).tolist()]
    
temp_age_lists = [mean_list_55, mean_list_60, mean_list_65, mean_list_70, mean_list_75, mean_list_81]

form = age_list     
dfAM30minus20_mean_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion.append(df)

  
for indx,age_range in enumerate(age_list):
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]])]
    
X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Male Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Male Ranked Change 6 age ranges.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Male Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


# Female, 6 age groups
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

mean_list_55 = []
mean_list_60 = []
mean_list_65 = []
mean_list_70 = []
mean_list_75 = []
mean_list_81 = []
    
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_55 = []
    temp_list_60 = []
    temp_list_65 = []
    temp_list_70 = []
    temp_list_75 = []
    temp_list_81 = []
    
    for tuples in same_sex_age['female']:
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
    mean_list_55.append(np.mean(temp_list_55))
    mean_list_60.append(np.mean(temp_list_60))
    mean_list_65.append(np.mean(temp_list_65))
    mean_list_70.append(np.mean(temp_list_70))
    mean_list_75.append(np.mean(temp_list_75))
    mean_list_81.append(np.mean(temp_list_81))
    mean_list_55_sorted = [(mean_list_55[i]) for i in np.argsort(mean_list_55).tolist()]
    mean_list_60_sorted = [(mean_list_60[i]) for i in np.argsort(mean_list_60).tolist()]
    mean_list_65_sorted = [(mean_list_65[i]) for i in np.argsort(mean_list_65).tolist()]
    mean_list_70_sorted = [(mean_list_70[i]) for i in np.argsort(mean_list_70).tolist()]
    mean_list_75_sorted = [(mean_list_75[i]) for i in np.argsort(mean_list_75).tolist()]
    mean_list_81_sorted = [(mean_list_81[i]) for i in np.argsort(mean_list_81).tolist()]
    
temp_age_lists = [mean_list_55, mean_list_60, mean_list_65, mean_list_70, mean_list_75, mean_list_81]

form = age_list     
dfAM30minus20_mean_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion.append(df)

  
for indx,age_range in enumerate(age_list):
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]])]
    
X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Female Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Female Ranked Change 6 age ranges.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Female Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))


# Both Sexes, 6 age groups
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

mean_list_55 = []
mean_list_60 = []
mean_list_65 = []
mean_list_70 = []
mean_list_75 = []
mean_list_81 = []
    
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
    mean_list_55.append(np.mean(temp_list_55))
    mean_list_60.append(np.mean(temp_list_60))
    mean_list_65.append(np.mean(temp_list_65))
    mean_list_70.append(np.mean(temp_list_70))
    mean_list_75.append(np.mean(temp_list_75))
    mean_list_81.append(np.mean(temp_list_81))
    mean_list_55_sorted = [(mean_list_55[i]) for i in np.argsort(mean_list_55).tolist()]
    mean_list_60_sorted = [(mean_list_60[i]) for i in np.argsort(mean_list_60).tolist()]
    mean_list_65_sorted = [(mean_list_65[i]) for i in np.argsort(mean_list_65).tolist()]
    mean_list_70_sorted = [(mean_list_70[i]) for i in np.argsort(mean_list_70).tolist()]
    mean_list_75_sorted = [(mean_list_75[i]) for i in np.argsort(mean_list_75).tolist()]
    mean_list_81_sorted = [(mean_list_81[i]) for i in np.argsort(mean_list_81).tolist()]
    
temp_age_lists = [mean_list_55, mean_list_60, mean_list_65, mean_list_70, mean_list_75, mean_list_81]

form = age_list     
dfAM30minus20_mean_sorted_by_subregion = list()
for i in form:
    df=pd.DataFrame()
    df[i] = i 
    df['subregions'] = dfAM20minus30.columns.tolist()
    df.set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion.append(df)

  
for indx,age_range in enumerate(age_list):
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range] = temp_age_lists[indx]

abbv_columns = ['lAB','rAB','lAAA','rAAA','lBa','rBa','lCe','rCe','lCo','rCo','lCAT','rCAT','lLa','rLa','lMe','rMe','lPL','rPL']
for indx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    temp_columns = []
    temp_columns = [abbv_columns[i] for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]]).tolist()]
    dfAM30minus20_mean_sorted_by_subregion[indx]['subregions'] = temp_columns
    dfAM30minus20_mean_sorted_by_subregion[indx].set_index("subregions", inplace=True)
    dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]] = [(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]].iloc[i]) for i in pd.Series.argsort(dfAM30minus20_mean_sorted_by_subregion[indx][age_range.columns.values[0]])]
    
X_AM_weights = []
labels = []
n_rois = len(dfAM30minus20_mean_sorted_by_subregion)

for idx,age_range in enumerate(dfAM30minus20_mean_sorted_by_subregion):
    X_AM_weights.append(np.array(dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].values))
    labels.append((dfAM30minus20_mean_sorted_by_subregion[idx][age_range.columns.values[0]].index.values))
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Ranked Change 6 age ranges.xls' % (OUT_DIR_11))


# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.2}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

# #PheWAS

# Expression Level for each participant

#Fit Transform
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical

pls_phe = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe.fit_transform(dfAM20minus30, dfHO20minus30)
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T3.index)
express['index'] = age_T3.index
express.set_index("index", inplace=True)

amygdala_expression_df = pd.DataFrame({'eid': eid.values,
     'Comp 1' : X_m[:, 0],
     'Comp 2' : X_m[:, 1],
     'Comp 3' : X_m[:, 2],
     'Comp 4' : X_m[:, 3],
     'Comp 5' : X_m[:, 4],
     'Comp 6' : X_m[:, 5],
     'Comp 7' : X_m[:, 6],
     'Comp 8' : X_m[:, 7],
    })

amygdala_expression_df.to_csv('amygdala_expression_df.csv')


brain_expression_df = pd.DataFrame({'eid': eid.values,
     'Comp 1' : Y_m[:, 0],
     'Comp 2' : Y_m[:, 1],
     'Comp 3' : Y_m[:, 2],
     'Comp 4' : Y_m[:, 3],
     'Comp 5' : Y_m[:, 4],
     'Comp 6' : Y_m[:, 5],
     'Comp 7' : Y_m[:, 6],
     'Comp 8' : Y_m[:, 7],
    })

brain_expression_df.to_csv('brain_expression_df.csv')
  
  

expression_df = pd.DataFrame({'eid': np.concatenate((eid.values,eid.values)),
     'Comp 1' : np.concatenate((X_m[:, 0], Y_m[:, 0])),
     'Comp 2' : np.concatenate((X_m[:, 1], Y_m[:, 1])),
     'Comp 3' : np.concatenate((X_m[:, 2], Y_m[:, 2])),
     'Comp 4' : np.concatenate((X_m[:, 3], Y_m[:, 3])),
     'Comp 5' : np.concatenate((X_m[:, 4], Y_m[:, 4])),
     'Comp 6' : np.concatenate((X_m[:, 5], Y_m[:, 5])),
     'Comp 7' : np.concatenate((X_m[:, 6], Y_m[:, 6])),
     'Comp 8' : np.concatenate((X_m[:, 7], Y_m[:, 7])),
    })

expression_df.to_csv('expression_df.csv')

expression_switch_df = pd.DataFrame({'eid': np.concatenate((eid.values,eid.values)),
     'Comp 1' : np.concatenate((Y_m[:, 0], X_m[:, 0])),
     'Comp 2' : np.concatenate((Y_m[:, 1], X_m[:, 1])),
     'Comp 3' : np.concatenate((Y_m[:, 2], X_m[:, 2])),
     'Comp 4' : np.concatenate((Y_m[:, 3], X_m[:, 3])),
     'Comp 5' : np.concatenate((Y_m[:, 4], X_m[:, 4])),
     'Comp 6' : np.concatenate((Y_m[:, 5], X_m[:, 5])),
     'Comp 7' : np.concatenate((Y_m[:, 6], X_m[:, 6])),
     'Comp 8' : np.concatenate((Y_m[:, 7], X_m[:, 7])),
    })
  
expression_switch_df.to_csv('expression_switch_df.csv')

# Checking relationship of a phenotype with other phenotypes
BASE_FOLDER = '/Users/dblab/Desktop/Project Amygdala/Amygdala/plotting_miami'
y_group = "_miller_mh_v1"


import manhattan_plot_util as man_plot
ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(y_group, BASE_FOLDER)

# POI = pd.DataFrame({'eid': ukbb_y['userID'][subs_keep],
#      'Comp 1' : ukbb_y['5001_5.0'][subs_keep]
#     }) 

corrdf = man_plot.phenom_correlat(brain_expression_df, ukbb_y, y_desc_dict, y_cat_dict)

lk = 'Comp 5'

thres = 0.05; n_t = corrdf.shape[0]; 
thresBon = thres/n_t
thresFDR = man_plot.findFDR(corrdf, lk, thresBon)

print(f"The -log10 of the BON for {lk} is {-np.log10(thresBon):.2f}")
print(f"The -log10 of the FDR for {lk} is {-np.log10(thresFDR):.2f}")

print("Before using the cat_name function, our categories were called: \n")
print('\n'.join(np.unique(y_cat_dict["Cat_Name"])))

cat_name = man_plot.cat_name(corrdf, y_cat_dict)
print("\n"+ 35*'-'+"\n\nAfter the cat_name function, our categories are:\n")
print('\n\n'.join(cat_name))

man_plot.manhattan_plot(corrdf, lk, cat_name)

cat_name = man_plot.cat_name(corrdf, y_cat_dict)

man_plot.manhattan_plot(corrdf, 'Comp 1', cat_name, ylim=(0,20), plot_height=10)

plt.savefig(os.path.join(BASE_FOLDER, "FI7 : synonym.pdf" + ".pdf"))

man_plot.hits(corrdf, 'Comp 1', y_desc_dict, y_cat_dict, useFDR= False)

coid = corrdf.loc[:,'phesid']
corrdf.iloc[coid[coid == '4957_1.0'].index[0]]

coid = corrdf.loc[:,'phesid']
corrdf.iloc[coid[coid == '5001_5.0'].index[0]]

coid = corrdf.loc[:,'phesid']
corrdf.iloc[coid[coid == '5012_25.0'].index[0]]

#PheWAS with the 40,000 participants at the first time point
# load AM subregion measurements
TAR_ANA = 'AM'
COLS_NAMES = []
COLS_IDS = []
# for fname in ['ukbbids_smoking_brain.txt', 'ukbbids_social_brain.txt', 'ukbbids_demographic.txt']:
# for fname in ['subcortical_labels_HC.txt']:
for fname in ['subcortical_labels_%s.txt' % TAR_ANA]:
# for fname in ['ukbbids_social_brain.txt']:
    with open(fname) as f:
        lines=f.readlines()
        f.close()
        for line in lines:
            # if "(R)" in line:
            #     COLS_NAMES.append(line.split('\t'))
            a = line[:line.find('\t')]
            b = line[line.find('\t') + 1:].rsplit('\n')[0]
            COLS_IDS.append(a + '-2.0')
            COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
sub_dict = {COLS_IDS[i_col] : COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}


dfS20_all = ukbb.loc[:, COLS_IDS]
# dfS.columns = COLS_NAMES
dfS20_all.columns = np.array([str(c.encode("ascii")) for c in COLS_NAMES])

FS_AM20_all = dfS20_all.values
FS_AM20_all_ss = S_scaler.fit_transform(FS_AM20_all)

# ukbb_sMRI = ukbb.loc[:, '25782-2.0':'25920-2.0']  # FSL atlas including Diederichsen cerebellar atlas
ukbb_HO20_all = ukbb.loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas
ukbb_HO20_all = ukbb_HO20_all.iloc[:, ~ukbb_HO20_all.columns.str.contains('-3.0')]
# ukbb_HO30 = ukbb.loc[:, '25782-3.0':'25892-3.0']  # FSL atlas without Diederichsen cerebellar atlas
# ukbb_HO30 = ukbb_HO30.iloc[:, ~ukbb_HO30.columns.str.contains('-2.0')]

HO_vol_names = np.array([descr_dict[c]['descr'].split('Volume of grey matter in ')[1]
    for c in ukbb_HO20_all.columns])
ukbb_HO20_all.columns = HO_vol_names
# ukbb_HO30.columns = HO_vol_names

FS_HO20_all= ukbb_HO20_all.values
FS_HO20_all_ss = HO_scaler.fit_transform(FS_HO20_all)

idx_nonAM = ~ukbb_HO20_all.columns.str.contains('Amygdala')
FS_HO20_all = FS_HO20_all[:, idx_nonAM]
FS_HO20_ss_all = FS_HO20_all_ss[:, idx_nonAM]

FS_AM20_all_ss = NonparametricImpute(FS_AM20_all_ss)
FS_HO20_all_ss = NonparametricImpute(FS_HO20_all_ss)

ukbb_2tp = ukbb 
if DECONF == True:
    beh = ukbb_2tp

    age = StandardScaler().fit_transform(beh['21022-0.0'].values[:, np.newaxis])  # Age at recruitment
    age2 = age ** 2
    sex = np.array(pd.get_dummies(beh['31-0.0']).values, dtype=np.int)  # Sex
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    head_motion_rest = np.nan_to_num(beh['25741-2.0'].values)  # Mean rfMRI head motion
    head_motion_task = np.nan_to_num(beh['25742-2.0'].values)  # Mean tfMRI head motion

    # added during previous paper revisions
    head_size = np.nan_to_num(beh['25006-2.0'].values)  # Volume of grey matter
    body_mass = np.nan_to_num(beh['21001-0.0'].values)  # BMI

    # motivated by Elliott et al., 2018
    head_pos_x = np.nan_to_num(beh['25756-2.0'].values)  # exact location of the head and the radio-frequency receiver coil in the scanner
    head_pos_y = np.nan_to_num(beh['25757-2.0'].values)
    head_pos_z = np.nan_to_num(beh['25758-2.0'].values)
    head_pos_table = np.nan_to_num(beh['25759-2.0'].values)
    scan_site_dummies = pd.get_dummies(beh['54-2.0']).values

    assert np.any(np.isnan(head_motion_rest)) == False
    assert np.any(np.isnan(head_motion_task)) == False
    assert np.any(np.isnan(head_size)) == False
    assert np.any(np.isnan(body_mass)) == False

    print('Deconfounding brain structural measures space!')
    conf_mat = np.hstack([
        # age, age2, sex, sex_x_age, sex_x_age2,
        np.atleast_2d(head_motion_rest).T, np.atleast_2d(head_motion_task).T,
        np.atleast_2d(head_size).T, np.atleast_2d(body_mass).T,

        np.atleast_2d(head_pos_x).T, np.atleast_2d(head_pos_y).T,
        np.atleast_2d(head_pos_z).T, np.atleast_2d(head_pos_table).T,
        np.atleast_2d(scan_site_dummies)
        ])

    FS_AM20_all_ss = clean(FS_AM20_all_ss, confounds=conf_mat,
                     detrend=False, standardize=False)
    FS_HO20_all_ss = clean(FS_HO20_all_ss, confounds=conf_mat,
                     detrend=False, standardize=False)

from sklearn.cross_decomposition import PLSCanonical 
n_comps = 8

pls_phe_20 = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe_20.fit_transform(
    pd.DataFrame(FS_AM20_all_ss, columns=dfS20_all.columns), pd.DataFrame(FS_HO20_all_ss, columns=ukbb_HO20_all.columns))
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T2.index)
express['index'] = age_T2.index

eid = ukbb['eid']

amygdala_expression_df_20 = pd.DataFrame({'eid': eid.values,
     'Comp 1' : X_m[:, 0],
     'Comp 2' : X_m[:, 1],
     'Comp 3' : X_m[:, 2],
     'Comp 4' : X_m[:, 3],
o     'Comp 5' : X_m[:, 4],
     'Comp 6' : X_m[:, 5],
     'Comp 7' : X_m[:, 6],
     'Comp 8' : X_m[:, 7],
    })

amygdala_expression_df_20.to_csv('amygdala_expression_df_20.csv')


brain_expression_df_20 = pd.DataFrame({'eid': eid.values,
     'Comp 1' : Y_m[:, 0],
     'Comp 2' : Y_m[:, 1],
     'Comp 3' : Y_m[:, 2],
     'Comp 4' : Y_m[:, 3],
     'Comp 5' : Y_m[:, 4],
     'Comp 6' : Y_m[:, 5],
     'Comp 7' : Y_m[:, 6],
     'Comp 8' : Y_m[:, 7],
    })

brain_expression_df_20.to_csv('brain_expression_df_20.csv')
  

#PheWAS with the 1414 participants at the first time point

pls_phe_20_1414 = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe_20_1414.fit_transform(
    pd.DataFrame(FS_AM20_ss, columns=dfS20.columns), pd.DataFrame(FS_HO20_ss, columns=ukbb_HO20.columns))
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T2.index)
express['index'] = age_T2.index
express.set_index("index", inplace=True)

eid = ukbb['eid']
eid = eid.loc[subs_keep]

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


#MEDIAN RANKED CHANGE
# Both Sexes, 6 age groups MEDIAN
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

median_list_55 = []
median_list_60 = []
median_list_65 = []
median_list_70 = []
median_list_75 = []
median_list_81 = []
    
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
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Median Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Median Ranked Change 6 age ranges.xls' % (OUT_DIR_11))

# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'label': 'Median of the Difference in Gray Matter Volume','shrink': 0.25}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
plt.xlabel('Volume Loss <                                                                                                           > Volume Gain', fontsize=20)
plt.ylabel('Age Bracket', fontsize=20)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Median Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

#MEDIAN RANKED CHANGE
# Male 6 age groups MEDIAN
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

median_list_55 = []
median_list_60 = []
median_list_65 = []
median_list_70 = []
median_list_75 = []
median_list_81 = []
    
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_55 = []
    temp_list_60 = []
    temp_list_65 = []
    temp_list_70 = []
    temp_list_75 = []
    temp_list_81 = []
    
    for tuples in same_sex_age['male']:
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
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Median Male Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Median Male Ranked Change 6 age ranges.xls' % (OUT_DIR_11))

# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Median Male Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))

#MEDIAN RANKED CHANGE
# Female 6 age groups MEDIAN
# Filling up the lists of dataframes with their respective data
age_list = ['48-54', '55-59', '60-64', '65-69', '70-74', '75-81']

median_list_55 = []
median_list_60 = []
median_list_65 = []
median_list_70 = []
median_list_75 = []
median_list_81 = []
    
for idx,name in enumerate(dfAM30minus20.columns):
    temp_list_55 = []
    temp_list_60 = []
    temp_list_65 = []
    temp_list_70 = []
    temp_list_75 = []
    temp_list_81 = []
    
    for tuples in same_sex_age['female']:
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
    
age_axis = np.array(age_list)
   
f = plt.figure(figsize=(20, 20), dpi = 600)
X_comp_weights = X_AM_weights

# dfdata = pd.DataFrame(X_comp_weights, index=ROI_netnames, columns=ROI_netnames)
dfdata = pd.DataFrame(X_AM_weights, index=age_axis, columns = ['']*18)

dfdata.to_csv('%s/Median Female Ranked Change 6 age ranges.csv' % (OUT_DIR_11))
dfdata.to_excel('%s/Median Female Ranked Change 6 age ranges.xls' % (OUT_DIR_11))

# sns.set(font_scale=0.8)
# pyplot.figure(figsize=(15, 15))
ax = sns.heatmap(dfdata, annot = labels, fmt = '', annot_kws={"size": 14}, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.5}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize=14)
# plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
# sns.despine(top=True, right=True, left=False, bottom=False)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# ax = sns.heatmap(dfdata, cbar=True, linewidths=.5,
#   cmap=plt.cm.RdBu_r, center=0)
# plt.title('Network couplings with loneliness: %s\nPearson rho' % TAR_ANA)
# plt.colorbar()
plt.tight_layout()
# plt.yticks(rotation = 45)

plt.savefig('%s/Median Female Ranked Change 6 age ranges.pdf' % (OUT_DIR_11))
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.png' % (counter + 1, SUFFIX), DPI=600)
# plt.savefig('results_fMRI/pls_lonely_topcomp%i%s.pdf' % (counter + 1, SUFFIX))



#PheWAS with the 1414 participants at the first time point

pls_phe_20_1414 = PLSCanonical(n_components=n_comps)
X_m, Y_m  = pls_phe_20_1414.fit_transform(
    pd.DataFrame(FS_AM20_ss, columns=dfS20.columns), pd.DataFrame(FS_HO20_ss, columns=ukbb_HO20.columns))
express =  pd.DataFrame(columns=  ['index', 'x', 'y'], index = age_T2.index)
express['index'] = age_T2.index
express.set_index("index", inplace=True)

#Age/Sex
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

OUT_DIR_9 = 'AMsub2cortex/Age_Sex/Amygdala/First Time Point'

for counter,mode  in enumerate(range(n_comps)): 
    express['x'] = X_m[:,mode]
    express['y'] = Y_m[:,mode]
    
    median_xm = []
    median_xf = []
    
    lower_th_m = []
    lower_th_f = []
    upper_th_m = [] 
    upper_th_f = []
    
    temp_list_age = []
    
    for tuples in same_sex_age['male']:
            if len(tuples)>10:
                temp_list_x = []
                for index in tuples:
                    temp_list_x.append(express.loc[index,'x'])
                    temp_list_age.append(age_T2.loc[index,'21003-2.0'])
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
                             
    for tuples in same_sex_age['female']:
            if len(tuples)>10:
                temp_list_x = []
                for index in tuples:
                    temp_list_x.append(express.loc[index,'x'])
                    temp_list_age.append(age_T2.loc[index,'21003-2.0'])
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
    # plt.savefig('%s/Age_Sex_mode_%i.png' % (OUT_DIR_9, counter + 1), bbox_inches='tight')
    plt.show()


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
    # plt.savefig('%s/Age_Sex_mode_%i.png' % (OUT_DIR_10, counter + 1), bbox_inches='tight')
    plt.show()

#Number of significant components
from sklearn.cross_decomposition import PLSCanonical 
n_comps = 8

#Analysis of difference in brain volumes between the two time regions
#PLS Canonical
pls = PLSCanonical(n_components=n_comps)
pls.fit(dfAM20minus30, dfHO20minus30)
r2 = pls.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`

# #Canonical Correlation
# cca = CCA(n_components=n_comps)
# cca.fit(dfAM20minus30, dfHO20minus30)
# r2 = cca.score(dfAM20minus30, dfHO20minus30)  # coefficient of determination :math:`R^2`


from scipy.stats import pearsonr

est = pls
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
# inds_max_to_min = np.argsort(actual_Rs)[::-1]
# actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs)
# [0.19654273, 0.24062467, 0.21549389, 0.19554875, 0.21232852, 0.25243018, 0.21673239, 0.22932024]

n_keep = 8
n_permutations = 1000
cur_X = np.array(dfAM20minus30)
cur_Y = np.array(dfHO20minus30)
perm_rs = np.random.RandomState(0)
perm_Rs = []
perm_scores = []
n_except = 0
for i_iter in range(n_permutations):
    print(i_iter + 1)

    perm_rs.shuffle(cur_Y)

    # cur_X_perm = np.array([perm_rs.permutation(sub_entry) for sub_entry in cur_X])

    # same procedure, only with permuted subjects on the right side
    try:
        perm_cca = PLSCanonical(n_components=n_keep, scale=False)  # VERIFY

        # perm_inds = np.arange(len(Y_netmet))
        # perm_rs.shuffle(perm_inds)
        # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
        perm_cca.fit(cur_X, cur_Y)

        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        cur_score = perm_cca.score(cur_X, cur_Y)
        print(np.sort(perm_R)[::-1][:10])
        print(cur_score)
        perm_Rs.append(perm_R)
        perm_scores.append(cur_score)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))
perm_Rs = np.array(perm_Rs)

pvals = []
for i_coef in range(n_keep):  # COMP-WISE comparison to permutation results !!!
    cur_pval = (np.sum(perm_Rs[:, i_coef] > actual_Rs[i_coef])) / n_permutations
    pvals.append(cur_pval)
    # print cur_pval
    
# =========Change of gray matter volume between the two time points============ 
# Previously - [0.004 !!!, 0.016 !!!, 0.935, 0.672, 0.434, 0.351, 0.968, 0.9] 
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

#Number of significant components
n_keep = 8
n_permutations = 1000
cur_X = np.array(pd.DataFrame(FS_AM20_ss, columns=dfS20.columns))
cur_Y = np.array(pd.DataFrame(FS_HO20_ss, columns=ukbb_HO20.columns))
perm_rs = np.random.RandomState(0)
perm_Rs = []
perm_scores = []
n_except = 0
for i_iter in range(n_permutations):
    print(i_iter + 1)

    perm_rs.shuffle(cur_Y)

    # cur_X_perm = np.array([perm_rs.permutation(sub_entry) for sub_entry in cur_X])

    # same procedure, only with permuted subjects on the right side
    try:
        perm_cca = PLSCanonical(n_components=n_keep, scale=False)  # VERIFY

        # perm_inds = np.arange(len(Y_netmet))
        # perm_rs.shuffle(perm_inds)
        # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
        perm_cca.fit(cur_X, cur_Y)

        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        cur_score = perm_cca.score(cur_X, cur_Y)
        print(np.sort(perm_R)[::-1][:10])
        print(cur_score)
        perm_Rs.append(perm_R)
        perm_scores.append(cur_score)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))
perm_Rs = np.array(perm_Rs)

pvals = []
for i_coef in range(n_keep):  # COMP-WISE comparison to permutation results !!!
    cur_pval = (np.sum(perm_Rs[:, i_coef] > actual_Rs[i_coef])) / n_permutations
    pvals.append(cur_pval)
    # print cur_pval
    
# =============Gray matter volume  at the first time point=====================

# Previously - [0.03 !!!, 0.043 !!!, 0.965, 0.921, 0.776, 0.635, 0.997, 0.988]
# Now - [0.03!!, 0.001!!!, 0.151, 0.729, 0.358, 0.0!!!, 0.196, 0.041!!]
# 2 CCs are significant at p<0.05
# 2 CCs are significant at p<0.01
# =============================================================================


# pvals = np.array(pvals)[inds_max_to_min]
# print(pvals)
# print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
# print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
# print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))
# [0.023 !!! 0.042 !!! 0.14  0.043 0.251 0.207 0.555 0.506 0.777 0.212]
# 3 CCs are significant at p<0.05
# 0 CCs are significant at p<0.01