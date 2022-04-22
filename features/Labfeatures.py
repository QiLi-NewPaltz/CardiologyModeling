#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np


# In[2]:


import warnings

warnings.filterwarnings("ignore")

# # Labs

# ## BP

bp = pd.read_csv('Raw_BP_demo.csv')#, names=['Pat_ID','Days_from_first', 'Days_from_last', 'systolic_bp', 'diastolic_bp','MAP', 'CAD'])
bp=bp.set_index('Pat_ID')

# ## eGFR
# 
# - NAA(Non-AA vs AA (Africa American)
# - Stage 1: eGFR in normal range (greater than 90) with other signs of kidney damage, like protein in urine or physical damage to the kidneys
# - Stage 2: eGFR in normal range (60-89) with other signs of kidney damage, like protein in urine or physical damage to the kidneys
# - Stage 3: eGFR 30-59, moderate kidney damage
# - Stage 4: eGFR 15-29, severe kidney damage
# - Stage 5: eGFR less than 15, the kidneys are close to failure or have already failed


eGFR = pd.read_csv('Raw_eGFR_demo.csv')#, names=['Pat_ID','Days_from_first', 'Days_from_last', 'systolic_bp', 'diastolic_bp','MAP', 'CAD'])
eGFR = eGFR.set_index("PAT_PAT_ID")

eGFR["eGFR_gr_AA_Combined"] = eGFR['eGFR_gr_AA_num']
eGFR.loc[eGFR["eGFR_gr_AA_Combined"].isnull(), 'eGFR_gr_AA_Combined']=">60"
eGFR.loc[(eGFR["eGFR_gr_AA_txt"].isnull()) & (eGFR["eGFR_gr_AA_num"].isnull()), 'eGFR_gr_AA_Combined']=np.nan


eGFR["eGFR_gr_AA_Stage"] = "2"
eGFR.loc[eGFR["eGFR_gr_AA_num"]<=15, 'eGFR_gr_AA_Stage']="5"
eGFR.loc[(eGFR["eGFR_gr_AA_num"]>15) & (eGFR["eGFR_gr_AA_num"]<=30), 'eGFR_gr_AA_Stage']="4"
eGFR.loc[(eGFR["eGFR_gr_AA_num"]>30) & (eGFR["eGFR_gr_AA_num"]<=60), 'eGFR_gr_AA_Stage']="3"
eGFR.loc[(eGFR["eGFR_gr_AA_txt"].isnull()) & (eGFR["eGFR_gr_AA_num"].isnull()), 'eGFR_gr_AA_Stage']=np.nan

eGFR["eGFR_gr_NAA_Combined"] = eGFR['eGFR_gr_NAA_num']
eGFR.loc[eGFR["eGFR_gr_NAA_Combined"].isnull(), 'eGFR_gr_NAA_Combined']=">60"
eGFR.loc[(eGFR["eGFR_gr_NAA_txt"].isnull()) & (eGFR["eGFR_gr_NAA_num"].isnull()), 'eGFR_gr_NAA_Combined']=np.nan

eGFR["eGFR_gr_NAA_Stage"] = "2"
eGFR.loc[eGFR["eGFR_gr_NAA_num"]<=15, 'eGFR_gr_NAA_Stage']="5"
eGFR.loc[(eGFR["eGFR_gr_NAA_num"]>15) & (eGFR["eGFR_gr_NAA_num"]<=30), 'eGFR_gr_NAA_Stage']="4"
eGFR.loc[(eGFR["eGFR_gr_NAA_num"]>30) & (eGFR["eGFR_gr_NAA_num"]<=60), 'eGFR_gr_NAA_Stage']="3"
eGFR.loc[(eGFR["eGFR_gr_NAA_txt"].isnull()) & (eGFR["eGFR_gr_NAA_num"].isnull()), 'eGFR_gr_NAA_Stage']=np.nan

eGFR["eGFR_gr_NAA_Stage"].value_counts()

# ## HBA1c

HBA1c = pd.read_csv('Raw_HBA1c_demo.csv')#, names=['Pat_ID','Days_from_first', 'Days_from_last', 'systolic_bp', 'diastolic_bp','MAP', 'CAD'])
HBA1c = HBA1c.set_index('Pat_ID')

# ## Lipid

lipid = pd.read_csv('Raw_lipid_demo.csv')#, names=['Pat_ID','Days_from_first', 'Days_from_last', 'systolic_bp', 'diastolic_bp','MAP', 'CAD'])
lipid = lipid.set_index('Pat_ID')
lipid['HDL_Value'] = lipid['HDL_Value'].astype(float)
lipid.shape

# ## min

bpmin=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].min()
bpmin.columns = ['MAP_min','Systolic_BP_min','Diastolic_BP_min']

HBA1cmin = HBA1c.groupby(['Pat_ID'])['Results'].min()
HBA1cmin=HBA1cmin.to_frame()
HBA1cmin.columns=['HBA1c_min']

lipidmin=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].min()
lipidmin.columns=['LDL_C_min','Total_cholesterol_min','HDL_C_min','NON_HDL_C_min']


# ## max

bpmax=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].max()
bpmax.columns = ['MAP_max','Systolic_BP_max','Diastolic_BP_max']

HBA1cmax = HBA1c.groupby(['Pat_ID'])['Results'].max()
HBA1cmax = HBA1cmax.to_frame()
HBA1cmax.columns=['HBA1c_max']
HBA1cmax.head()


lipidmax=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].max()
lipidmax.columns=['LDL_C_max','Total_cholesterol_max','HDL_C_max','NON_HDL_C_max']

# ## Time Range


bptrange=bp.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].max()-bp.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].min()
bptrange.columns=['BP_Days_from_first_range','BP_Days_from_last_range']

HBA1ctrange=HBA1c.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].max()-HBA1c.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].min()
HBA1ctrange.columns=['A1c_Days_from_first_range','A1c_Days_from_last_range']

lipidtrange=lipid.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].max()-lipid.groupby(['Pat_ID'])['Days_from_first','Days_from_last'].min()
lipidtrange.columns=['Lipid_Days_from_first_range','Lipid_Days_from_last_range']


eGFRtrange=eGFR.groupby(['PAT_PAT_ID'])['Days_from_first','Days_from_last'].max()-eGFR.groupby(['PAT_PAT_ID'])['Days_from_first','Days_from_last'].min()
eGFRtrange.columns=['eGFR_Days_from_first_range','eGFR_Days_from_last_range']

# ## range

bpvrange=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].max()-bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].min()
bpvrange.columns = ['MAP_vrange','Systolic_BP_vrange','Diastolic_BP_vrange']

HBA1cvrange = HBA1c.groupby(['Pat_ID'])['Results'].max() - HBA1c.groupby(['Pat_ID'])['Results'].min()
HBA1cvrange = HBA1cvrange.to_frame()
HBA1cvrange.columns=['HBA1c_vrange']

eGFRvrange = eGFR.groupby(['PAT_PAT_ID'])['eGFR_gr_AA_num', 'eGFR_gr_NAA_num'].max() - eGFR.groupby(['PAT_PAT_ID'])['eGFR_gr_AA_num', 'eGFR_gr_NAA_num'].min()
#eGFRvrange = eGFRvrange.to_frame()
eGFRvrange.columns=['eGFR_gr_AA_vrange', 'eGFR_gr_NAA_vrange']

lipidvrange=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].max() - lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].min()
lipidvrange.columns=['LDL_c_vrange','TC_vrange','HDL_c_vrange','NON_HDL_c_vrange']

# ## Count

bpfreq=bp.groupby(['Pat_ID']).size()
bpfreq = bpfreq.to_frame()
bpfreq.columns=['BP_freq']

HBA1cfreq=HBA1c.groupby(['Pat_ID']).size()
HBA1cfreq = HBA1cfreq.to_frame()
HBA1cfreq.columns=['HBA1c_freq']


lipidfreq = lipid.groupby(['Pat_ID']).size()
lipidfreq = lipidfreq.to_frame()
lipidfreq.columns = ['lipid_freq']

eGFRfreq=eGFR.groupby(['PAT_PAT_ID']).size()
eGFRfreq = eGFRfreq.to_frame()
eGFRfreq.columns=['eGFR_freq']

# ## Mean

bpmean=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].mean()
bpmean.columns = ['MAP_mean','Systolic_BP_mean','Diastolic_BP_mean']

HBA1cmean = HBA1c.groupby(['Pat_ID'])['Results'].mean()
HBA1cmean = HBA1cmean.to_frame()
HBA1cmean.columns=['HBA1c_mean']
HBA1cmean.head()

#['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C']
lipidmean=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].mean()
lipidmean.columns=['LDL_c_mean','TC_mean','HDL_c_mean','NON-HDL_c_mean']
lipidmean.head()


# ## Standard Deviation

bpstd=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].std()
bpstd.columns = ['MAP_std','Systolic_BP_std','Diastolic_BP_std']
#bpfeatures = pd.concat([bpfeatures, bpf], axis=1, join='inner')

HBA1cstd = HBA1c.groupby(['Pat_ID'])['Results'].std()
HBA1cstd = HBA1cstd.to_frame()
HBA1cstd.columns=['HBA1c_std']

lipidstd=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].std()
lipidstd.columns=['LDL_C_std','TC_std','HDL_C_std','NON_HDL_C_std']


# ## Freq test / Avg Day for test

bpavgtestday = bp.groupby(['Pat_ID'])['Days_from_first'].max()/bp.groupby(['Pat_ID'])['Days_from_first'].count()
bpavgtestday = bpavgtestday.to_frame()
bpavgtestday.columns=['BP_Avg_Test_Day']
#bpfeatures = pd.concat([bpfeatures, bpfdf], axis=1, join='inner')

HBA1cavgtestday = HBA1c.groupby(['Pat_ID'])['Days_from_first'].max()/HBA1c.groupby(['Pat_ID'])['Days_from_first'].count()
HBA1cavgtestday = HBA1cavgtestday.to_frame()
HBA1cavgtestday.columns=['a1c_Avg_Test_Day']

lipidavgtestday = lipid.groupby(['Pat_ID'])['Days_from_first'].max()/lipid.groupby(['Pat_ID'])['Days_from_first'].count()
lipidavgtestday = lipidavgtestday.to_frame()
lipidavgtestday.columns=['lipid_Avg_Test_Day']

eGFRavgtestday = eGFR.groupby(['PAT_PAT_ID'])['Days_from_first'].max()/eGFR.groupby(['PAT_PAT_ID'])['Days_from_first'].count()
eGFRavgtestday = eGFRavgtestday.to_frame()
eGFRavgtestday.columns=['eGFR_Avg_Test_Day']

# ## coefficient of variation

bpcv=bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].std()/bp.groupby(['Pat_ID'])['MAP','systolic_bp','diastolic_bp'].mean()
bpcv.head()
bpcv.columns = ['MAP_cv','Systolic_BP_cv','Diastolic_BP_cv']
#bpfeatures = pd.concat([bpfeatures, bpf], axis=1, join='inner')

HBA1ccv=HBA1c.groupby(['Pat_ID'])['Results'].std()/HBA1c.groupby(['Pat_ID'])['Results'].mean()
HBA1ccv=HBA1ccv.to_frame()
HBA1ccv.columns=['HBA1c_cv']

lipidcv=lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].std()/lipid.groupby(['Pat_ID'])['LDL_C_Value','TC_Value','HDL_Value','NON_HDL_C'].mean()
lipidcv.columns=['LDL_c_cv','TC_cv','HDL_c_cv','NON_HDL_c_cv']

# ## Value rate
# - Stage 2/3/4/5 with the rate, #count/#total

eGFRNAAStageDummy = pd.get_dummies(eGFR['eGFR_gr_NAA_Stage'], prefix='eGFR_gr_NAA_Stage')
eGFRNAAStageDummy.head()
eGFRAAStageDummy = pd.get_dummies(eGFR['eGFR_gr_AA_Stage'], prefix='eGFR_gr_AA_Stage')
eGFRAAStageDummy.head()

eGFRAAStageRate=eGFRAAStageDummy.groupby(['PAT_PAT_ID'])['eGFR_gr_AA_Stage_2','eGFR_gr_AA_Stage_3','eGFR_gr_AA_Stage_4','eGFR_gr_AA_Stage_5'].mean()
eGFRAAStageRate.columns=['eGFR_gr_AA_Stage_2_VRATE','eGFR_gr_AA_Stage_3_VRATE','eGFR_gr_AA_Stage_4_VRATE','eGFR_gr_AA_Stage_5_VRATE']

eGFRNAAStageRate=eGFRNAAStageDummy.groupby(['PAT_PAT_ID'])['eGFR_gr_NAA_Stage_2','eGFR_gr_NAA_Stage_3','eGFR_gr_NAA_Stage_4','eGFR_gr_NAA_Stage_5'].mean()
eGFRNAAStageRate.columns=['eGFR_gr_NAA_Stage_2_VRATE','eGFR_gr_NAA_Stage_3_VRATE','eGFR_gr_NAA_Stage_4_VRATE','eGFR_gr_NAA_Stage_5_VRATE']

# ## number of reading raising

def NumReadingRaise (x):
    numincrease = 0
    numdecrease = 0
    base = x.iloc[0]
    for v in x:
        if base < v:
            #print (v)
            numincrease = numincrease + 1;
        if base > v:
            numdecrease = numdecrease + 1;
        base = v;

    return numincrease

bpf = bp.groupby(['pid'])['MAP'].apply(NumReadingRaise)
bpf.head()


bpg = bp.groupby(['pid'])


numincrease = 0
numdecrease = 0
base = bpgtest['MAP'].iloc[0]
for v in bpgtest["MAP"]:
    if base < v:
        print (v)
        numincrease = numincrease + 1;
    if base > v:
        numdecrease = numdecrease + 1;
    base = v;
    

# ## Combination


bpfeatures = pd.concat([bpmin,bpmax,bptrange,bpvrange,bpmean,bpstd,bpcv,bpfreq,bpavgtestday], axis=1, join='inner')

HBA1cfeatures = pd.concat([HBA1cmin,HBA1cmax,HBA1ctrange,HBA1cvrange,HBA1cmean,HBA1cstd,HBA1ccv,HBA1cfreq,HBA1cavgtestday], axis=1, join='inner')


lipidfeatures = pd.concat([lipidmin,lipidmax,lipidtrange,lipidvrange,lipidmean,lipidstd,lipidcv,lipidfreq,lipidavgtestday], axis=1, join='inner')


eGFRfeatures =  pd.concat([
    #lipidmin,lipidmax,eGFRmean,eGFRstd,eGFRcv,
    eGFRtrange,eGFRvrange,eGFRfreq,eGFRavgtestday,eGFRNAAStageRate,eGFRNAAStageRate], axis=1, join='inner')
eGFRfeatures.head()


features = pd.concat([bpfeatures,HBA1cfeatures,lipidfeatures,eGFRfeatures], axis=1, join='inner')

features.to_csv("data/features.demo.csv")



