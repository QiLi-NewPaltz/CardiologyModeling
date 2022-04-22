#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


# In[4]:


import pandas as pd
import numpy as np

df = pd.read_csv("data/deidentified_2020_04_29.csv", index_col="PAT_PAT_ID")

# # Features

# ## Features: Aggregate Risk Scores 

df['ASCVD_10_YR_SCORE_cat'] = 0
df.loc[df['ASCVD_10_YR_SCORE'].isnull(), 'ASCVD_10_YR_SCORE_cat'] = 3
df.loc[df['ASCVD_10_YR_SCORE']<5, 'ASCVD_10_YR_SCORE_cat'] = 1
df.loc[df['ASCVD_10_YR_SCORE']>=5, 'ASCVD_10_YR_SCORE_cat'] = 2

df['RisksCore_null']=0
df.loc[df['RisksCore'].isnull(),'RiskCore_null'] = 1



grhcc = pd.get_dummies(df['Gr_HCC'],prefix=['Gr_HCC'])
df['Gr_HCC_null'] = 0
df.loc[df['Gr_HCC'].isnull(),'Gr_HCC_null'] = 1

#RisksCore : HCC
aggriskscores = pd.concat([grhcc,df[['Gr_HCC_null','RisksCore','RisksCore_null','ASCVD_10_YR_SCORE',
                                    'ASCVD_10_YR_SCORE_cat','Dutch_LCN_Gr_1','Dutch_LCN_Gr4']]], axis=1)

# ## Features: Family History

df['FHX_Premature_null'] = 0
df.loc[df['FHX_Premature'].isnull(), 'FHX_Premature_null'] = 1

df['FHX_++_null'] = 0
df.loc[df['FHX_++'].isnull(), 'FHX_++_null'] = 1

df['FHX_None_Premature_null'] = 0
df.loc[df['FHX_None_Premature'].isnull(), 'FHX_None_Premature_null'] = 1

fhdummy = pd.get_dummies(df[[
    #'FHH_Confirmed','FHH_Confirmed_Combined',
    'FHX_Premature']])
fh = pd.concat([fhdummy, df[['FHX_++','FHX_None_Premature']], 
               # df['FHH_Confirmed_null'], df['FHH_Confirmed_Combined_null'], 
                df['FHX_Premature_null'], df['FHX_++_null'],
               df['FHX_None_Premature_null']], axis=1)
fh.shape

# ## Features: Demographics


df['agecat'] = 0
df.loc[(df['Age']>=30) & (df['Age']<40), 'agecat' ] = 1
df.loc[(df['Age']>=40) & (df['Age']<50), 'agecat' ] = 2
df.loc[(df['Age']>=50) & (df['Age']<55), 'agecat' ] = 3
df.loc[(df['Age']>=55) & (df['Age']<60), 'agecat' ] = 4
df.loc[(df['Age']>=60) & (df['Age']<65), 'agecat' ] = 5
df.loc[(df['Age']>=65) & (df['Age']<70), 'agecat' ] = 6
df.loc[(df['Age']>=70) & (df['Age']<75), 'agecat' ] = 7
df.loc[(df['Age']>=75) & (df['Age']<80), 'agecat' ] = 8
df.loc[(df['Age']>=80), 'agecat' ] = 9

df['gender'] = 0
df.loc[df['Gender']=='Female', 'gender'] = 1


df['Age'].describe()



#zip freq > 25
zipcounttop = zipcount[0:200] 
zipcounttop
zipfreqtop = zipcounttop.index
#zipfreqtop

df['hashed_zip_freq25'] = df['Hashed Value']
df.loc[~df['Hashed Value'].isin(zipfreqtop),'hashed_zip_freq25']=0

smoking = pd.get_dummies(df[['Current_Smoking_Status']])


demo = df[['gender','Age','agecat']]
           #,'BMI_Max_PreCAD','BMI_Avg_PreCAD','Current_Smoking_Status_Code' ]]



# ## Clinical Care group
# - 244-SEP_Affiliation
# - 14-Carrier 
# - 286-MyChart
# - 254-Months_Since_LOV_Comp_PCP
# - 434-Saw_Endo


clinalcaredummy = pd.get_dummies(df[['Carrier','SEP_Affiliation','Saw_Endo','MyCHart']])
clinicalcare = pd.concat([clinalcaredummy,df['Months_Since_LOV_Comp_PCP']], axis=1)


# ## Features: Labs

# - 131-LDL>190_YN
# - 132-LDL>190_x2_YN
# - 157-LDL_NUM_Before_CAD_AVG
# - 168-NonHDL_Num_Before_CAD_Avg
# - 169-NonHDL_NonLDL_C_Num_Before_CAD_Avg
# - 191-MAX_LPA ((ONLY When > 29) OR continuous variable
# - MAX_LPA > 50 , >29
# - Dutch_LCN_Gr4
# - MAX_LDL_C_


df['MAX_LPA_cat'] = 2 
df.loc[df['MAX_LPA']<29, 'MAX_LPA_cat'] = 1
df.loc[df['MAX_LPA']>50, 'MAX_LPA_cat'] = 3
df.loc[df['MAX_LPA'].isnull(), 'MAX_LPA_cat'] = 0


df['MAX_LDL_C_null'] = 0
df.loc[df['MAX_LDL_C_'].isnull(),'MAX_LDL_C_null'] = 1


df['LDL_Num_Before_CAD_Avg_null'] = 0
df.loc[df['LDL_Num_Before_CAD_Avg'].isnull(),'LDL_Num_Before_CAD_Avg_null'] = 1


df['NonHDL_NonLDL_C_Num_Before_CAD_Avg_null'] = 0
df.loc[df['NonHDL_NonLDL_C_Num_Before_CAD_Avg'].isnull(),'NonHDL_NonLDL_C_Num_Before_CAD_Avg_null'] = 1


labs = df[[
    #'Actual_LDL>190_x2_YN','Actual_LDL>190_YN',
    'LDL>190_x2_YN','LDL>190_YN',
           'LDL_Num_Before_CAD_Avg','NonHDL_Num_Before_CAD_Avg',
            'NonHDL_NonLDL_C_Num_Before_CAD_Avg','MAX_LPA','MAX_LPA_cat','MAX_LDL_C_null',
          'LDL_Num_Before_CAD_Avg_null', 'NonHDL_NonLDL_C_Num_Before_CAD_Avg_null']]


