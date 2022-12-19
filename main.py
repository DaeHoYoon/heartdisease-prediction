#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from dbmodule import uloaddb, dloaddb
from trainmodule import trainmodel, Roccurve, Learn_curve
from sqlalchemy import create_engine
from dataprep.eda import create_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
# %%
filepath = r'.\data\chapter25_heart_disease.csv'
id = 'root'
pw = 276018
host = 'localhost'
dbname = 'heartdisease'
tbname = 'heartdiseasetbl'
uloaddb(filepath, id, pw, host, dbname, tbname)
# %%
df = dloaddb(host, id, pw, dbname, tbname)
# %%
report = create_report(df)
report.save('heartdisease.html')
# %%
print('데이터프레임 shape:', df.shape)
print('데이터프레임 info:', df.info())
# %%
# 컬럼명 소문자로 변환, 타겟값 컬럼명 target으로 이름 변경
col_list = df.columns
df.columns = col_list.str.lower()
df.rename(columns = {'heartdiseaseorattack' : 'target'}, inplace=True)
#%%
# 타겟값 빈도 퍼센테이지 확인
sns.countplot(data=df, x='target')
print('target distribution \n', df['target'].value_counts()/len(df)) 
#%%
df1 = df.copy()

cols = list(df1) # 컬럼 추출
cols.remove('bmi') # 연속형 컬럼 제외
cols

df1[cols] = df1[cols].astype('int').astype('category') # 범주형 컬럼을 카테고리로 변환
#%%

# cols = ['highbp', 'highchol', 'cholcheck','smoker', 'stroke', 'physactivity', 'fruits', 'veggies','hvyalcoholconsump','anyhealthcare', 'nodocbccost', 'diffwalk', 'sex']
# for col in cols:
#     df1[col] = df1[col].replace(0.0, '0').replace(1.0,'yes').astype('category')

# df1['sex'] = df1['sex'].replace('no', 'woman').replace('yes', 'man')
# df1
#%%
nrow = 7
ncol = 3
fig = plt.figure(figsize=(10,20))
for idx, col in enumerate(cols):
    plt.subplot(nrow, ncol, idx+1)
    sns.histplot(data=df1, x=col, y='target')

sns.histplot(data=df1, x=df1['bmi'], y = 'target')
#%%
## 상관관계 확인
sns.set(style='white')
cor = df.corr() # 상관계수
mask = np.zeros_like(cor, dtype=np.bool_) # 상관계수 배열과 같은 형태의 zero numpy생성/ 0을 bool처리해서 F로 나옴
mask[np.triu_indices_from(mask,1)] = True # 삼각함수 인덱스를 True로 변환 
fig = plt.figure(figsize=(20,20))
sns.heatmap(cor, annot=True, mask=mask, vmin=-1, vmax=1) # annot = 박스 안에 글자쓰는 파라미터
plt.title('heart disease correlation', size = 30)

#%%
# 스케일링
df2 = df.copy()

mmscale = MinMaxScaler()
col = ['bmi']
df2['bmi'] = mmscale.fit_transform(df[col])
#%%
# 데이터 분리

X_data = df2.iloc[:,1:]
y_target = df2['target']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=0)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#%%
# 데이터 학습 및 평가
xgb_clf = xgboost.XGBClassifier()
rf_clf = RandomForestClassifier()
rg_reg = LogisticRegression()

models = [xgb_clf, rf_clf, rg_reg]
for model in models:
    Roccurve(model, X_train, y_train, X_test, y_test)
    Learn_curve(model, X_train, y_train)
#%%
skf = StratifiedKFold(n_splits=40, shuffle=False, random_state=None)
for trainidx, testidx in skf.split(X_data, y_target):
    X_train, X_test = X_data.iloc[trainidx], X_data.iloc[testidx]
    y_train, y_test = y_target.iloc[trainidx], y_target.iloc[testidx]

#%%
tsne = TSNE(n_components=2)
tsne_df= tsne.fit_transform(X_test)
tsne_df
# %%
plt.scatter(tsne_df[:,0], tsne_df[:,1], c=(y_test==0))
plt.scatter(tsne_df[:,0], tsne_df[:,1], c=(y_test==1))
# %%
### 데이터 균형 맞춘 후 다시 돌려보기
# smote하기 위한 데이터 전처리
cols = list(X_data)
cols.remove('bmi')
X_data[cols] = X_data[cols].astype('int')
X_data
# %%
smote = SMOTE() # 오버샘플링 (x와 y의 갯수를 맞춤)

X_res, y_res = smote.fit_resample(X_data, y_target)
print(y_res.value_counts())
X_res['highbp'].value_counts()
# %%
X_trains, X_tests, y_trains, y_tests = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
# %%
models = [xgb_clf, rf_clf, rg_reg]
for model in models:
    Roccurve(model, X_trains, y_trains, X_tests, y_tests)
    Learn_curve(model, X_trains, y_trains)
# %%
skf = StratifiedKFold(n_splits=40, shuffle=False, random_state=None)
for trainidx, testidx in skf.split(X_res, y_res):
    X_trains, X_tests = X_res.iloc[trainidx], X_res.iloc[testidx]
    y_trains, y_tests = y_res.iloc[trainidx], y_res.iloc[testidx]

#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(X_tests)
tdf
#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_tests==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_tests==1))
# %%
### 언더샘플링
df3 = df.copy()
df3 = df3.sample(frac=1) # 데이터를 랜덤으로 섞는다

dis_df = df3[df3.target == 1.0]
nodis_df = df3[df3.target == 0.0][:23893] # y가 1인 값의 갯수와 동일한 숫자만큼 0인 값을 맞춘다

print("타겟값이 1인 데이터프레임 shape:", dis_df.shape)
print("타겟값이 0인 데이터프레임 shape:", nodis_df.shape)
# %%
under_df = pd.concat([dis_df, nodis_df], axis = 0).sample(frac=1) # 두 데이터프레임을 합치고 랜덤으로 섞는다.
under_df.shape
# %%
under_X = under_df.drop(['target'], axis=1)
under_y = under_df['target']

X_trainu, X_testu, y_trainu, y_testu = train_test_split(under_X, under_y, test_size=0.2, random_state=0)

models = [xgb_clf, rf_clf, rg_reg]
for model in models:
    Roccurve(model, X_trainu, y_trainu, X_testu, y_testu)
    Learn_curve(model, X_trainu, y_trainu)

# %%
tsne = TSNE(n_components=2)
utdf = tsne.fit_transform(under_X)
utdf
#%%
plt.scatter(utdf[:,0], utdf[:,1], c=(under_y==0))
plt.scatter(utdf[:,0], utdf[:,1], c=(under_y==1))
# %%
