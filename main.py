from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import os
import openpyxl
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

sample_data = pd.read_excel(r'C:\Users\pc\Documents\쇼핑몰_최종2.xlsx')

dt_clf = DecisionTreeClassifier(random_state=156)
params = {'max_depth': [3,6,8,10,12,14,16,18,20,22,24,26,28]}
le = LabelEncoder()

#font
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (14,4)
mpl.rcParams['axes.unicode_minus'] = False

#결측치 제거
def nullRemove(model):
    model.isnull()
    model = model.dropna(axis=0)
    print(model)
    return model

#Object형 수치화[라벨 인코딩]
def obj_numeric(data):
    data['평일휴일'] = le.fit_transform(data['평일휴일'])
    data['요일'] = le.fit_transform(data['요일'])
    data['시간대'] = le.fit_transform(data['시간대'])
    data['성별'] = le.fit_transform(data['성별'])
    data['연령대'] = le.fit_transform(data['연령대'])
    print('LabelEncoding 적용 후 ', data.head())
    return data

#train, test set 분리
def train_test_set(data):
    y= data['TAG']
    y= pd.DataFrame(y)
    X=pd.DataFrame(data, columns=['CRI_YM','평일휴일','요일','시간대','성별','연령대','건수합계','네이버 태그 클릭량'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=3)
    dt_clf.fit(X_train, y_train)
    print('X_train', X_train, 'y_train', y_train)
    return data, X_train, X_test, y_train, y_test

#결정 트리 정확도 구하는 함수

def model_acc(X_train,y_train, y_test):
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test,pred)
    print('결정 트리 예측 정확도:{0:.4f}'.format(accuracy))

#GridSearchCV

def decision_grid(dt_clf, params, X_train,y_train):
    grid_cv = GridSearchCV(dt_clf,param_grid=params, scoring='accuracy', cv = 5, verbose = 1)
    grid_cv.fit(X_train,y_train)
    print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))
    print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)
    # GridSearchCV 객체의 cv_results_ 속성을 DataFrame으로 생성
    cv_results_df = pd.DataFrame(grid_cv.cv_results_)

    # max_depth 파라미터 값과 그때의 테스트 세트, 학습 데이터 세트의 정확도 수치 추출
    cv_results_df[['param_max_depth', 'mean_test_score']]

    best_df_clf = grid_cv.best_estimator_
    pred1 = best_df_clf.predict(X_test)
    dt_acc = accuracy_score(y_test, pred1)
    print('결정 트리 예측 정확도:{0:.4f}'.format(dt_acc))


def pred_tag(input_data):
    # 예측 결과 출력
    result = dt_clf.predict_proba([input_data])
    if result[0][0] == 1:
        print('예측 카테고리는 생활/사무용품입니다.')
    elif result[0][1] == 1:
        print('예측 카테고리는 식료품입니다. ')
    elif result[0][2] == 1:
        print('예측 카테고리는 애완용품입니다.')
    elif result[0][3] == 1:
        print('예측 카테고리는 인테리어입니다.')
    elif result[0][4] == 1:
        print('예측 카테고리는 취미용품입니다.')
    elif result[0][5] == 1:
        print('예측 카테고리는 패션입니다.')
    elif result[0][6] == 1:
        print('예측 카테고리는 화장품입니다.')

sample_data = nullRemove(sample_data)
sample_data = obj_numeric(sample_data)
sample_data, X_train, X_test, y_train, y_test  = train_test_set(sample_data)
model_acc(X_train,y_train,y_test)
decision_grid(dt_clf,params,X_train,y_train)
print(X_train)

#user값 입력 및 예측
print('정수형 숫자 8개를 입력해주세요! ex)1 1 1 1 1 1 1 1')
num_list = list(map(int, input().split()))
pred_tag(num_list)

#user = np.array([num_list])

#[[0.0.0.0.1.0.0.]]
#대괄호 하나 없애는 법
#대괄호 변수이름[0]

