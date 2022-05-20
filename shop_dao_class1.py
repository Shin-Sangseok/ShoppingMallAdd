import pymysql
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

class DAO:
    conn = None
    cur = None
    #mean_rating = 0

    def __init__(self):
        try:
            self.conn = pymysql.connect(
                host = 'localhost',
                port = 3366,
                user = 'root',
                password = '1234',
                db = 'shop',
                charset = 'utf8'
            )

            print('연결 성공', self.conn.host_info)

            self.cur = self.conn.cursor()
        except Exception as e:
            print('db연결 중 에러 발생!!')
            print('에러정보>>', e)

    def recom_read(self):
        sql = "select p.product_idx , p.user_id , p.purchase_quantity , m.user_age " \
              "from purchaseinfo p " \
              "inner join `member` m " \
              "on m.user_id  = p.user_id "

        result = self.cur.execute(sql)
        print('sql문 전송 결과>', result)

        #read인 경우 , 커서로 연결통로(스트림)에 검색결과를 꺼내주어야 한다
        row = self.cur.fetchall()
        print(row)
        self.conn.close()
        return row #검색결과 return

    #멤버테이블 user_id, user_age
    def member_read(self):
        sql = "select user_id, user_age from member "

        result = self.cur.execute(sql)
        print('sql문 전송 결과>', result)

        #read인 경우, 커서로 연결통로(스트림)에 검색결과를 꺼내주어야 한다.
        row = self.cur.fetchall()
        print(row)
     #   self.conn.close()
        return row

    #구매 테이블 user_id, product_idx, purchase_quantity

    def purchase_read(self):
        sql = "select user_id, product_idx, purchase_quantity from purchaseinfo"

        result = self.cur.execute(sql)
        print('sql문 전송 결과>', result)

        #read인 경우, 커서로 연결통로(스트림)에 검색결과를 꺼내주어야 한다
        row = self.cur.fetchall()
        print(row)
        self.conn.close()
        return row

    #정확도(RMSE)를 계산하는 함수
    def RMSE(self,y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

    #모델별 RMSE를 계산하는 함수
    #model은 함수 매개변수 들어가는 곳.
    def score(self,model):
        id_pairs = zip(x_test['user_id'], x_test['product_idx'])
        y_pred = np.array([model(user_id,product_idx) for (user_id,product_idx) in id_pairs])
        y_true = np.array(x_test['purchase_quantity'])
        return self.RMSE(y_true, y_pred)

    #평균으로 예측치 계산하는 모델
    def predict_qua(self,user_id,product_idx):
        try:
            purchase_quantity = train_mean[product_idx]
        except:
            purchase_quantity = 3
        return purchase_quantity

    #사용자와 다른 사용자 간의 유사도(user_similarity)
    def CF_sample(self,user_id,product_idx):
        if product_idx in rating_matrix:
            sim_scores = user_similarity[user_id].copy()
            product_quantity = rating_matrix[product_idx].copy()
            none_rating_idx = product_quantity[product_quantity.isnull()].index
            sim_scores = sim_scores.drop(none_rating_idx)
            mean_rating = np.dot(sim_scores, product_quantity)/sim_scores.sum()
        else:
            mean_rating = 3
        return mean_rating

    #


#from db import dao
#from 패키지 import 모듈명
#from 패키지명.모듈명 import 함수명 , 클래스명, *
#-->함수()

if __name__ == '__main__':

    d = DAO()
    #shop_dao_class1.py
    #dao = shop_dao_class1.DAO()

    #recom_result= dao.recom_read()
    #recom_columns = ['product_idx', 'user_id', 'purchase_quantity', 'user_age']
    #df = pd.DataFrame(data = recom_result, columns=recom_columns)
    #print(df)

    member_result = d.member_read()
    purchase_result = d.purchase_read()

    m_cols = ['user_id', 'user_age']
    p_cols = ['user_id', 'product_idx', 'purchase_quantity']

    #member dataFrame
    member_df = pd.DataFrame(data = member_result, columns=m_cols)

    #purchase_info dataFrame
    purchase_df = pd.DataFrame(data = purchase_result, columns=p_cols)
    print(member_df)
    print(purchase_df)

    x = purchase_df.copy()
    y = purchase_df['user_id']

    #각 user_id별 train set과 test set 비율 동일하게 유지
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

   # lr_reg = LinearRegression()


    train_mean = x_train.groupby(['user_id'])['purchase_quantity'].mean()

    #print(train_mean)
    #print(dao.predict_qua)

    rating_matrix = x_train.pivot(index='user_id', columns = 'product_idx', values = 'purchase_quantity')
    print(rating_matrix)

    #print(shop_dao_class1.DAO().predict_qua('1234','1'))
    #score(shop_dao_class1.DAO().predict_qua('1234','1'))
    #dao.RMSE(y_true, y_pred)

    matrix_dummy = rating_matrix.copy().fillna(0)
    print(matrix_dummy)
    user_similarity = cosine_similarity(matrix_dummy,matrix_dummy)
    user_similarity = pd.DataFrame(user_similarity, index = rating_matrix.index, columns= rating_matrix.index)
    print(user_similarity)

    print(d.CF_sample('1234','150'))

    print(d.score(d.CF_sample('1234','1')))
    #print(d.score(d.CF_sample()))















