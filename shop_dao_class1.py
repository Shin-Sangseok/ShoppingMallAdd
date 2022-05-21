import pymysql
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

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
        #model--> 모델 함수 대입.

        def score(self,model):
            #zip()함수는 여러 개의 iterable 객체를 인자로 받고
            # 각 객체가 담고 있는 원소를 튜플 형태로 반환.
            id_pairs = zip(x_test['user_id'], x_test['product_idx'])
            y_pred = np.array([model(user_id,product_idx) for (user_id,product_idx) in id_pairs])
            y_true = np.array(x_test['purchase_quantity'])
            return self.RMSE(y_true, y_pred)


        def score2(self,model, neighbor_size = 0):
            #zip()함수는 여러 개의 iterable 객체를 인자로 받고
            # 각 객체가 담고 있는 원소를 튜플 형태로 반환.
            id_pairs = zip(x_test['user_id'], x_test['product_idx'])
            y_pred = np.array([model(user_id,product_idx, neighbor_size) for (user_id,product_idx) in id_pairs])
            y_true = np.array(x_test['purchase_quantity'])
            return self.RMSE(y_true, y_pred)

        #RMSE함수로 리턴

        #평균으로 예측치 계산하는 모델
        def predict_qua(self,user_id,product_idx):
            try:
                purchase_quantity = train_mean[product_idx]
            except:
                purchase_quantity = 3
            return purchase_quantity

        #사용자와 다른 사용자 간의 유사도(user_similarity)
        def cf_sample(self,user_id,product_idx):
            if product_idx in rating_matrix:
                sim_scores = user_similarity[user_id].copy()
                product_quantity = rating_matrix[product_idx].copy()
                none_rating_idx = product_quantity[product_quantity.isnull()].index
                sim_scores = sim_scores.drop(none_rating_idx)
                mean_rating = np.dot(sim_scores, product_quantity)/sim_scores.sum()
            else:
                mean_rating = 3
            return mean_rating

        #KNN(Neighbor Size를 정해서 예측치 계산하는 함수)
        def cf_knn(self,user_id, product_idx, neighbor_size = 0):
                if product_idx in rating_matrix:
                    sim_scores = user_similarity[user_id].copy()
                    product_ratings = rating_matrix[product_idx].copy()
                    none_rating_idx = product_ratings[product_ratings.isnull()].index
                    product_ratings = product_ratings.drop(none_rating_idx)
                    sim_scores = sim_scores.drop(none_rating_idx)

                    if neighbor_size == 0:
                       mean_rating = np.dot(sim_scores, product_ratings)/ sim_scores.sum()

                    #neighbor_size 지정
                    else:
                        if len(sim_scores)>1: #구매자 수가 최소 2명 이상인 경우 진행
                            neighbor_size = min(neighbor_size, len(sim_scores)) #해당 상품을 구매한 사람 수 중에서 작은 것을 크기로 설정
                                                                            #유사도 계산이 가능한 이웃 수가 neighbor_size보다 작을수 있기 때문
                            sim_scores = np.array(sim_scores)
                            #np.argsort() 원본 행렬이 정렬되었을 때, 원본 행렬의 원소에 대한
                            #인덱스를 필요로 할때 np.argsort()를 이용
                            #내림차순을 얻고 싶을 때는 [::-1]을 붙여준다.
                            product_ratings = np.array(product_ratings)
                            user_idx = np.argsort(sim_scores)
                            sim_scores = sim_scores[user_idx][-neighbor_size] #가장 유사도가 높은 사용자 k명의 사용자 선정
                            product_ratings = product_ratings[user_idx][-neighbor_size]
                            #numpy.dot() 두 어레이의 내적(Dot product)를 계산
                            # numpy.dot(a,b)
                            #a와 b가 모두 0차원이면 곱 연산
                            #a와 b가 모두 1차원이면 두 벡터의 내적
                            #a와 b가 모두 2차원이면 행렬곱
                            mean_rating = np.dot(sim_scores, product_ratings)/sim_scores.sum()
                        else:
                            mean_rating = 3.0
                else:
                  mean_rating = 3.0
                return mean_rating

        def recom_product(self,user_id, product_idx, neighbor_size = 30):
            user_product= rating_matrix.loc[user_id].copy()
            for product in rating_matrix:
                #값이 null이 아닌 경우 - 상품을 이미 산 경우 이기 떄문에 제외
                if pd.notnull(user_product.loc[product]):
                    user_product.loc[product] = 0
                #null인 경우 - ck_knn()함수 호출해서 구매자의 예상 개수 예상
                else:
                    user_product.loc[product] = self.cf_knn(user_id, product_idx, neighbor_size)
            product_sort = user_product.sort_values(ascending=False)[:product_idx]
            recom_products = purchase_df.loc[product_sort.index]
            recommendations = recom_products['product_idx']
            return recommendations











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

        #print(d.CF_sample('1234','150'))
        #print(d.score(d.CF_sample('1234','1')))
        print('cf_sample의 score:',d.score(d.cf_sample))
        print('cf_knn의 score', d.score2(d.cf_knn,neighbor_size=30))

        print('---------------------------------------------------------------------------------------------------------')

        #columns는 선택 적용 관심을 가지는 values를 추가로 구분하기 위해 선택하는 옵션
        #전체 데이터 full matrix , cosine similarity
        product_matrix= purchase_df.pivot_table(values = 'purchase_quantity', index = 'user_id',columns='product_idx')
        #print('product_matrix', product_matrix)

        #코사인 유사도란?
        #벡터와 벡터간의 유사도를 비교할 때 두 벡터 간의 사잇값을 구해서 얼마나 유사한지 수치로 나타낸 것
        #벡터 방향이 비슷할수록 두 벡터는 서로 유사하며, 벡터 방향이 90도 일 때는
        #두 벡터간의 관련성이 없으며, 벡터 방향이 반대가 될수록 두 벡터는 반대 관계

        print('-----------------------------------------------------------------------')
        matrix_dummy = product_matrix.copy().fillna(0)
        user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
        #print('user_similairty', user_similarity)
        #user_similarity = pd.DataFrame(user_similarity, index = cosine_similarity.index, columns=product_matrix.index)\

        print(d.recom_product(user_id='1234', product_idx=1, neighbor_size=20))


























