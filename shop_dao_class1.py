import pymysql
import shop_use
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DAO:
    conn = None
    cur = None

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

 #   def recom_read(self):
 #       sql = "select p.product_idx , p.user_id , p.purchase_quantity , m.user_age " \
 #             "from purchaseinfo p " \
 #             "inner join `member` m " \
 #             "on m.user_id  = p.user_id "

 #       result = self.cur.execute(sql)
 #       print('sql문 전송 결과>', result)

 #       #read인 경우 , 커서로 연결통로(스트림)에 검색결과를 꺼내주어야 한다
 #       row = self.cur.fetchall()
 #       print(row)
 #       self.conn.close()
 #       return row #검색결과 return

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
    def score(self,model):
        id_pairs = zip(shop_use.x_test['user_id'], shop_use.x_test['product_idx'])
        y_pred = np.array([model(user,movie) for (user,movie) in id_pairs])
        y_true = np.array(shop_use.x_test['purchase_quantity'])
        return self.RMSE(y_true, y_pred)

    #평균으로 예측치 계산하는 모델
    def predict_quantity(self,user_id,purchase_quantity):
        try:
            purchase_quantity = shop_use.train_mean[purchase_quantity]
        except:
            purchase_quantity = 3
        return purchase_quantity








