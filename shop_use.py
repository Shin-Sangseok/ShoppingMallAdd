import pandas as pd
import numpy as np
import score as score

import shop_dao_class1
from sklearn.model_selection import train_test_split
#from db import dao
#from 패키지 import 모듈명
#from 패키지명.모듈명 import 함수명 , 클래스명, *
#-->함수()

if __name__ == '__main__':

    #shop_dao_class1.py
    dao = shop_dao_class1.DAO()

    #recom_result= dao.recom_read()
    #recom_columns = ['product_idx', 'user_id', 'purchase_quantity', 'user_age']
    #df = pd.DataFrame(data = recom_result, columns=recom_columns)
    #print(df)

    member_result = dao.member_read()
    purchase_result = dao.purchase_read()

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

    train_mean = x_train.groupby(['user_id'])['purchase_quantity'].mean()
    print(dao.predict_quantity)
    score(dao.predict_quantity)


