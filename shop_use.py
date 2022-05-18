import pandas as pd

import shop_dao_class1

#from db import dao
#from 패키지 import 모듈명
#from 패키지명.모듈명 import 함수명 , 클래스명, *
#-->함수()

if __name__ == '__main__':
    dao = shop_dao_class1.DAO()
    recom_result= dao.recom_read()
    recom_columns = ['product_idx', 'user_id', 'purchase_quantity', 'user_age']
    df = pd.DataFrame(data = recom_result, columns=recom_columns)
    print(df)