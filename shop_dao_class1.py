import pymysql

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

