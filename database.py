# -*- coding:utf-8 -*-
from settings import cnx
from settings import db_cursor as cursor


class GinQuery(object):
    @staticmethod
    def create_similarity_matrix():
        query = ("""
            CREATE 
        """)

    @staticmethod
    def get_pk_count():
        query = ("""
            SELECT count(*)
            FROM GIN.apt_detail
        """)
        cursor.execute(query, params=())
        return cursor

    @staticmethod
    def get_pk_list(size):
        query = ("""
            SELECT idx 
            FROM GIN.apt_detail 
            order by idx LIMIT %s;
        """)
        cursor.execute(query, params=(size, ))
        return cursor

    @staticmethod
    def get_similarity(pk):
        query = ("""
            SELECT similarity
            FROM apt_similarity
            WHERE pk_apt_detail=%s
        """)
        cursor.execute(query, params=(pk, ))
        return cursor

    @staticmethod
    def get_trade_price(pk):
        query = ("""
            SELECT pk_apt_detail, year, mon, real_day, floor, t_amt
            FROM apt_trade
            WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
            ORDER BY deal_ymd;""")

        cursor.execute(query, params=(pk, ))
        return cursor

    @staticmethod
    def get_trade_price_with_date(pk, date):
        query = """
            SELECT deal_ymd, t_amt
            FROM apt_trade
            WHERE EXCEPT_YN='n'
            AND pk_apt_detail=%s 
            AND deal_ymd IN (%s)
            ORDER BY deal_ymd;""" % (pk, date, )

        cursor.execute(query)
        return cursor


    @staticmethod
    def get_trade_price_calc(pk, time, year, mon, real_day):
        query = ("""
            SELECT pk_apt_detail, year, mon, real_day, floor, t_amt
            FROM apt_trade
            WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
            AND year=%s
            AND mon=%s
            AND real_day=%s
            ORDER BY deal_ymd;""")

        cursor.execute(query, params=(pk, ))
        return cursor

    @staticmethod
    def get_naver_sale_price_with_floor(pk, time, floor):
        query = """                
            SELECT c.pk_apt_detail, d.reg_date, d.floor, d.price
            FROM apt_detail a
             INNER JOIN naver_apt_sale_detail_group c
              ON a.idx = c.pk_apt_detail
             INNER JOIN naver_apt_sale d
              ON c.pk_naver_apt_master = d.idx
             AND c.supply_extent = d.supply_extent
             AND c.private_extent = d.extent
            WHERE c.pk_apt_detail = %s
             AND d.reg_date IN (%s)
             AND d.floor = %s
             AND d.use_yn = 'y'
             AND d.trade_cd IN ('t')
           ORDER BY reg_date;
        """ % (pk, time, floor)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_naver_sale_price_for_trend(pk, time):
        query = """
         SELECT c.pk_apt_detail, d.reg_date, d.floor, d.price
          FROM apt_detail a
         INNER JOIN naver_apt_sale_detail_group c
            ON a.idx = c.pk_apt_detail
         INNER JOIN naver_apt_sale d
            ON c.pk_naver_apt_master = d.idx
           AND c.supply_extent = d.supply_extent
           AND c.private_extent = d.extent
         WHERE c.pk_apt_detail = %s
           AND d.reg_date IN (%s)
           AND d.use_yn = 'y'
           AND d.trade_cd IN ('t')
         ORDER BY reg_date
        ;""" % (pk, time)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_naver_sale_price_all(pk):
        query = """
         SELECT d.price
          FROM apt_detail a
         INNER JOIN naver_apt_sale_detail_group c
            ON a.idx = c.pk_apt_detail
         INNER JOIN naver_apt_sale d
            ON c.pk_naver_apt_master = d.idx
           AND c.supply_extent = d.supply_extent
           AND c.private_extent = d.extent
         WHERE c.pk_apt_detail = %s
           AND d.use_yn = 'y'
           AND d.trade_cd IN ('t')
         ORDER BY reg_date
        ;""" % pk
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_trade_price_all(pk):
        query = """
        SELECT * FROM GIN.apt_trade
          WHERE 
        pk_apt_detail=%s
        AND except_yn='n';
        """

    @staticmethod
    def get_actual_price(idx):
        # 실거래가 매매
        query = ("""
            SELECT a.master_idx AS pk_apt_master,b.pk_apt_detail,a.supply_extent, a.extent,b.t_amt,b.floor,b.deal_ymd, 
               b.day, b.real_day
            FROM apt_detail a
            INNER JOIN apt_trade b
                ON a.idx = b.pk_apt_detail
            WHERE b.pk_apt_detail = %s 
        """)

        cursor.execute(query, params=(idx, ))
        return cursor

    @staticmethod
    def get_apt_simple_detail(idx):
        query = ("""            
            SELECT a.idx AS pk_apt_master       # 지인 master 아파트키
                 , b.idx AS pk_apt_detail       # 지인 detail 아파트 key
                 , a.edit_bldg_nm AS bldg_nm    # 단지명
                 , a.location_site              # 대지위치
                 , a.search_dt                  # 준공일
                 , a.latlngx                    # 위도
                 , a.latlngy                    # 경도
              FROM apt_master a
                   INNER JOIN apt_detail b
                   ON a.idx = b.master_idx
                   INNER JOIN mapping_master c
                   ON a.idx = c.idx_apt_master
             WHERE b.idx = %s
               AND a.bldg_cd IN (1, 2, 4, 5)                                                         #사용하는 건물 타입
        """)
        cursor.execute(query, params=(idx, ))
        return cursor

    @staticmethod
    def get_apt_detail(idx):
        query = ("""            
            SELECT a.idx AS pk_apt_master       # 지인 master 아파트키
                 , a.edit_bldg_nm               # 단지명
                 , a.sigungu_cd                 # 시군구 코드            
                 , a.dong_cd                    # 법정동 코드
                 , a.location_site              # 대지위치
                 , a.apt_dong_nm                # 동정보
                 , a.total_num_of_family        # 총 세대수
                 , a.total_dong_cnt             # 총 동수
                 , a.max_jisang_floor           # 최고층
                 , a.total_jucha                # 총 주차대수
                 , a.search_dt                  # 준공일
                 , a.latlngx                    # 위도
                 , a.latlngy                    # 경도
                 , b.idx AS pk_apt_detail       # 지인 detail 아파트 key 
                 , b.supply_extent              # 공급면적
                 , b.extent                     # 전용면적
                 , b.num_of_family              # 면적별 세대수
                 , b.apt_dong                   # 읍면동 이름
                 , b.price                      # 분양가격
              FROM apt_master a
                   INNER JOIN apt_detail b
                   ON a.idx = b.master_idx
                   INNER JOIN mapping_master c
                   ON a.idx = c.idx_apt_master
             WHERE b.idx = %s
               AND a.bldg_cd IN (1, 2, 4, 5)                                                         #사용하는 건물 타입
        """)
        cursor.execute(query, params=(idx, ))
        return cursor

    @staticmethod
    def create_similarity():
        cursor.execute("""
            DROP TABLE IF EXISTS apt_similarity
        """, params=())

        cursor.execute("""
            CREATE TABLE apt_similarity (
                idx int(11) not null auto_increment,
                pk_apt_detail int(11) not null ,
                similarity blob not null,
                primary key (idx, pk_apt_detail)
            );
        """, params=())
        return cursor

    @staticmethod
    def insert_similarity(pk_apt_detail, similarity):
        cursor.execute("""
            INSERT INTO apt_similarity (pk_apt_detail, similarity)
            VALUES (%s, %s)
        """, params=(pk_apt_detail, similarity))
        cnx.commit()
        return cursor

    @staticmethod
    def select_similarity(pk_apt_detail):
        cursor.execute("""
            SELECT a_s.pk_apt_detail
             , a_d.bldg_nm
             , a_d.sigungu_cd
             , a_d.dong_cd
             , a_m.latlngx
             , a_m.latlngy
             , a_m.gen_dt
             , a_s.similarity
        FROM apt_similarity AS a_s
            INNER JOIN apt_detail AS a_d
            ON a_s.pk_apt_detail = a_d.idx
            INNER JOIN apt_master as a_m
            ON a_d.master_idx = a_m.idx
        WHERE a_s.pk_apt_detail = %s
        """, params=(pk_apt_detail, ))
        return cursor

    @staticmethod
    def select_similarity_i(idx):
        cursor.execute("""
        SELECT a_s.pk_apt_detail
            , a_m.edit_bldg_nm as m_bldg_nm
            , a_d.bldg_nm
            , a_d.sigungu_cd
            , a_d.dong_cd
            , a_m.latlngx
            , a_m.latlngy
            , a_m.gen_dt
            , a_s.similarity
        FROM apt_similarity AS a_s
            INNER JOIN apt_detail AS a_d
            ON a_s.pk_apt_detail = a_d.idx
            INNER JOIN apt_master as a_m
            ON a_d.master_idx = a_m.idx
        WHERE a_s.idx = %s
        """, params=(idx, ))
        return cursor

    @staticmethod
    def select_t_amt(pk_detail):
        cursor.execute("""
            SELECT idx as master_idx
                 , deal_ymd as year
                 , totcnt
                 , floor
                 , t_amt
            FROM apt_trade
            WHERE pk_apt_detail = %s;
        """, params=(pk_detail, ))
        return cursor

    @staticmethod
    def test2():
        query = ("""
            SELECT a.idx AS pk_apt_master       # 지인 master 아파트키
                , a.edit_bldg_nm               # 단지명
                , a.sigungu_cd                 # 시군구 코드            
                , a.dong_cd                    # 법정동 코드
                , a.location_site              # 대지위치
                , a.apt_dong_nm                # 동정보
                , a.total_num_of_family        # 총 세대수
                , a.total_dong_cnt             # 총 동수
                , a.max_jisang_floor           # 최고층
                , a.total_jucha                # 총 주차대수
                , a.search_dt                  # 준공일
                , a.latlngx                    # 위도
                , a.latlngy                    # 경도
                , b.idx AS pk_apt_detail       # 지인 detail 아파트 key 
                , b.supply_extent              # 공급면적
                , b.extent                     # 전용면적
                , b.num_of_family              # 면적별 세대수
                , b.apt_dong                   # 읍면동 이름
                , b.price                      # 분양가격
            FROM apt_master a
                INNER JOIN apt_detail b
                ON a.idx = b.master_idx
                INNER JOIN mapping_master c
                ON a.idx = c.idx_apt_master
            WHERE a.bldg_cd IN (1, 2, 4, 5)                                                         #사용하는 건물 타입
        """)
        cursor.execute(query, params=())
        return cursor

