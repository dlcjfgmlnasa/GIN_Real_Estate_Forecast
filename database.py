# -*- coding:utf-8 -*-
from settings import cnx
from settings import db_cursor as cursor


class GinAptQuery(object):
    @staticmethod
    def get_predicate_apt_list():
        # 예측할 아파트 단지 리스트 출력
        query = ("""
             SELECT a.idx, b.idx
              FROM apt_master a 
             INNER JOIN apt_detail b 
                ON a.idx = b.master_idx
             WHERE a.total_num_of_family > 99
               AND a.bldg_cd IN (1, 6)
               AND b.NUM_OF_FAMILY > 29
             ORDER BY a.idx, b.idx
               ;
        """)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_predicate_apt_list_with_extent():
        # 예측할 아파트 단지 리스트 출력
        query = ("""
             SELECT a.idx, b.idx, b.extent
              FROM apt_master a 
             INNER JOIN apt_detail b 
                ON a.idx = b.master_idx
             WHERE a.total_num_of_family > 99
               AND a.bldg_cd IN (1, 6)
               AND b.NUM_OF_FAMILY > 29
             ORDER BY a.idx, b.idx
               ;
        """)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_apt_master_list():
        # 아파트 단지 리스트 출력
        query = ("""
            SELECT idx
            FROM apt_master
            WHERE apt_master.bldg_cd=1 AND
                  apt_master.total_num_of_family>=100;
        """)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_apt_detail_list(size):
        # 우리가 예측해야되는 아파트 리스트를 출력해주는 쿼리
        # apt_detail_pk, apt_master_pk 를 반환
        query = ("""
            SELECT  d.master_idx, d.idx
            FROM apt_detail d
            INNER jOIN apt_master m
              ON d.master_idx = m.idx
            WHERE m.bldg_cd=1 AND
                  m.total_num_of_family>=100
            LIMIT %s;
        """)
        cursor.execute(query, params=(size, ))
        return cursor

    @staticmethod
    def get_extent(apt_detail_pk_list: str):
        # 아파트 면적 출력
        query = """
            SELECT idx, extent
            FROM apt_detail
            WHERE idx in (%s)
        """ % (apt_detail_pk_list, )
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_max_floor(apt_detail_pk: int):
        query = ("""
            SELECT b.max_jisang_floor
            FROM apt_detail a
            INNER JOIN apt_master b
              ON a.master_idx = b.idx
            WHERE a.idx = %s
        """)
        cursor.execute(query, params=(apt_detail_pk, ))
        return cursor

    @staticmethod
    def get_trade_price_with_sql_date(apt_detail_pk: int, trade_cd: str,
                                      date_sql: str):
        # 아파트의 매매, 전세/월세, 가격
        if trade_cd == 't':
            # 아파트 매매가격 리스트 출력
            query = """
                SELECT apt_detail.master_idx, pk_apt_detail,  year, mon, day, floor, apt_detail.extent, t_amt
                FROM apt_trade
                INNER JOIN apt_detail
                  ON apt_detail.idx = apt_trade.pk_apt_detail
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                AND (%s) 
                ORDER BY deal_ymd;"""
            query = query % (apt_detail_pk, date_sql)
        elif trade_cd == 'd/r':
            # 아파트 전월세가격 리스트 출력
            query = """
                SELECT apt_detail.master_idx, pk_apt_detail,  year, mon, day, floor, apt_detail.extent, deposit as t_amt
                FROM apt_rent
                INNER JOIN apt_detail
                  ON apt_detail.idx = apt_rent.pk_apt_detail
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                AND (%s) 
                ORDER BY deal_ymd;"""
        else:
            raise NotImplemented()

        cursor.execute(query, params=())
        return cursor

    @staticmethod
    def get_trade_price_simple(apt_detail_pk: int, trade_cd: str):
        # 아파트의 매매, 전세/월세, 가격
        if trade_cd == 't':
            # 아파트 매매가격 리스트 출력
            query = ("""
                SELECT year, mon, day, t_amt
                FROM apt_trade
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                ORDER BY deal_ymd;""")
        elif trade_cd == 'd/r':
            # 아파트 전월세가격 리스트 출력
            query = ("""
                SELECT year, mon, day, deposit as t_amt
                FROM apt_rent
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                ORDER BY deal_ymd;""")
        else:
            raise NotImplemented()
        cursor.execute(query, params=(apt_detail_pk, ))
        return cursor

    @staticmethod
    def get_sale_price(apt_detail_pk, trade_cd):
        # 해당 아파트 층의 매물가격 리스트 출력
        query = """                
                    SELECT c.pk_apt_detail, CAST(d.reg_date AS DATETIME), 
                           CASE d.floor WHEN '고' THEN CAST(m.max_jisang_floor AS SIGNED)
                                        WHEN '중' THEN CAST(m.max_jisang_floor/2 AS SIGNED)
                                        WHEN '저' THEN 1
                                        WHEN '-' THEN CAST(m.max_jisang_floor/2 AS SIGNED)
                                        ELSE CAST(d.floor AS SIGNED)
                           END floor,
                           a.extent, d.price/a.extent AS price
                    FROM apt_detail a
                     INNER JOIN apt_master m
                      ON m.idx = a.master_idx
                     INNER JOIN naver_apt_sale_detail_group c
                      ON a.idx = c.pk_apt_detail
                     INNER JOIN naver_apt_sale d
                      ON c.pk_naver_apt_master = d.idx
                     AND c.supply_extent = d.supply_extent
                     AND c.private_extent = d.extent
                    WHERE c.pk_apt_detail IN (%s)
                     AND d.use_yn = 'y'
                     AND d.trade_cd IN ("%s")
                   ORDER BY reg_date;
                """ % (apt_detail_pk, trade_cd)
        #print(query)
        cursor.execute(query)
        return cursor
        
    @staticmethod
    def get_trade_price(apt_detail_pk: int, trade_cd: str):
        # 아파트의 매매, 전세/월세, 가격
        if trade_cd == 't':
            # 아파트 매매가격 리스트 출력
            query = ("""
                SELECT pk_apt_trade, pk_apt_detail, year, mon, day, floor, extent, t_amt
                FROM apt_trade
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                ORDER BY deal_ymd;""")
        elif trade_cd == 'd/r':
            # 아파트 전월세가격 리스트 출력
            query = ("""
                SELECT pk_apt_rent, pk_apt_detail, year, mon, day, floor, extent, deposit as t_amt
                FROM apt_rent
                WHERE EXCEPT_YN='n' AND pk_apt_detail=%s
                ORDER BY deal_ymd;""")
        else:
            raise NotImplemented()
        cursor.execute(query, params=(apt_detail_pk, ))
        return cursor

    @staticmethod
    def get_trade_price_optimize(apt_detail_pk, trade_cd):
        if trade_cd == 't':
            query = """
                SELECT pk_apt_trade, pk_apt_detail, 
                       CAST(CONCAT(year, '-', LPAD(mon,2,'0'), '-', LPAD(day,2,'0')) AS DATETIME), 
                       CAST(floor AS SIGNED) AS floor, apt_detail.extent, t_amt / apt_detail.extent AS price  
                FROM apt_trade
                INNER JOIN apt_detail
                      ON apt_detail.idx = apt_trade.pk_apt_detail
                WHERE pk_apt_detail in (%s)
                  AND EXCEPT_YN='n'
                ORDER BY deal_ymd, pk_apt_trade;
            """ % (apt_detail_pk,)
        else:
            # TODO : 코드 변경 필요!!
            query = """
                SELECT pk_apt_trade, pk_apt_detail, deal_ymd, floor, apt_detail.extent, t_amt 
                FROM apt_rent
                WHERE pk_apt_detail in (%s)
                  AND EXCEPT_YN='n'
                ORDER BY deal_ymd;
            """ % (apt_detail_pk,)
        #print(query)
        cursor.execute(query)
        return cursor


    @staticmethod
    def get_sale_price_with_floor(apt_detail_pk, date_range, floor, trade_cd):
        # 해당 아파트 층의 매물가격 리스트 출력
        query = """                
            SELECT c.pk_apt_detail, d.reg_date, d.floor, d.price
            FROM apt_detail a
             INNER JOIN naver_apt_sale_detail_group c
              ON a.idx = c.pk_apt_detail
             INNER JOIN naver_apt_sale d
              ON c.pk_naver_apt_master = d.idx
             AND c.supply_extent = d.supply_extent
             AND c.private_extent = d.extent
            WHERE c.pk_apt_detail IN (%s)
             AND d.reg_date IN (%s)
             AND d.floor IN (%s)
             AND d.use_yn = 'y'
             AND d.trade_cd IN ("%s")
           ORDER BY reg_date;
        """ % (apt_detail_pk, date_range, floor, trade_cd)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_sale_price_with_floor_extent(apt_detail_pk, date_range, floor, trade_cd):
        # 해당 아파트 층의 매물가격 리스트 출력
        if date_range is None:
             query = """                
                SELECT c.pk_apt_detail, d.reg_date, d.floor, d.extent, d.price
                FROM apt_detail a
                 INNER JOIN naver_apt_sale_detail_group c
                  ON a.idx = c.pk_apt_detail
                 INNER JOIN naver_apt_sale d
                  ON c.pk_naver_apt_master = d.idx
                 AND c.supply_extent = d.supply_extent
                 AND c.private_extent = d.extent
                WHERE c.pk_apt_detail IN (%s)
                 AND d.floor IN (%s)
                 AND d.use_yn = 'y'
                 AND d.trade_cd IN ("%s")
               ORDER BY reg_date;
             """ % (apt_detail_pk, floor, trade_cd)
        else:
            query = """                
                SELECT c.pk_apt_detail, d.reg_date, d.floor, d.extent, d.price
                FROM apt_detail a
                 INNER JOIN naver_apt_sale_detail_group c
                  ON a.idx = c.pk_apt_detail
                 INNER JOIN naver_apt_sale d
                  ON c.pk_naver_apt_master = d.idx
                 AND c.supply_extent = d.supply_extent
                 AND c.private_extent = d.extent
                WHERE c.pk_apt_detail IN (%s)
                 AND d.reg_date IN (%s)
                 AND d.floor IN (%s)
                 AND d.use_yn = 'y'
                 AND d.trade_cd IN ("%s")
               ORDER BY reg_date;
             """ % (apt_detail_pk, date_range, floor, trade_cd)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_trade_price_with_floor(apt_detail_pk, date_range, floor, trade_cd):
        if trade_cd == 't':
            query = """
                SELECT pk_apt_trade, pk_apt_detail, deal_ymd, floor, t_amt 
                FROM apt_trade
                WHERE EXCEPT_YN='n'
                  AND pk_apt_detail in (%s)
                  AND floor in (%s)
                  AND deal_ymd IN (%s)
                ORDER BY deal_ymd;
            """ % (apt_detail_pk, floor, date_range)
        else:
            query = """
                SELECT pk_apt_rent, pk_apt_detail, deal_ymd, floor, deposit as t_amt 
                FROM apt_rent
                WHERE EXCEPT_YN='n'
                  AND pk_apt_detail in (%s)
                  AND floor in (%s)
                  AND deal_ymd IN (%s)
                ORDER BY deal_ymd;
            """ % (apt_detail_pk, floor, date_range)
        cursor.execute(query)
        return cursor

    @staticmethod
    def get_trade_price_with_floor_extent(apt_detail_pk, date_range, floor, trade_cd):
        if date_range is None:
            if trade_cd == 't':
                query = """
                SELECT pk_apt_trade, pk_apt_detail, deal_ymd, floor, apt_detail.extent, t_amt
                    FROM apt_trade
                   INNER JOIN apt_detail
                      ON apt_detail.idx = apt_trade.pk_apt_detail
                   WHERE EXCEPT_YN='n'
                     AND pk_apt_detail in (%s)
                     AND floor in (%s)
                   ORDER BY deal_ymd;
                """ % (apt_detail_pk, floor)
            else:
                # TODO : 코드 변경 필요!!
                query = """
                    SELECT pk_apt_rent, pk_apt_detail, deal_ymd, floor, apt_detail.extent, t_amt 
                     FROM apt_rent
                    INNER JOIN apt_detail
                       ON apt_detail.idx = apt_rent.pk_apt_detail
                    WHERE EXCEPT_YN='n'
                      AND pk_apt_detail in (%s)
                      AND floor in (%s)
                    ORDER BY deal_ymd;
                """ % (apt_detail_pk, floor)
            cursor.execute(query)
        else:
	        if trade_cd == 't':
	            query = """
	            SELECT pk_apt_trade, pk_apt_detail, deal_ymd, floor, apt_detail.extent, t_amt
	                FROM apt_trade
	                INNER JOIN apt_detail
	                  ON apt_detail.idx = apt_trade.pk_apt_detail
	                WHERE EXCEPT_YN='n'
	                  AND pk_apt_detail in (%s)
	                  AND floor in (%s)
	                  AND deal_ymd IN (%s)
	                ORDER BY deal_ymd;
	            """ % (apt_detail_pk, floor, date_range)
	
	
	
	
	
	
	
	
	
	
	        else:
	            query = """
	                SELECT pk_apt_rent, pk_apt_detail, deal_ymd, floor, apt_detail.extent, deposit as t_amt
	                FROM apt_rent
	                INNER JOIN apt_detail
	                  ON apt_detail.idx = apt_rent.pk_apt_detail
	                WHERE EXCEPT_YN='n'
	                  AND pk_apt_detail in (%s)
	                  AND floor in (%s)
	                  AND deal_ymd IN (%s)
	                ORDER BY deal_ymd;
	            """ % (apt_detail_pk, floor, date_range)
	        cursor.execute(query)
        return cursor

    @staticmethod
    def get_new_apt_trade_list(apt_detail_pk, date_range, trade_cd):
        query = """
            SELECT a.master_idx, c.pk_apt_detail, d.reg_date, d.floor, d.extent, d.price
            FROM apt_detail a
            INNER JOIN naver_apt_sale_detail_group c
              ON a.idx = c.pk_apt_detail
            INNER JOIN naver_apt_sale d
              ON c.pk_naver_apt_master = d.idx
              AND c.supply_extent = d.supply_extent
              AND c.private_extent = d.extent
            WHERE
              c.pk_apt_detail=%s
              AND d.reg_date in ("%s")
              AND d.use_yn = 'y'
              AND d.confirm_cd = 'new'
              AND d.trade_cd IN ("%s")
            ORDER BY reg_date;
        """ % (apt_detail_pk, date_range, trade_cd)
        cursor.execute(query, params=())
        return cursor

    @staticmethod
    def get_similarity_apt_list(apt_detail_pk):
        query = ("""
            SELECT similarity
            FROM apt_similarity
            WHERE idx = %s
        """)
        cursor.execute(query, params=(apt_detail_pk, ))
        return cursor

    @staticmethod
    def get_nearest_3km_apt(apt_detail_pk: int):
        query = ("""
            SELECT
               ROUND((6371*acos(cos(radians(b.latlngx))*cos(radians(a.latlngx))
               *cos(radians(a.latlngy)-radians(b.latlngy))\
               +sin(radians(b.latlngx))*sin(radians(a.latlngx)))) * 1000)  AS distance
              , c.idx as pk_apt_detail
            FROM apt_master a
            CROSS JOIN apt_master b
            INNER JOIN apt_detail c
            ON c.master_idx = b.idx
            WHERE a.idx = (select apt_detail.master_idx from apt_detail where apt_detail.idx=%s)
              AND a.idx != b.idx
              AND a.bldg_cd IN ('1')
              AND b.bldg_cd IN ('1')
              AND b.total_num_of_family >= 100  
            HAVING distance <= 3000
            ORDER BY distance;
        """)
        cursor.execute(query, params=(apt_detail_pk, ))
        return cursor

    @staticmethod
    def get_training_volume_standard_area(apt_detail_pk: int, trg_date: str):
        query = ("""
            SELECT yyyymmdd, area_20lt_trade_volume_cnt, area_20ge_30lt_trade_volume_cnt, 
                   area_30ge_40lt_trade_volume_cnt, area_40ge_50lt_trade_volume_cnt, area_50ge_trade_volume_cnt,
                   standard_area_20lt_trade_volume_cnt, standard_area_20ge_30lt_trade_volume_cnt, 
                   standard_area_30ge_40lt_trade_volume_cnt, standard_area_40ge_50lt_trade_volume_cnt, 
                   standard_area_50ge_trade_volume_cnt
            FROM bi_volume
            WHERE yyyymmdd like %s
              AND lawd_cd = (
                SELECT distinct lawd_cd
                FROM apt_rtms_cnt
                WHERE pk_apt_detail=%s
                LIMIT 1
              )
        """)
        cursor.execute(query, params=(trg_date, apt_detail_pk))

        return cursor

    @staticmethod
    def get_training_volume_all(apt_detail_pk: int):
        query = ("""
            SELECT CAST(yyyymmdd AS DATETIME), 
                   area_20lt_trade_volume_cnt / standard_area_20lt_trade_volume_cnt AS area_20lt_trade_volume_rate, 
                   area_20ge_30lt_trade_volume_cnt / standard_area_20ge_30lt_trade_volume_cnt AS area_20ge_30lt_trade_volume_rate,
                   area_30ge_40lt_trade_volume_cnt / standard_area_30ge_40lt_trade_volume_cnt AS area_30ge_40lt_trade_volume_rate, 
                   area_40ge_50lt_trade_volume_cnt / standard_area_40ge_50lt_trade_volume_cnt AS area_40ge_50lt_trade_volume_rate, 
                   area_50ge_trade_volume_cnt / standard_area_50ge_trade_volume_cnt AS area_50ge_trade_volume_rate,
                   year_05lt_trade_volume_cnt / standard_year_05lt_trade_volume_cnt AS year_05lt_trade_volume_rate, 
                   year_05ge_10lt_trade_volume_cnt / standard_year_05ge_10lt_trade_volume_cnt AS year_05ge_10lt_trade_volume_rate,
                   year_10ge_15lt_trade_volume_cnt / standard_year_10ge_15lt_trade_volume_cnt AS year_10ge_15lt_trade_volume_rate, 
                   year_15ge_25lt_trade_volume_cnt / standard_year_15ge_25lt_trade_volume_cnt AS year_15ge_25lt_trade_volume_rate, 
                   year_25ge_trade_volume_cnt / standard_year_25ge_trade_volume_cnt AS year_25ge_trade_volume_rate
            FROM bi_volume
            WHERE yyyymmdd < NOW()
              AND lawd_cd = (
                SELECT distinct lawd_cd
                FROM apt_rtms_cnt
                WHERE pk_apt_detail=%s
                LIMIT 1
              )
        """)
        cursor.execute(query, params=(apt_detail_pk,))

        return cursor

    @staticmethod
    def get_training_volume_standard_year(apt_detail_pk: int, trg_date: str):
        query = ("""
            SELECT yyyymmdd, year_05lt_trade_volume_cnt, year_05ge_10lt_trade_volume_cnt,
             year_10ge_15lt_trade_volume_cnt, year_15ge_25lt_trade_volume_cnt, year_25ge_trade_volume_cnt, 
             standard_year_05lt_trade_volume_cnt, standard_year_05ge_10lt_trade_volume_cnt, 
             standard_year_10ge_15lt_trade_volume_cnt, standard_year_15ge_25lt_trade_volume_cnt, 
             standard_year_25ge_trade_volume_cnt
            FROM bi_volume
            WHERE yyyymmdd like %s
              AND lawd_cd = (
                SELECT distinct lawd_cd
                FROM apt_rtms_cnt
                WHERE pk_apt_detail=%s
                LIMIT 1
              )
        """)
        cursor.execute(query, params=(trg_date, apt_detail_pk))
        return cursor

    @staticmethod
    def insert_or_update_predicate_value(values):
        query = """
            insert into predicate_price (reg_date, price_max, price_avg, price_min, apt_detail_pk, trade_cd) 
            values (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            price_min=VALUES(price_min), price_avg=VALUES(price_avg), price_max=VALUES(price_max)
        """
        cursor.executemany(query, values)
        cnx.commit()

    @staticmethod
    def insert_or_update_predicate_smoothing_value(values):
        query = """
            insert into predicate_price (reg_date, price_max_smoothing, price_avg_smoothing, price_min_smoothing, 
            apt_detail_pk, trade_cd)
            values (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            price_min_smoothing=VALUES(price_min_smoothing), price_avg_smoothing=VALUES(price_avg_smoothing), 
            price_max_smoothing=VALUES(price_max_smoothing)
        """
        cursor.executemany(query, values)
        cnx.commit()
