# -*- coding:utf-8 -*-
from settings import db_cursor as cursor


class GinQuery(object):
    @staticmethod
    def get_actual_price(pk_apt_detail):
        # 실거래가 매매
        query = ("""
            SELECT a.master_idx AS pk_apt_master,b.pk_apt_detail,a.supply_extent, a.extent,b.t_amt,b.floor,b.deal_ymd, 
               b.day, b.real_day
            FROM apt_detail a
            INNER JOIN apt_trade b
                ON a.idx = b.pk_apt_detail
            WHERE b.pk_apt_detail = '%s'
        """)

        cursor.execute(query, params=(pk_apt_detail, ))
        return cursor
