# -*- coding:utf-8 -*-
from database import GinQuery

pk_detail = 2
cursor = GinQuery.select_t_amt(pk_detail)
result = cursor.fetchall()
for line in result:
    master_id, ymd, cnt, floor, t_amt = line
    print(floor, end=' ')
    print(t_amt)