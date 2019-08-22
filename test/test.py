# -*- coding:utf-8 -*-
from database import GinQuery

test = GinQuery.get_actual_price(2)
for data in test:
    print(data)