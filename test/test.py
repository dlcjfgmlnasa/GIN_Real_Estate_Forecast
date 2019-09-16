# -*- coding:utf-8 -*-
from database import GinQuery

test = GinQuery.test()
total_temp = [ data[0] for data in test]
print(len(total_temp))
temp = set(total_temp)
print(len([i for i in total_temp if i is None]))
print(len([i for i in total_temp if i is not None]))
print(len(temp))
