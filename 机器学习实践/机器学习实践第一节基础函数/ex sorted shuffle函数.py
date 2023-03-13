import numpy as np
import random

x1 = list(range(9,21))
random.shuffle(x1)
print(x1)
x2 = sorted(x1)
print(x2)

# 倒序排序
x4 = sorted(x1, reverse=True)
print(x4)

# x1作为变量进入key，然后排序
x5 = sorted(x1, key=lambda i: 10-1)

print(x5)

#
x3 = [i*2 for i in x1]
print(x3)
