
# filter(方法：返回值为boolean，可迭代对象)


x = [1,11,2,45,7,6,14]

x2 = filter(lambda x:x>10,x)
print(list(x2))


x3 = ['abc','xy12','+++']
# 过滤由字母或数字组成的字符串
x3_f = filter(lambda x:x.isalnum(),x3)
print(list(x3_f))
