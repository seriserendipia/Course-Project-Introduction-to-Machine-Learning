def jieccheng(x):
    if x < 0:
        raise Exception("阶乘的输入必须大于零")
    elif not isinstance(x,type(1)):
        raise Exception("阶乘的输入必须为整数")
    else:
        if x == 1 or x == 0:
            return 1
        elif x > 1:
            return x*jieccheng(x - 1)
        else:
            raise Exception("超出域外")

for i in range(10):
    print(f"{i}   的阶乘为  {jieccheng(i)}")
