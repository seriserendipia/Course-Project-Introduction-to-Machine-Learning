from itertools import combinations


def createlist(df):
    items = []
    for i in df:
        if i not in items:
            items.append(i)
    items.sort()
    return items


def create_candidate_k(transaction_list, k):
    '''创建候选k项集'''
    itemlist = createlist(transaction_list)

    candidate_set_k = dict()   # 初始化k项候选集

    if k == 1:
        for i in itemlist:   # 初始化计数为0
            candidate_set_k[i] = 0
        for i in itemlist:   # 遍历计数
            for transaction in transaction_list:
                if i in transaction:
                    candidate_set_k[i] +=1

    else:
        combinations_iter = combinations(itemlist,k)
        itemk = [i for i in combinations_iter]
        itemk.sort()
        for i in itemk:
            candidate_set_k[i] = 0
        for i in itemk:
            for transaction in transaction_list:
                if i in transaction:
                    candidate_set_k[i] += 1

    return candidate_set_k
