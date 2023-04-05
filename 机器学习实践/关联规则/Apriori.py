import numpy as np
import pandas as pd
from itertools import combinations

def scan_transaction_list_and_get_items(transaction_list):
    items = []
    for i in transaction_list:
        if i not in items:
            items.append(i)
    items.sort()
    return items


def genetate_candidate_set_k(k_1_itemsets):
    combinations_iter = combinations(k_1_itemsets,2)
    candidate_set_k = [set(np.array(i).flatten()) for i in combinations_iter]
    candidate_set_k = list(set(candidate_set_k))
    return candidate_set_k



def get_k_frequent_itemsets(candidate_set_k, support_threshold, transaction_list):
    candidate_dict_k = dict()
    for i in candidate_set_k:
        candidate_dict_k[i] = 0
    for i in candidate_set_k:
        for transaction in transaction_list:
            if i in transaction:
                candidate_dict_k[i] += 1


if __name__ == '__main__':
    support_threshold, confidence_threshold = 0.3
    k_1_itemsets = []
    transaction_list = []
    items = scan_transaction_list_and_get_items(transaction_list)
    frequent_itemsets = []

    for i in range(len(items)):

        candidate_set_k = genetate_candidate_set_k(k_1_itemsets)
        k_frequent_itemsets = get_k_frequent_itemsets(candidate_set_k,support_threshold,transaction_list)
        frequent_itemsets.extend(k_frequent_itemsets)
        k_1_itemset = k_frequent_itemsets
        if len(k_1_itemset) == 0:
            break
