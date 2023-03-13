

from statsmodels.tsa.stattools import adfuller
def is_data_stationary(data):
  result = adfuller(data)
  p_value = result[1]
  print("p_value = ",result[1])
  if p_value >0.05:
    return True
  return False


def cal_ARIMA_parameter(data):
  #TODO
  p, d, q = 0
  return (p,d,q)
