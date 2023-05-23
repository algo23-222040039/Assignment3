# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:38:26 2023

@author: lijia
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime,date,time






def RSI(data,n):
    close = data['close']
    low = data['low']
    high = data['high']
    change = data['chg']
    
    #n天上涨或下跌点数：1.首先构造上涨序列和下跌序列；2.取rolling计算平均值
    change_p = change.apply(lambda x: x if x>0 else 0)
    change_n = change.apply(lambda x: -x if x< 0 else 0)
    change_p2 = change.apply(lambda x: x if x>0 else None)
    change_n2 = change.apply(lambda x: -x if x< 0 else None)
    
    
    
    rolling_low = low.rolling(n)
    rolling_high = high.rolling(n)
    rolling_close = close.rolling(n)#n天的
    # mean_chg_p = change_p.rolling(n).mean() / change_p2.rolling(n).count()
    # mean_chg_n = change_n.rolling(n).mean() / change_n2.rolling(n).count()
    mean_chg_p = change_p.rolling(n).mean()
    mean_chg_n = change_n.rolling(n).mean() 
  
    RS = mean_chg_p/mean_chg_n
    RSI = 100 - (100/(1+RS))
    data['RSI'] = RSI
   
    return data

def PEPB(data):
    close = data['close']
    low = data['low']
    high = data['high']
    change = data['chg']
    PE = data['PE']
    PB = data['PB']
    
    PE_pct= []
    PB_pct  = []
    PBPE_pct = []
    for i in range(0,len(PE)):
        if i < 1000:
            PE_pct.append(None)
            PB_pct.append(None)
            PBPE_pct.append(None)
        else:
            PE_pct.append(arg_percentile(PE[i-1], PE[i-1000:i]))
            PB_pct.append(arg_percentile(PB[i-1], PB[i-1000:i]))
            PBPE_pct.append((PE_pct[i] * PB_pct[i])**(1/2))
            
    data['PB_pct'] =PB_pct
    data['PE_pct'] =PE_pct
    data['PBPE_pct'] = PBPE_pct
    
    return data
  
    
    #要获取历史分位数

def arg_percentile(x,series):
    '''
    求series中x的分位数
    '''
    return 1 - np.count_nonzero(x <= series)/series.size
    
    



#计算信号
def signal2(data):
    PBPE = data['PBPE_pct']
    signals = []
    
    
    K1 = 0.2
    K2 = 0.6
    K3 = 0.5
    K4 = 0.9
    
    i = 0
    for pre_PBPE, PBPE in zip(PBPE.shift(1),PBPE):
        print(i)
        signal = None
        print(PBPE)
        if i ==0 :
            signal =0
        elif pd.isna(PBPE):
            signal =0
        else:
            if signals[i-1] == 0:
                if PBPE <= K1:
                    signal = 1
                elif PBPE >K1 and PBPE <= K4:
                    signal = 0
                elif PBPE > K4:
                    signal = -1
            elif signals[i-1] == 1:
                if PBPE < K2:
                    signal = 1
                else:
                    signal =0
            elif signals[i-1] == -1:
                if PBPE > K3:
                    signal = -1
                else:
                    signal = 0
        i = i + 1
        
        print(signal)

        signals.append(signal)
        
    
    data['signal'] = signals
    
    return data

#计算持仓
def position(data):
    data['signal_last'] = data['signal'].shift(1)
    data['position'] = data['signal'].fillna(method='ffill').shift(1).fillna(0)
    return data



def date2row(date,dateSeries, type = "yyyy-mm-dd",):
    
    if type == "yyyy-mm-dd":
        d = datetime.strptime(date,'%Y-%m-%d')
        
    index_date = 0
    for i in range(0,np.size(dateSeries)):
        
        if dateSeries[i] <= d:
            index_date = index_date + 1
        else:
            break
        
    return index_date





    
def statistic_performance1(data, initCurrency = 10000, fee =0.00035):
    
    position = data['position'] # 获得交易方向
    high = data['high']
    low = data['low']
    avg = high*0.5+low*0.5
    
    data_period = (data['日期'].max()-data['日期'].min()).days
    r0 = 0
    currency = []
    hold = []
    value = []
    limit_cash = 0.3
    
    #生成持仓数据：无卖空
    i =0
    for posi, pri,pre_posi,pre_pri in zip(position,avg,position.shift(1),avg.shift(1)):
        if i ==0:
            c = initCurrency
            h = 0
        else:
            c = currency[i-1]
            h = hold[i-1]
            
        p = pri #指数价格
        if posi == 1 :
            h = h + c/ (1+fee)/p
            c = 0
            #转换为多头
        elif posi == 0:
            #0时平仓
            if pre_posi == 1:
                c = c + h * p *(1-fee)
                h = 0
            elif pre_posi ==-1:
                c = c + h * p*(1-fee)
                h =0
            elif pre_posi ==0:
                c=c
                h=h
        elif posi ==-1:
            if pre_posi == 0:
                h = h - c/(1+fee)/p
                c = c + c
                
            elif pre_posi == -1:
                v = c + p*h
                temp = (-p*h)/v
                if temp < limit_cash:
                    h = h + ( -(1+limit_cash)* p *h - limit_cash * c)/(1+limit_cash)/p
                    c = c
                else:
                    h = h
                    c = c

            
        
        
        currency.append(c)
        hold.append(h)
        value.append(h * p +c)
        i = i+1
        
        
    data['value']=value
    data['hold']=hold
    data['currency']=currency
    
    hold_r = data['value']/data['value'].shift(1) -1
    hold_win = hold_r>0
    hold_cumu_r = (1+hold_r).cumprod() - 1
    drawdown = (hold_cumu_r.cummax()-hold_cumu_r)/(1+hold_cumu_r).cummax()  
    data['hold_r'] = hold_r
    data['hold_win'] = hold_win
    data['hold_cumu_r'] = hold_cumu_r
    data['drawdown'] = drawdown
    v_hold_cumu_r = hold_cumu_r.tolist()[-1]
    v_pos_hold_times= 0 
    v_pos_hold_win_times = 0
    v_pos_hold_period = 0
    v_pos_hold_win_period = 0
    v_neg_hold_times= 0 
    v_neg_hold_win_times = 0
    v_neg_hold_period = 0
    v_neg_hold_win_period = 0
    tmp_hold_r = 0
    tmp_hold_period = 0
    tmp_hold_win_period = 0
   
    for w, r, pre_pos, pos in zip(hold_win, hold_r, position.shift(1), position):
        # 有换仓（先结算上一次持仓，再初始化本次持仓）
        if pre_pos!=pos: 
            # 判断pre_pos非空：若为空则是循环的第一次，此时无需结算，直接初始化持仓即可
            if pre_pos == pre_pos:
                # 结算上一次持仓
                if pre_pos>0:
                    v_pos_hold_times += 1
                    v_pos_hold_period += tmp_hold_period
                    v_pos_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:
                        v_pos_hold_win_times+=1
                elif pre_pos<0:
                    v_neg_hold_times += 1      
                    v_neg_hold_period += tmp_hold_period
                    v_neg_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:                    
                        v_neg_hold_win_times+=1
            # 初始化本次持仓
            tmp_hold_r = 0
            tmp_hold_period = 0
            tmp_hold_win_period = 0  
        else: # 未换仓
            if abs(pos)>0:
                tmp_hold_period += 1
                if r>0:
                    tmp_hold_win_period += 1
                if abs(r)>0:
                    tmp_hold_r = (1+tmp_hold_r)*(1+r)-1       

    v_hold_period = (abs(position)>0).sum()
    v_hold_win_period = (hold_r>0).sum()
    v_max_dd = drawdown.max()    
    #年化收益 =总收益/
    v_annual_ret = pow( 1+v_hold_cumu_r, 
                      365/data_period)-1
    v_annual_std = hold_r.std() * np.sqrt(250*1440/data_period) 
    v_sharpe = v_annual_ret / v_annual_std
    performance_cols = ['累计收益', 
                        '多仓次数', '多仓胜率', '多仓平均持有期(交易日)', 
                        '空仓次数', '空仓胜率', '空仓平均持有期(交易日)', 
                        '日胜率', '最大回撤', '年化收益/最大回撤',
                        '年化收益', '年化标准差', '年化夏普']
    performance_values = ['{:.2%}'.format(v_hold_cumu_r),
                          v_pos_hold_times, 
                          '{:.2%}'.format(v_pos_hold_win_times/v_pos_hold_times), 
                          '{:.2f}'.format(v_pos_hold_period/v_pos_hold_times),
                          v_neg_hold_times, 
                          '{:.2%}'.format(v_neg_hold_win_times/v_neg_hold_times), 
                          '{:.2f}'.format(v_neg_hold_period/v_neg_hold_times),
                          '{:.2%}'.format(v_hold_win_period/v_hold_period), 
                          '{:.2%}'.format(v_max_dd), 
                          '{:.2f}'.format(v_annual_ret/v_max_dd),
                          '{:.2%}'.format(v_annual_ret), 
                          '{:.2%}'.format(v_annual_std), 
                          '{:.2f}'.format(v_sharpe)]
    performance = pd.DataFrame(performance_values, index=performance_cols)
    return data, performance
    
    
    

'''
下面开始回测
'''

#读取数据
data0=pd.read_excel('000300.xlsx')
str1 = 'HS300'

# data0=pd.read_excel('000300.xlsx')
# str1 ='HS300'


data=data0.copy()
#对列名重命名，便于后面计算
data.rename(columns={'开盘价(元)':'open','最高价(元)':'high','最低价(元)':'low','收盘价(元)':'close','成交额(百万元)':'vol'},inplace=True)
#计算每日涨跌幅

data['pre_close'] = data['close'].shift(1)
data['pct_chg'] = (data['close']-data['pre_close'])/data['pre_close']
data['chg'] = data['close']-data['pre_close']
#计算历史分位数PEPB指标





#生成因子序列
# data_RSI = RSI(data, 14)
data_PEPB =PEPB(data)
#生成信号序列和持仓序列
data_signal = signal2(data_PEPB)
data_position = position(data_signal)

#回测变量：
date1 = "2012-01-01"
date2 = "2022-12-31"
fee = 0.00035






data_test = data_position.iloc[date2row(date1,data['日期']):date2row(date2,data['日期'])]


result, performance = statistic_performance1(data_test)
print(performance)



#结果可视化
plt.figure(figsize=(20,8))
plt.title('PBPE Performance')
cumu_hold_close = (result['hold_cumu_r']+1)*data_test['close'].tolist()[0]
plt.plot(data_test['日期'],cumu_hold_close,color='black')
plt.plot(data_test['日期'],data_test['close'],color='red')
plt.legend(['PBPE',str1])
plt.grid()
plt.show()

plt.figure(figsize=(20,8))
plt.title('Drawdown')
plt.plot(data_test['日期'],-result['drawdown'],color='black')
# plt.ylim([-0.8,0])
plt.grid()
plt.show()




