# Assignment3

#复刻研报：国泰君安-量化择时研究系列01 ：宽基指数如何择时，通过估值、流动性和拥挤度构建量化择时策略

#研报思路：按照估值分位数构建PB分位数因子、PE分位数因子，刻画市场的拥挤度。当市场拥挤度较高时，空头，市场拥挤度较低时，多头

#结果：见附图

#评价：对于空头面临更大的风险，尤其是面对极端上涨行情时。如在2014-2016行情中，事实上在上涨不到一半时，空头信号就已经发出，如果完全按照空头信号操作，应该在该行情中价值归0。因此有必要进行止损操作，即在回撤到达一定阈值时，停止空头。遇到多头信号之后再进场。
