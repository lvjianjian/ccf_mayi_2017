# CCF 蚂蚁金服 wifi室内定位问题		

具体信息请查看[比赛地址](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231620)

* 主要目录结构: 

  \src 存放代码

  \data 存放原始数据

  \data\wifi_info_cache2 存放生成的wifi信息

  \result 存放一些结果,具体子目录这里不说了

* 运行:

  1. 首先调用util.py 生成wifi cache 信息
  2. 再调用stack_model_strong_sig_matrix_lonlat_wh.py 生成结果

* 主要思路:

  思路很简单,过滤掉信号较弱且出现次数较少的wifi, 用rf 和 ovr(rf)作为一级学习器,在用rf作为二级学习器做的stacking模型,二级学习器除了用一级学习器的输出概率为特征外,还是用原来特征.

* 成绩:

  初赛B榜0.9129, 排名72.

   

  ​

  ​







