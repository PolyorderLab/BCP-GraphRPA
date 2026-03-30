这个文件中有四嵌段聚合物进行单任务回归(预测chiN)的结果
你可以进行另外一个单任务回归(预测k*)
训练脚本路径/home/zxy/project/TAPD/Demo/SingleTaskLearning/kSTLtrain.py
训练集路径/home/zxy/project/TAPD/TrainData/data/train_topo
推荐结果保存路径"/home/zxy/project/TAPD/Demo/outcome/xxxx"(自由命名)
测试集是从训练集中分割出的20%数据
因不涉及超参数优化，暂时没有设计验证集的逻辑
