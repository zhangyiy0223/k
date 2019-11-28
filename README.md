# Dilibraa

1. 初始代码    0.00970
2. 改变优化器（SGD→RMSProp），去掉train/val的normalization    0.01460
3. 增加epoch（epoch = 30，发现应该在18比较好）    0.01610
4. 之前忘记去掉test的normalization，这次去掉    0.02150

5. 发现测试有bug，将test.py复制到train.py中一起跑。 