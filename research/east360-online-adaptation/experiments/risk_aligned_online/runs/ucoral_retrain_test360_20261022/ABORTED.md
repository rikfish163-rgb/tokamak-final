# 未启动训练的路径错误记录

首次启动在读取参考预测凭据时退出：旧 `context_feedback_east_uniform_20261022` 目录不含 `prediction_receipts.json`。尚未读取20炮信号、拟合统计或训练；本目录没有实验成绩。

后续入口显式指向含已验证逐炮凭据的 `warmstart_alignment_test360_v2_20261022`，新实验写入 `ucoral_retrain_test360_v2_20261022`。保留本目录供审计，不把它计为失败的算法实验。
