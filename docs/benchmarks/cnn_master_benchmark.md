# 📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 8 MÔ HÌNH CNN (TASK #84)

*Bảng đối chuẩn chính (Main Paper Table) đánh giá toàn diện trên tập HyperKvasir (23 lớp, Fold 0).*

Task ID                Model               Architecture    Attention Mechanism Resolution  Params (M)  FLOPs (G)  Accuracy (%)  F1-macro (%)  Weighted F1 (%)  AUC-ROC (%) Training Time (50ep)  GPU Latency (ms)  GPU FPS  CPU Latency (ms)  CPU FPS
    #76 ResNet-50 (Baseline)          Standard Residual                   None    224x224       23.51       4.12         88.70         65.51            88.88        97.42     4.3 h (~311s/ep)              6.75    148.2             54.05     18.5
    #77           ResNet-101              Deep Residual                   None    224x224       44.55       7.85         89.45         68.04            89.24        97.85     7.6 h (~549s/ep)             11.30     88.5             89.29     11.2
    #78         DenseNet-121         Dense Connectivity                   None    224x224        7.98       2.88         89.83         68.04            89.54        98.12     3.5 h (~254s/ep)              8.90    112.4             67.57     14.8
    #79      EfficientNet-B4           Compound Scaling          SE (Internal)    380x380       19.34       4.40         88.61         62.70            88.91        97.21     7.9 h (~570s/ep)             23.75     42.1            217.39      4.6
    #80      EfficientNet-B5           Compound Scaling          SE (Internal)    456x456       30.40       9.90         88.61         66.11            89.01        97.68   16.8 h (~1207s/ep)             41.15     24.3            476.19      2.1
    #81     EfficientNetV2-S Fused-MBConv / Progressive          SE (Internal)    384x384       21.46       8.40         90.44         67.97            90.16        98.45    1.49 h (~107s/ep)             13.07     76.5            112.36      8.9
    #82    MobileNetV3-Large           Lightweight Edge          SE (Internal)    224x224        4.23       0.22         88.37         62.55            88.10        97.10     0.25 h (~18s/ep)              6.39    156.6             21.42     46.7
    #83         SE-ResNet-50 Channel Attention Residual Squeeze-and-Excitation    224x224       26.14       4.14         88.42         64.65            88.32        97.35     0.65 h (~46s/ep)              7.12    140.4             58.82     17.0
