import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from imbens.ensemble import (
    BalanceCascadeClassifier,
    SelfPacedEnsembleClassifier,
    UnderBaggingClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
    BalancedRandomForestClassifier,
    AdaCostClassifier,
    AdaUBoostClassifier,
    AsymBoostClassifier
)
from UADF import DualGranularBalancedDeepForest
from demo import get_config

if __name__ == '__main__':
    # 加载数据
    X, y = np.load('pred_results/X.npy'), np.load('pred_results/y.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # 配置不平衡数据处理模型
    ensemble_methods = {
        'BalanceCascade': BalanceCascadeClassifier(n_estimators=50),  # 设置生成50棵树
        'SelfPacedEnsemble': SelfPacedEnsembleClassifier(n_estimators=50),
        'UnderBagging': UnderBaggingClassifier(n_estimators=50),
        'EasyEnsemble': EasyEnsembleClassifier(n_estimators=50),
        'RUSBoost': RUSBoostClassifier(n_estimators=50),
        'BalancedRandomForest': BalancedRandomForestClassifier(n_estimators=50),
        'AdaCost': AdaCostClassifier(n_estimators=50),
        'AdaUBoost': AdaUBoostClassifier(n_estimators=50),
        'AsymBoost': AsymBoostClassifier(n_estimators=50)
    }

    # 创建用于保存图像的文件夹
    output_dir = "tree_sampling_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # 对每个集成方法进行训练和采样数据的可视化
    # for method_name, model in ensemble_methods.items():
    for method_name, model in {'DualGranularBalancedDeepForest': DualGranularBalancedDeepForest(get_config())}.items():
        print(f"Training {method_name}...")
        model.fit(X_train, y_train)

        # # 绘制第50棵树的采样数据
        # plt.figure(figsize=(6, 6))
        # plt.scatter(selected_samples[:, 0], selected_samples[:, 1], color="blue", alpha=0.5, label="Selected samples")
        # plt.scatter(not_selected_samples[:, 0], not_selected_samples[:, 1], color="gray", alpha=0.5, label="Not selected samples")
        # plt.title(f"Selected vs Not Selected samples (50th Tree) - {method_name}")
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, f"{method_name}_50th_tree_sampling.png"), format="png")
        # plt.show()
