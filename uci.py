import os
import openml
import numpy as np
from scipy.io import savemat
from pandas import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 步骤 1: 获取所有带 'uci' 标签的数据集
print("正在从OpenML获取带有'uci'标签的数据集...")
uci_datasets = openml.datasets.list_datasets(tag='uci', output_format='dataframe')

# 打印总数和部分数据集的信息
print(f"找到 {len(uci_datasets)} 个数据集。")
print("前10个数据集示例:")
print(uci_datasets[['did', 'name']].head(10))

# 从数据集中选择前100个数据集的ID（可根据需要调整）
dataset_ids = uci_datasets['did'].head(5000).tolist()

# 输出目录
output_dir = './mat_datasets'
os.makedirs(output_dir, exist_ok=True)

# 步骤 2: 下载并保存为.mat格式
print(f"开始下载和转换数据集，共计 {len(dataset_ids)} 个数据集...")
for dataset_id in dataset_ids:
    try:
        # 下载数据集
        dataset = openml.datasets.get_dataset(dataset_id)
        X_df, y_df, _, _ = dataset.get_data(
            dataset_format='dataframe', target=dataset.default_target_attribute
        )

        # 创建拷贝避免对原数据集修改
        X = X_df.copy()
        y = y_df.to_numpy().flatten()

        # 对X中的列进行类型检查和转换
        for column in X.columns:
            # 判断该列的数据类型是否为 CategoricalDtype
            if isinstance(X[column].dtype, CategoricalDtype):
                print(f"数据集 {dataset.name} 列 '{column}' 为 CategoricalDtype 类型，正在转换为字符串...")
                X[column] = X[column].astype(str)

            # 接着如果还是非数值型，则用 LabelEncoder 处理
            if not np.issubdtype(X[column].dtype, np.number):
                print(f"数据集 {dataset.name} 列 '{column}' 包含非数值数据，正在进行数值化编码...")
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))

        # 现在X应全部为数值类型，可以安全转换为Numpy数组
        X = X.to_numpy()

        # 填充缺失值
        if np.isnan(X).any():
            print(f"数据集 {dataset.name} 存在缺失值，正在进行填充...")
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # 对Y进行数值化处理
        if not np.issubdtype(y.dtype, np.number):
            print(f"数据集 {dataset.name} 的Y包含非数字值，正在转换为数字编码...")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # 确保Y为列向量 (n, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 确保所有数据为double类型 (float64)
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # 确保X和Y行数一致
        assert X.shape[0] == y.shape[0], f"数据集 {dataset.name} 的X和Y行数不一致！"

        # 保存为.mat文件
        num_samples = X.shape[0]
        dataset_name = f"{dataset.name.replace(' ', '_')}_{num_samples}"
        mat_file_path = os.path.join(output_dir, f"{dataset_name}.mat")
        savemat(mat_file_path, {'X': X, 'Y': y})

        print(f"数据集 {dataset.name} (ID: {dataset_id}) 已保存为 {mat_file_path}.")
    except Exception as e:
        print(f"数据集ID {dataset_id} 处理失败，错误: {str(e)}")

print("全部转换完成！文件已保存至:", output_dir)
