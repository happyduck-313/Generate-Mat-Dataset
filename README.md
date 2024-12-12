# 生成MAT格式数据集 | Generate MAT Datasets

## 项目简介 | Project Description
本项目包含生成MAT格式数据集的Python脚本。代码从OpenML获取数据集，并将特征（`X`）和标签（`Y`）处理后保存为`.mat`文件。用户可以通过调整脚本中的参数来更改需求。

This project includes Python scripts to generate MAT-format datasets. The scripts fetch datasets from OpenML, process the features (`X`) and labels (`Y`), and save them as `.mat` files. Users can modify the parameters in the scripts to meet their specific needs.

## 文件结构 | File Structure
- `uci.py`: 获取带有 `uci` 标签的数据集，处理后保存为MAT格式。
- `datasets.py`: 根据特定过滤条件选择数据集并保存为MAT格式。

- `uci.py`: Fetches datasets tagged with `uci`, processes them, and saves them in MAT format.
- `datasets.py`: Selects datasets based on specific filtering criteria and saves them in MAT format.

## 环境依赖 | Environment Requirements

在运行代码前，请确保安装以下依赖：

Before running the code, make sure to install the following dependencies:

```bash
pip install openml numpy scipy pandas scikit-learn
```

## 使用方法 | Usage

### 1. 运行 `uci.py` | Run `uci.py`

`uci.py` 脚本会从OpenML中获取所有带有 `uci` 标签的数据集，将数据处理并保存为MAT格式：

The `uci.py` script fetches all datasets tagged with `uci` from OpenML, processes the data, and saves it in MAT format:

```bash
python uci.py
```

#### 脚本主要功能 | Main Features
- 从OpenML获取数据集列表。
- 逐一下载数据集，处理特征和标签。
- 将处理后的数据保存到 `./mat_datasets` 目录。

- Fetches dataset list from OpenML.
- Downloads datasets one by one, processing features and labels.
- Saves processed data in the `./mat_datasets` directory.

#### 代码片段 | Code Snippet
```python
# 获取带有 'uci' 标签的数据集 | Fetch datasets tagged with 'uci'
uci_datasets = openml.datasets.list_datasets(tag='uci', output_format='dataframe')

# 下载和保存为.mat文件 | Save as .mat file
savemat(mat_file_path, {'X': X, 'Y': y})
```

### 2. 运行 `datasets.py` | Run `datasets.py`

`datasets.py` 脚本会根据以下条件筛选数据集并保存：
- 实例数小于1000。
- 类别数大于1。

The `datasets.py` script filters datasets based on the following criteria and saves them:
- Number of instances less than 1000.
- Number of classes greater than 1.

运行脚本 | Run the script:

```bash
python datasets.py
```

#### 脚本主要功能 | Main Features
- 从OpenML获取所有数据集。
- 筛选满足条件的数据集。
- 保存为MAT格式文件至 `./datasets_less_than_1000` 目录。

- Fetches all datasets from OpenML.
- Filters datasets based on specified criteria.
- Saves them as MAT files in the `./datasets_less_than_1000` directory.

#### 代码片段 | Code Snippet
```python
# 过滤条件 | Filtering criteria
filtered_datasets = all_datasets[
    (all_datasets['NumberOfInstances'] < 1000) &
    (all_datasets['NumberOfClasses'] > 1)
]

# 保存为.mat文件 | Save as .mat file
savemat(mat_file_path, {'X': X, 'Y': y})
```

## 输出格式 | Output Format
处理后的数据集以 `.mat` 文件保存，包含以下内容：
- `X`: 数据集的特征矩阵，已转为数值类型并填充缺失值。
- `Y`: 数据集的标签，已转为数值类型。

Processed datasets are saved as `.mat` files, containing:
- `X`: Feature matrix converted to numeric type and with missing values filled.
- `Y`: Labels converted to numeric type.

## 示例输出 | Example Output
以下是生成的MAT文件目录示例：

Below is an example of the generated MAT file directory:

```plaintext
mat_datasets/
├── dataset1_100.mat
├── dataset2_200.mat
...

datasets_less_than_1000/
├── dataset3_500.mat
├── dataset4_300.mat
...
```

## 常见问题 | FAQs

1. **如何更改保存目录？ | How to change the save directory?**
   - 修改脚本中的 `output_dir` 变量即可。
   - Modify the `output_dir` variable in the script.

2. **如何更改筛选条件？ | How to change the filtering criteria?**
   - 修改 `datasets.py` 中 `filtered_datasets` 的过滤逻辑。
   - Modify the filtering logic in `filtered_datasets` in `datasets.py`.

3. **运行脚本时遇到错误？ | Errors while running the script?**
   - 检查依赖库是否已安装，或确认网络连接。
   - Check if the required libraries are installed or ensure a stable network connection.

## 贡献 | Contribution
欢迎提交问题或提出改进建议！

Contributions and suggestions are welcome!

## 许可证 | License
本项目基于 [MIT License](LICENSE)。

This project is licensed under the [MIT License](LICENSE).
