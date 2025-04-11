"""
date = 20250202
author = zjy
todo = 载入多表格数据, 完成多表格学习, 习得跨表数据的统一表示
    预训练: 认识表格、发现表间相关性
    任务1: 认识表格
    支持 子表载入
    支持 不同顺序的子表载入, 使得编码器能认识所有相似、相关的表的含义, 不会因为顺序影响.
    支持 数据重构学习, 掩盖掉一定的数据(通过随机替换的表中的值), 网络学习表示后能正确预测原先的值, 包括 分类、回归
    支持 列类型判断: 需要有列级属性预测; 进行表级编码后, 用列级表示来预测 (能认出位于不用表的相同列)
    支持 子表预测 只用部分列, 预测整行分类
    
    任务2: 发现表间相关性
    支持  一次载入两个来自不同表格的子表
    *支持 列级对比学习, 当两列有足够的相关性构建为正例, 能帮助发现两表之间的联系。 认识列级联系。
    *支持 行级对比学习, 判断输入的两个表格的行是否对应同一个实体。 认识行级联系。 (前提是能进行主外键连接。)

    微调: 在其中一个表上, 进行任务微调
    支持 按batch载入子表, 对预测任务进行微调。

    不微调直接零样本预测: 在其中一个表上, 直接预测结果
"""

import openml
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from collections import namedtuple
import random 
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.init as nn_init
from torch import Tensor
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
import os
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from options import get_opt

from multi_table_transformer import AtcaNet
import torch.optim as optim
from tqdm import tqdm

"""
数据处理: 给定一组表格
预处理
1. 缺失值：数值型补充平均值, 类别型补充众数.
2. 数值型进行归一化

设置标签
1. 记录全部列名, 同时存储多分类编号
2. 记录每个类别型列的全部类型, 同时存储相应多分类编号  or 二分类编号
3. 记录实体序号, 如果存在实体对应关系
"""

"""
编码处理:
由于方案1有现成代码, 先用方案1
方案1: bert直接将单元格编码成表示
1. 数值型, 与bert编码的表示相乘
2. 类别型, 采用bert编码的表示, 与[cls] token一起取平均值

方案2: 只用bert的词表
1. 数值型, 与列名对应的可学习嵌入相乘, 并与可学习bias相加
2. 类别型, 与列名, bert编码的表示一起组成句子, 与可学习嵌入进行查找转换为表示, 再处理为单元格表示
方案2的难点, 要设计批量载入的方案, 
数值型要拼成子表载入, 同时要维护一个列表来表示每个数值型对应的列名是什么。 
类别型要先拼成句子, 经过padding后载入, 同时要维护一个标记, 表示哪几个嵌入对应那个类别, 才能获取列级表示, 或者每个单元格前面也加上cls token。
"""

"""
载入处理：
重构数据载入
1. 随机一个表, 随机选位置, 载入不同大小的子表.
2. 对这个子表进行随机排列, 随机生成一张一定比例的掩码表, 与原先的位置进行替换； 生成另一张掩码表, 用[mask]与对应位置替换
3. 对[MASK]或损坏的部分, 生成标签表, 最后对不同的列进行判断

对比数据载入
1. 构建列级正负例对, 只需任意载入
2. 行级正负例对, 需要将表1和表2相同的实体序号进行载入
"""

Tensor_type = torch.float32
device = "cuda:5"
# 定义一个数据类来存储数据集的信息
DatasetInfo = namedtuple('DatasetInfo', ['table_name','table_embs', 'columns_name_list', 'columns_embs_list', 'columns_types_list', 'catvalues_embs_dict'])

def set_random_seed(seed=42):
    """
    设置 NumPy 和 PyTorch 的全局随机数种子。
    """
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机数种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU，还需要设置 CUDA 的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置 PyTorch 的其他随机性相关选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)


#给定openml数据集id，返回datasetInfo
class SingleTableProcessor():
    def __init__(self, dataset_path, cut=None):
        """
        数据处理: 给定一组表格
        预处理
        1. 缺失值：数值型补充平均值, 类别型补充众数.
        2. 数值型进行归一化

        设置标签
        1. 记录全部列名, 同时存储多分类编号
        2. 记录每个类别型列的全部类型, 同时存储相应多分类编号  or 二分类编号
        3. 记录实体序号, 如果存在实体对应关系
        """
        """
        输入: 单表地址
        输出: 预处理的单表, 单表对应的全部列名. 每个类别列对应的类别名字典, 
        """
        dataset_file_name = os.path.basename(dataset_path)
        self.data_name, ext = os.path.splitext(dataset_file_name)
        if ext.lower() == '.csv':
            self.dataset = pd.read_csv(dataset_path, skipinitialspace=True)
        elif ext.lower() == '.arff':
            data, _ = arff.loadarff(dataset_path)
            dataset = pd.DataFrame(data)
            self.dataset = dataset.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        if cut!=None:
            self.dataset = self.dataset.iloc[:cut]      # :x表示[0,x)区间     

        # 将剩下的部分作为特征
        self.X = self.dataset.iloc[:, :-1]

        # 将最后一列提取为 target
        self.y = self.dataset.iloc[:, -1]

        # 处理数据集
        self.X, self.y, self.column_names_list, self.cat_column_names, self.num_column_names = self._process_dataset()

    def get_column_type(self, threshold=10):
        """
        根据列的类型或者唯一值的数量生成categorical_indicator。
        如果列的类型是object或类别数量小于阈值, 则认为是分类特征。
        """
        column_names_list = self.X.columns.tolist()
        cat_column_names = []
        num_column_names = []
        for col in self.X.columns:
            # 判断数据类型是否为类别数据（例如：object 或 category）
            if self.X[col].dtype == 'object' or len(self.X[col].unique()) < threshold:
                cat_column_names.append(col)  # 分类特征
            else:
                num_column_names.append(col)# 数值特征
        return column_names_list, cat_column_names, num_column_names


    def _process_dataset(self):
        """
        处理数据集, 提取表名、列名、数据类型, 并转换为BERT嵌入。
        """
        #获得列名, 列类别
        column_names_list, cat_column_names, num_column_names= self.get_column_type()

        #根据列类别进行预处理
        #类别列填充众数
        if cat_column_names != []:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.X[cat_column_names] = imputer_cat.fit_transform(self.X[cat_column_names])

        if num_column_names != []:
            #数值列填充均值        
            imputer_num = SimpleImputer(strategy='mean')
            self.X[num_column_names] = imputer_num.fit_transform(self.X[num_column_names])
            #归一化
            scaler = MinMaxScaler()
            self.X[num_column_names] = scaler.fit_transform(self.X[num_column_names])

        #去除缺失值
        self.X = self.X.dropna(axis=0)
        self.y = self.y[self.X.index]
        self.y = LabelEncoder().fit_transform(self.y.values)

        return self.X, self.y, column_names_list, cat_column_names, num_column_names
    
    def get_table_info(self):
        return self.X, self.y, self.column_names_list, self.cat_column_names, self.num_column_names


class TablesProcessor():
    def __init__(self, tables_paths, split_ratio=(0.7, 0.15, 0.15), cut=None):
        """
        将多个表格的信息整合在一块儿; 
        一是 整合所有列名, 用对应的数字标记, 方便重构预测;
        二是 为每个类别列生成数字标记, 方便重构预测, 对于跨表相同的列, 只生成一组通用的数字标记; 
        三是 为每个表格生成一组由数字标记的映射, 方便直接取数
        四是 将X变为bert嵌入, 以单元格为单位
        五是 生成表格信息字典, 方便组成数据集。
        """
        self.tables_paths = tables_paths
        self.split_ratio = split_ratio
        # 验证split_ratio的合法性
        assert len(split_ratio) == 3, "split_ratio必须包含三个元素"
        assert sum(split_ratio) == 1.0, "split_ratio总和必须为1"
        assert all(r >= 0 for r in split_ratio), "split_ratio中的值不能为负"


        self.table_info_dict = {}  # 最终输出的表格信息字典
        all_col_names = []         # 存储所有列名
        all_col_cats = {}          # 存储所有分类列的唯一值
        
        # 第一次遍历: 收集所有列名和分类列唯一值
        self.tables = []
        for t_path in tables_paths:
            # 处理单个表格
            single_table = SingleTableProcessor(t_path, cut)
            X, y, column_names, cat_cols, num_cols = single_table.get_table_info()
            self.tables.append((X, y, column_names, cat_cols, num_cols))
            
            # 收集列名
            all_col_names.extend(column_names)
            
            # 合并分类列唯一值
            for col in cat_cols:
                curr_values = X[col].unique().tolist()
                if col not in all_col_cats:
                    all_col_cats[col] = curr_values
                else:
                    all_col_cats[col] = list(set(all_col_cats[col] + curr_values))

        # 任务一: 生成列名到数字的全局映射
        all_col_names = list(set(all_col_names))  # 去重
        self.col_names_label_dict = {name: idx for idx, name in enumerate(all_col_names)}
        
        # 任务二: 生成分类列值到数字的全局映射
        self.cat_label_mapping = {}
        for col, values in all_col_cats.items():
            # 去重并排序保证编码一致性
            sorted_values = sorted(list(set(values)))
            self.cat_label_mapping[col] = {v: i for i, v in enumerate(sorted_values)}
        

        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained('/home/zhangjunyu/zjywork/zjycode/VPCL-main/multi_table_bert_base_reconstruct/tokenizer_bert-base-uncased')
        self.model = BertModel.from_pretrained('/home/zhangjunyu/zjywork/zjycode/VPCL-main/multi_table_bert_base_reconstruct/tokenizer_bert-base-uncased').to(device)
        self.model.eval()  # 设置模型为评估模式(禁止反向传播参数更新)

        # 任务四: 将X变为bert嵌入, 获得每个列名, 类别, 特殊token对应映射
        self.col_name_emb_mapping, self.col_types_emb_mapping, spec_token_mapping = self._get_bert_embedding(all_col_names, all_col_cats)


        # 第二遍遍历: 生成每个表格的映射
        for idx, t_path in enumerate(tables_paths):
            X, y, _, cat_cols, num_cols = self.tables[idx]
            
            # 处理分类列编码
            X_processed = X.copy()
            for col in cat_cols:
                X_processed[col] = X_processed[col].map(self.cat_label_mapping[col])
            
            # 生成列名到数字的映射
            column_mapping = {
                col: self.col_names_label_dict[col]
                for col in X_processed.columns
            }
            X_processed.rename(columns=column_mapping, inplace=True)
            X_processed_colname = X_processed.columns

            # 生成分类列和数值列的数字ID列表
            cat_ids = [self.col_names_label_dict[col] for col in cat_cols]
            num_ids = [self.col_names_label_dict[col] for col in num_cols]
            
            # 任务四: 获取X的嵌入版本
            X_bert, X_bert_col = self.get_X_bert(X, cat_cols, num_cols)
            
            # 任务五: 填充表格信息字典, 划分训练,验证,测试集合
            # 划分数据集
            indices = np.arange(len(X_processed))
            # y_array = y.values  # 转换为numpy数组
            train_ratio, valid_ratio, test_ratio = self.split_ratio
            
            # 第一次划分训练集和剩余集
            X_train_idx, X_rem_idx, y_train, y_rem = train_test_split(
                indices, y, train_size=train_ratio, random_state=42, stratify=y
            )
            
            # 计算剩余部分中验证集的比例
            remaining_ratio = valid_ratio + test_ratio
            valid_size = valid_ratio / remaining_ratio
            
            # 第二次划分验证集和测试集
            X_valid_idx, X_test_idx, y_valid, y_test = train_test_split(
                X_rem_idx, y_rem, train_size=valid_size, random_state=42, stratify=y_rem
            )
            
            # 划分X
            X_train = X.iloc[X_train_idx].to_numpy()
            X_valid = X.iloc[X_valid_idx].to_numpy()
            X_test = X.iloc[X_test_idx].to_numpy()
            
            # 划分X_bert，假设X_bert是数组
            X_bert_train = X_bert[X_train_idx]
            X_bert_valid = X_bert[X_valid_idx]
            X_bert_test = X_bert[X_test_idx]
            
            # # 转换y为Series，保持索引
            # y_train = np.array(y.iloc[X_train_idx])
            # y_valid = np.array(y.iloc[X_valid_idx])
            # y_test = np.array(y.iloc[X_test_idx])

            # 划分X_process
            X_processed_train = X_processed.iloc[X_train_idx].to_numpy()
            X_processed_valid = X_processed.iloc[X_valid_idx].to_numpy()
            X_processed_test = X_processed.iloc[X_test_idx].to_numpy()

            self.table_info_dict[t_path] = {
                "allset":{
                        'X': X,
                        'X_bert': X_bert,           #包含值
                        'X_bert_col': X_bert_col,       #包含列名嵌入
                        'y': y,
                        'X_mapping': X_processed,
                        'X_map_colname': X_processed_colname,
                        'Special_tokens': spec_token_mapping,
                        'row_len': len(y),
                        'column_len': len(X.iloc[0])
                        # 'column_mapping': column_mapping,       #每列使用第几个分类/回归头
                        # 'categorical_columns': cat_ids,     
                        # 'numerical_columns': num_ids
                },
                "train_set":{
                        'X': X_train,
                        'X_bert': X_bert_train,           #包含值和列名嵌入两部分
                        'X_bert_col': X_bert_col,       #包含列名嵌入
                        'y': y_train,
                        'X_mapping': X_processed_train, 
                        'X_map_colname': X_processed_colname,
                        'Special_tokens': spec_token_mapping,
                        'row_len': len(y_train),
                        'column_len': len(X_train[0])
                },
                "valid_set":{
                        'X': X_valid,
                        'X_bert': X_bert_valid,           #包含值和列名嵌入两部分
                        'X_bert_col': X_bert_col,       #包含列名嵌入
                        'y': y_valid,
                        'X_mapping': X_processed_valid,
                        'X_map_colname': X_processed_colname,
                        'Special_tokens': spec_token_mapping,
                        'row_len': len(y_valid),
                        'column_len': len(X_valid[0])
                },
                "test_set":{
                        'X': X_test,
                        'X_bert': X_bert_test,           #包含值和列名嵌入两部分
                        'X_bert_col': X_bert_col,       #包含列名嵌入
                        'y': y_test,
                        'X_mapping': X_processed_test,
                        'X_map_colname': X_processed_colname,
                        'Special_tokens': spec_token_mapping,
                        'row_len': len(y_test),
                        'column_len': len(X_test[0])
                }
            }

            #任务六: 用列表存放每列的预测头维度
            pred_head_dims = []
            for col in all_col_names:
                if col in all_col_cats.keys():
                    # 类别列的预测头维度为类别数量
                    pred_head_dims.append(len(all_col_cats[col]))
                else:
                    # 默认为 1（如果列不分类别或数值列）
                    pred_head_dims.append(1)

            self.common_info_dict = {
                "col_numbers":  len(all_col_names), #列类型共有多少种, 提供给模型做预测头
                "col_prd_dims":  pred_head_dims, #数值型记为1, 类别型几位对应数值, 提供给模型做预测头
                'Special_tokens': spec_token_mapping,
            }


    def _get_bert_embedding(self, all_col_names, all_col_cats):
        """
        使用BERT模型生成文本的嵌入向量。
        1. 获取每个列名的集合, 获取每个类别名的集合(列名_类别名), 组成一个集合
        2. 将这个集合直接用bert编码, 获取列名/类别名-->emb的映射
        *3. 对类别型和数值型单元格分别处理
        *4. 类别型列, 直接字典替换
        *5. 数值型列, 将数值x列名嵌入, 再进行替换
        """
        # 获取所有列名和分类列的类别值
        # all_texts = {"col_names": all_col_names}

        # for col_name, cat_types in all_col_cats.items():
        #     texts = [col_name + '_' + type for type in cat_types]
        #     all_texts[col_name] = texts
        
        # 处理列名和分类值嵌入
        text_mappings = {"col_names": all_col_names}
        for col in all_col_cats:
            text_mappings[col] = [f"{col}_||_{val}" for val in all_col_cats[col]]  # 使用唯一分隔符

        # 对所涉及到的文本都进行编码
        all_embs = {}
        for name, texts in text_mappings.items():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True).to(device)
            with torch.no_grad():  # 禁用梯度计算
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                num_valid_tokens = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())                
                # 计算有效位置的总和
                sum_embeddings = (last_hidden_state * attention_mask).sum(dim=1)

                # 计算有效位置的平均嵌入表示
                embeddings = sum_embeddings / num_valid_tokens

            all_embs[name] = embeddings.to("cpu")


        col_name_emb_mapping = dict(zip(all_col_names, all_embs['col_names'])) # 列名bert嵌入
        col_types_emb_mapping = {} # 分类型列名+唯一值的bert嵌入
        for col_name, types in all_col_cats.items():
            col_types_emb_mapping[col_name] = dict(zip(types, all_embs[col_name]))
        
        # 特殊token处理
        special_tokens = {
            "[CLS]": self.model.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id].to(Tensor_type).to("cpu"),
            "[MASK]": self.model.embeddings.word_embeddings.weight[self.tokenizer.mask_token_id].to("cpu")
        }
        return col_name_emb_mapping, col_types_emb_mapping, special_tokens


    def get_X_bert(self, X, cat_cols, num_cols):
        """
        使用BERT模型生成文本的嵌入向量。

        *1. 获取每个列名的集合, 获取每个类别名的集合(列名_类别名), 组成一个集合
        *2. 将这个集合直接用bert编码, 获取列名/类别名-->emb的映射
        3. 对类别型和数值型单元格分别处理
        4. 类别型列, 直接字典替换
        5. 数值型列, 将数值x列名嵌入, 再进行替换
        """
        embedded_data = []

        for col in X.columns:
            if col in cat_cols:
                embedded_col = X[col].map(self.col_types_emb_mapping[col])
                embedded_data.append(torch.tensor(np.stack(embedded_col.values)))
            elif col in num_cols:
                embedded_data.append(self.col_name_emb_mapping[col].unsqueeze(0).expand(torch.tensor(X[col]).shape[0],-1) * torch.tensor(X[col]).unsqueeze(1))


        X_bert_value_emb = torch.stack(embedded_data).transpose(0,1).to(Tensor_type).to(device)
        X_bert_col_embs = torch.tensor(np.stack(X.columns.map(self.col_name_emb_mapping))).unsqueeze(0).to(Tensor_type).to(device)

        return X_bert_value_emb, X_bert_col_embs


    def get_processed_tables(self):
        """获取处理后的所有表格信息"""
        return self.table_info_dict, self.common_info_dict

class  Singledataset(Dataset):
    def __init__(self, opt, ds_info_dict):
        self.opt = opt
        self.ds_info_dict = ds_info_dict

        self.length = self.ds_info_dict["row_len"]

        self.columns = list(self.ds_info_dict["X_map_colname"])
        self.special_tokens = self.ds_info_dict["Special_tokens"]
        self.mask_embedding = self.special_tokens["[MASK]"].to(device)
        self.cls_emb = self.special_tokens["[CLS]"].to(device)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        """
        "train_set":{
            'X': X_train,
            'X_bert': X_bert_train,           #包含值和列名嵌入两部分
            'X_bert_col': X_bert_col,       #包含列名嵌入
            'y': y_train,
            'X_mapping': X_processed_train, 
            'X_map_colname': X_processed_colname,
            'Special_tokens': spec_token_mapping,
            'row_len': len(y_train),
            'column_len': len(X_train.iloc[0])
        },
        
        返回
        batch_X_bert,
        batch_X_bert_col
        batch_y
        batch_X_mapping
        special_tokens
        """

        return self.ds_info_dict["X_bert"][idx], self.ds_info_dict["y"][idx], self.ds_info_dict["X_mapping"][idx]
    
    def valid_collate_fn(self, batch):

        # 解包原始数据
        X_bert = [item[0] for item in batch]
        y = [item[1] for item in batch]
        X_mapping = [item[2] for item in batch]

        # 转换为张量
        X_bert = torch.stack(X_bert)
        y = torch.tensor(y)
        X_mapping = torch.tensor(np.array(X_mapping))

        device = X_bert.device
        # 验证集不添加任何掩码
        combined_mask = torch.zeros_like(X_mapping, dtype=torch.float32)

        # ====== 保持结构一致性 ======
        # 添加CLS标记（与训练时相同的结构）
        
        # 行方向添加CLS
        cls_row = self.cls_emb.view(1, 1, -1).expand(X_bert.size(0), 1, -1).to(device)
        X_bert_augmented = torch.cat((cls_row, X_bert), dim=1)
        
        # 列方向添加CLS
        cls_col = self.cls_emb.view(1, 1, -1).expand(1, X_bert_augmented.size(1), -1).to(device)
        X_bert_augmented = torch.cat((cls_col, X_bert_augmented), dim=0)

        return {
            "X_masked_bert": X_bert_augmented,  # 注意实际未做mask
            "y_gt": y,
            "col_ids": self.columns,
            "X_map_gt": X_mapping,
            "X_mask": combined_mask,  # 全零掩码
        }

    def recon_collate_fn(self, batch):
        X_bert = [item[0] for item in batch]
        y =  [item[1] for item in batch]
        X_gt =  [item[2] for item in batch]

        X_bert = torch.stack(X_bert)
        device = X_bert.device
        row_len, column_len, dim = X_bert.shape

        y = torch.tensor(y)
        X_gt = torch.tensor(np.array(X_gt))

        #随机选2-全部列
        num_columns_to_select = np.random.randint(2, column_len + 1)
        selected_columns = np.random.choice(column_len, num_columns_to_select, replace=False)        

        # 提取选定的列
        X_bert = X_bert[:, selected_columns, :]
        X_gt = X_gt[:, selected_columns]        
        columns_id = [self.columns[_] for _ in selected_columns]

        #生成和X_mapping一样的损坏掩码, 处理X_bert
        #生成和X_mapping一样的缺失掩码, 处理X_bert
        #将两个掩码合并, 作为预测位置


        # 2. 生成随机缺失掩码
        missing_mask = np.random.binomial(
            1, self.opt.miss_mask_prob, size=X_gt.shape
        )

        # ====== 处理X_bert输入 ======
        # 获取MASK标记的嵌入向量

        # 扩展掩码维度用于嵌入替换
        missing_mask_tensor = torch.tensor(missing_mask, dtype=torch.bool).to(device)  # (B, cols)
        mask_expanded = missing_mask_tensor.unsqueeze(-1)  # (B, cols, 1)
        mask_embedding = self.mask_embedding.view(1, 1, -1).to(device)  # (1, 1, emb_dim)

        # 使用torch.where进行向量化替换
        X_bert_masked = torch.where(
            mask_expanded.expand_as(X_bert),
            mask_embedding.expand_as(X_bert),
            X_bert
        )


        # ====== 掩码生成逻辑 ======
        # 1. 生成随机错位掩码, 用表中其他位置的值代替
        corruption_mask = np.random.binomial(
            1, self.opt.cor_mask_prob, size=X_gt.shape
        )


        indices_row = torch.randperm(row_len)  # 随机打乱行索引
        indices_column = torch.randperm(num_columns_to_select)  # 随机打乱列索引
        X_bert_shuffled = X_bert[indices_row, :, :][:, indices_column, :]

        # 将 corruption_mask 转换为 torch.Tensor
        corruption_mask_tensor = torch.tensor(corruption_mask, dtype=torch.bool).to(device)
        X_bert[corruption_mask_tensor] = X_bert_shuffled[corruption_mask_tensor]




        # 3. 合并掩码（OR操作）
        combined_mask = corruption_mask_tensor | missing_mask_tensor


        #在X_bert_masked左边加一列cls token, 再上面也加一列cls token, 分别用于预测列类型和行类型, 同时,X_mapping也要补一格
        #左边先加一列

        cls_row = self.cls_emb.view(1, 1, -1).expand(X_bert_masked.size(0), 1, -1)
        X_bert_masked = torch.cat((cls_row, X_bert_masked), dim=1)

        #顶部添加一行
        cls_col = self.cls_emb.view(1, 1, -1).expand(1, X_bert_masked.size(1), -1)  # (1, cols+1, emb_dim)
        X_bert_masked = torch.cat((cls_col, X_bert_masked), dim=0)  # (B+1, cols+1, emb_dim)



        return {
            "X_masked_bert": X_bert_masked,
            "y_gt": y,
            "col_ids": columns_id,
            "X_map_gt": X_gt,
            "X_mask": combined_mask,
        }

from sklearn.metrics import roc_auc_score
import numpy as np
# 在模型定义后添加评估函数
def evaluate(model, dataloaders_dict, criterion, device, num_classes=2):
    """
    增强版评估函数，支持：
    - 分数据集统计指标
    - 多分类AUC计算
    - 自动处理异常情况
    """
    model.eval()
    results = {}
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for dset_name, dl in dataloaders_dict.items():
            dset_labels = []
            dset_probs = []
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            for batch in tqdm(dl, desc=f"Eval {dset_name}", leave=False):
                # 数据转换
                X = batch['X_masked_bert'].to(Tensor_type).to(device)
                y = batch['y_gt'].to(device)
                
                # 前向传播
                logits, _, _, _, _ = model(X, batch['col_ids'])
                loss = criterion(logits, y)
                
                # 统计指标
                total_loss += loss.item() * y.size(0)
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == y).sum().item()
                total_samples += y.size(0)
                
                # 收集概率和标签
                probs = torch.softmax(logits, dim=1) if num_classes > 2 \
                        else torch.sigmoid(logits[:, 1])
                dset_probs.append(probs.cpu().numpy())
                dset_labels.append(y.cpu().numpy())

            # 计算数据集指标
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            
            # 计算AUC
            try:
                auc = roc_auc_score(
                    np.concatenate(dset_labels),
                    np.concatenate(dset_probs),
                    multi_class='ovr' if num_classes > 2 else 'raise'
                )
            except ValueError as e:
                print(f"AUC计算失败({dset_name}): {str(e)}")
                auc = float('nan')
            
            # 保存结果
            results[dset_name] = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'auc': auc
            }
            
            # 收集全局统计
            all_labels.append(np.concatenate(dset_labels))
            all_probs.append(np.concatenate(dset_probs))

    # 计算全局指标
    try:
        global_auc = roc_auc_score(
            np.concatenate(all_labels),
            np.concatenate(all_probs),
            multi_class='ovr' if num_classes > 2 else 'raise'
        )
    except ValueError as e:
        print(f"全局AUC计算失败: {str(e)}")
        global_auc = float('nan')
    
    return results, global_auc



if __name__=="__main__":
    # #随机获得一个来自id的子表    
    # test_dataset_processor(31)

    #给定一个列表，定义数据集
    # table_name, column_name_list, set_M, set_N, emb_list = test_tabulardataset([31, 29])
    # _ = test_tabulardataset([29])

    opt = get_opt()

    # loader_data()
    paths = ["/home/zhangjunyu/mhwork/Mutitable-Recon/multi_table_bert_base_reconstruct_fusion/data/data1/first_part.csv",
"/home/zhangjunyu/mhwork/Mutitable-Recon/multi_table_bert_base_reconstruct_fusion/data/data1/first_part.csv"]    
    Tables = TablesProcessor(paths, cut=1000)
    tables_dict, common_dict = Tables.get_processed_tables()
    #划分训练/验证/测试dict
    train_sets, valid_sets, test_sets = {}, {}, {}
    for dset, info in tables_dict.items():
        train_sets[dset] = info["train_set"]
        valid_sets[dset] = info["valid_set"]
        test_sets[dset] = info["test_set"]

    #重构训练载入
    re_dataset = {}
    for tb, info in train_sets.items():
        re_dataset[tb] = Singledataset(opt, info)

    re_dataloader = {}
    for tb, dset in re_dataset.items():
        re_dataloader[tb] = DataLoader(re_dataset[tb], 
                                       batch_size=opt.batch_size, 
                                       collate_fn=re_dataset[tb].recon_collate_fn, 
                                       shuffle=True)
    #验证集, 测试集载入
    val_dataset = {}
    for tb, info in valid_sets.items():
        val_dataset[tb] = Singledataset(opt, info)
    val_dataloader = {}
    for tb, dset in val_dataset.items():
        val_dataloader[tb] = DataLoader(dset, 
                                        batch_size=opt.batch_size, 
                                        collate_fn=dset.valid_collate_fn,
                                        shuffle=False)
        
    test_dataset = {}
    for tb, info in test_sets.items():
        test_dataset[tb] = Singledataset(opt, info)
    test_dataloader = {}
    for tb, dset in test_dataset.items():
        test_dataloader[tb] = DataLoader(dset, 
                                        batch_size=opt.batch_size, 
                                        collate_fn=dset.valid_collate_fn,
                                        shuffle=False)


    """
        self.common_info_dict = {
        "col_numbers":  len(all_col_names), #列类型共有多少种, 提供给模型做预测头
        "col_prd_dims":  pred_head_dims, #数值型记为1, 类别型几位对应数值, 提供给模型做预测头
        'Special_tokens': spec_token_mapping,
    }
    """
    model = AtcaNet(common_dict["col_numbers"], common_dict["col_prd_dims"])
    model.to(opt.device)

    criterion_rclf = nn.CrossEntropyLoss()  # 二分类损失函数
    optimizer = optim.AdamW(model.parameters(), 
                        lr=0.0001, 
                        weight_decay=0.00001)

    opt.epochs = 10
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    best_val_auc = 0.0
    best_model_path = "best_model.pth"
    for epoch in range(opt.epochs):
        model.train()

        # 为每个dataloader创建新迭代器
        dataloader_iters = [iter(dl) for dl in re_dataloader.values()]
        total_batches = sum(len(dl) for dl in re_dataloader.values())

        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{opt.epochs}") as pbar:
            total_loss = 0.0
            finished = [False] * len(dataloader_iters)
            
            # 循环直到所有dataloader耗尽
            while not all(finished):
                for i, dl_iter in enumerate(dataloader_iters):
                    if finished[i]:
                        continue
                    try:
                        batch = next(dl_iter)
                    except StopIteration:
                        finished[i] = True
                        continue
                    
                    # 数据类型转换
                    X_masked_bert = batch['X_masked_bert'].to(Tensor_type).to(opt.device)
                    y_gt = batch['y_gt'].to(opt.device)
                    col_ids = batch['col_ids']
                    X_map_gt = batch['X_map_gt'].to(opt.device)
                    X_mask = batch['X_mask'].to(opt.device)

                    # 混合精度训练
                    with torch.cuda.amp.autocast():
                        row_cls = model(X_masked_bert, col_ids)
                        loss = criterion_rclf(row_cls, y_gt)

                    # 反向传播优化
                    optimizer.zero_grad()  # 清空梯度
                    # 梯度裁剪
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                    loss.backward()  # 反向传播计算梯度      
                    optimizer.step()  # 更新模型参数

                    # 统计指标
                    total_loss += loss.item()
                    avg_loss = total_loss / (pbar.n + 1)  # 使用进度条的当前计数
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    pbar.update(1)

            # 更新学习率
            scheduler.step()
            
            # 打印epoch统计
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        # ================= 验证阶段 =================
        # 获取详细评估结果和全局AUC
        val_results, val_global_auc = evaluate(
            model, 
            val_dataloader, 
            criterion_rclf, 
            opt.device,
            num_classes=2  # 根据任务修改类别数
        )
        
        # 打印详细验证结果
        print(f"\n=== Epoch {epoch+1} 验证结果 ===")
        print(f"全局验证AUC: {val_global_auc:.4f}")
        for dset_name, metrics in val_results.items():
            print(f"{dset_name}:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Acc: {metrics['accuracy']:.2%}")
            print(f"  AUC: {metrics['auc']:.4f}")

        # 保存最佳模型（基于全局AUC）
        if val_global_auc > best_val_auc:
            best_val_auc = val_global_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"发现新的最佳模型，验证AUC: {val_global_auc:.4f}")

    # ================= 最终测试阶段 =================
    print("\n=== 最终测试 ===")
    model.load_state_dict(torch.load(best_model_path))
    test_results, test_global_auc = evaluate(
        model,
        test_dataloader,
        criterion_rclf,
        opt.device,
        num_classes=2
    )

    # 打印完整测试结果
    print("\n=== 测试结果 ===")
    print(f"全局测试AUC: {test_global_auc:.4f}")
    for dset_name, metrics in test_results.items():
        print(f"{dset_name}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Acc: {metrics['accuracy']:.2%}")
        print(f"  AUC: {metrics['auc']:.4f}")








    pass