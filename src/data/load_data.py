# -*- coding: utf-8 -*-
import time
from sklearn.model_selection import train_test_split
from collections import namedtuple
import pandas as pd
import os
import openpyxl


class DataLoader:
    def __init__(self,
                 input_file_path: str = None,
                 sep: str = None,
                 columns: list = None,
                 train_size: float = 0.7,
                 encoding: str = "utf-8"):
        """
        :param input_file_path: 数据集路径
        :param sep: 如果是.txt文件，则指定分隔符
        :param columns: 如果是.txt文件，则指定列名
        :param train_size: 拆分数据集时的训练集比例
        :param encoding: 文件编码格式
        """
        self.input_file_path = input_file_path
        self.sep = sep
        self.columns = columns
        self.train_size = train_size
        self.encoding = encoding

    def splitData(self):
        """
        数据集拆分
        :return: namedtuple
        """
        X, Y, cols = self.loadData()
        self.log("开始拆分数据集：")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=self.train_size, stratify=Y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
        result = namedtuple("splitDatasets",
                            ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "columns"])(
            X_train, y_train, X_val, y_val, X_test, y_test, cols)
        self.log("数据集拆分成功！")
        return result

    def loadData(self):
        """
        数据集加载
        :return:
        """
        file_type = self.determine_file_type()
        self.log("开始判断文件类型：")
        if file_type:
            if file_type == "CSV":
                self.log("开始解析CSV文件：")
                data = pd.read_csv(self.input_file_path, encoding=self.encoding, index_col=0)
                self.log("数据加载成功！")
            elif file_type == "Excel":
                self.log("开始解析Excel文件：")
                data = pd.read_excel(self.input_file_path)
                self.log("数据加载成功！")
            else:
                self.log("开始解析text文件：")
                data = pd.read_table(self.input_file_path, sep=self.sep, names=self.columns, encoding=self.encoding)
                self.log("数据加载成功！")
            X = data.drop("label", axis=1).values
            Y = data["label"].values
            cols = data.columns
            return X, Y, cols

    def determine_file_type(self):
        """
        输入文件类型判断
        :return:
        """
        if self.input_file_path:
            file_name, file_extension = os.path.splitext(self.input_file_path)
            if file_extension.lower() == '.csv':
                return 'CSV'
            elif file_extension.lower() in ['.xls', '.xlsx']:
                return 'Excel'
            elif file_extension.lower() == '.txt':
                return 'Text'
            else:
                raise TypeError("Only following file is needed:csv, excel, txt.")
        else:
            raise ValueError("Input File needed!")

    def log(self, contents=None):
        """
        日志打印
        :param contents:待打印的日志内容
        :return:
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] {contents}")


if __name__ == '__main__':
    loader = DataLoader(input_file_path="test/data.csv", encoding="gbk")
    result = loader.splitData()
    train_Y = result.X_train
    print(train_Y)
    print(type(train_Y))
    print(result.columns)
