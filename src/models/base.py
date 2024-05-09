# -*- coding: utf-8 -*-
import abc
import time


class BaseModel(metaclass=abc.ABCMeta):
    def training(self):
        self.fit()
        metrics = self.val()
        return metrics

    @abc.abstractmethod
    def fit(self):
        """
        拟合模型
        :return:
        """
        pass

    @abc.abstractmethod
    def val(self):
        """
        模型评估，返回结果指标
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """
        输入特征（新样本），返回预测结果
        :return:
        """
        pass

    def log(self, contents):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] {contents}")
