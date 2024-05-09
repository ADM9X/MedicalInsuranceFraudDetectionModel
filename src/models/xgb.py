# -*- coding: utf-8 -*-
import pickle

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score

from src.models.base import BaseModel


class XgboostClassifier(BaseModel):

    def __init__(self, trainset, save_path=None, load_path=None) -> None:
        """
        :params trainset: 训练数据集, 如果需要加载已训好的模型, 则trainset赋None, 模型地址填入load_path
        :params save_path: 模型训练完毕后的存储位置, 默认为: None(不保存)
        :params load_path: 如果trainset为None, 则从load_path处加载模型
        """
        super().__init__()
        self.model = None
        if trainset is None:
            self._load_model(load_path)
        else:
            self._training_config(trainset, save_path)

    def _training_config(self, data, save_path):
        self.save_path = save_path
        self.dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
        self.deval = xgb.DMatrix(data.X_val, label=data.y_val)
        self.dtest = xgb.DMatrix(data.X_test, label=data.y_test)
        self.y_test = data.y_test

        task_param = {
            'booster': 'gbtree',  # 基模型类型，可选'gblinear'，'dart'或'gbtree'
        }
        self.num_class = len(set(data.y_train))
        if self.num_class == 2:
            task_param.update({
                'objective': 'binary:logistic'
            })
        elif self.num_class > 2:
            task_param.update({
                'objective': 'multi:softprob',
                'num_class': self.num_class
            })
        else:
            raise Exception(f"Class num of training set ('{self.num_class}') < 2")
        tree_param = {
            'tree_method': 'hist',  # 如果有gpu可选'gpu_hist'
            'eta': 0.1,  # 学习率
            'max_depth': 6  # 最大树深
        }
        self.booster_params = {**task_param, **tree_param}

    def _save_model(self, file_path):
        data = {
            'num_class': self.num_class,
            'model': self.model
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_model(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)

    def fit(self):
        if not getattr(self, "booster_params"):
            raise Exception("Please init class with 'start_from_training()'")
        train_config = {
            'num_boost_round': 2000,  # 迭代轮次上限
            'evals': [(self.dtrain, 'train'), (self.deval, 'eval')],
            'verbose_eval': 25,  # 多少轮迭代后打印一次模型验证指标
            'early_stopping_rounds': 25,  # 早停计数轮次，验证集指标在n轮迭代中无进展则停止训练
        }
        self.model = xgb.train(self.booster_params, self.dtrain, **train_config)
        if self.save_path is not None:
            self._save_model(self.save_path)

    def val(self):
        if getattr(self.model, "best_iteration"):
            y_prob = self.model.predict(self.dtest, iteration_range=(0, self.model.best_iteration + 1))
        else:
            y_prob = self.model.predict(self.dtest)
        auc_params = {}
        if self.num_class > 2:
            auc_params.update({
                "multi_class": "ovr",
                "average": "weighted"
            })
        auc = roc_auc_score(self.y_test, y_prob, **auc_params)
        if self.num_class > 2:
            y_pred = y_prob.argmax(axis=1)
        else:
            y_pred = y_prob.round()
        precision = precision_score(self.y_test, y_pred, average='weighted')
        confusion_matrix = self.get_confusion_matrix()
        print(f"Testset Auc: {auc}")
        print(f"Testset Precision: {precision}")
        print(f"Confusion Matrix: \n{confusion_matrix}")
        return auc, precision, confusion_matrix

    def predict(self, features, prob=False):
        dinfer = xgb.DMatrix(features)
        if getattr(self.model, "best_iteration"):
            y_prob = self.model.predict(dinfer, iteration_range=(0, self.model.best_iteration + 1))
        else:
            y_prob = self.model.predict(dinfer)
        if self.num_class > 2:
            if not prob:
                y_prediction = np.argmax(y_prob, axis=1)
            else:
                y_prediction = y_prob.tolist()
        else:
            if not prob:
                y_prediction = np.around(y_prob)
            else:
                y_prediction = y_prob.tolist()
        return y_prediction

    def get_confusion_matrix(self, binary_threshold=0.5):
        """
        每行对应一个真实类别，每列对应一个预测类别
        例如：假设类别依次为"A", "B"，则第二行第一列代表“真实值B被预测为A的样本占比”
        """
        if getattr(self.model, "best_iteration"):
            y_prob = self.model.predict(self.dtest, iteration_range=(0, self.model.best_iteration + 1))
        else:
            y_prob = self.model.predict(self.dtest)
        if self.num_class > 2:
            y_pred = np.argmax(y_prob, axis=1)
        else:
            delta = 0.5 - binary_threshold
            y_pred = np.round(np.clip(y_prob + delta, 0, 1))
        res = confusion_matrix(self.y_test, y_pred) / len(self.y_test)
        return res

    def get_model_params(self):
        return self.model.save_config()
