# -*- coding: utf-8 -*-
from src.models.base import BaseModel
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score
import pickle


class LGBMClassifier(BaseModel):

    def __init__(self, trainset, save_path=None, load_path=None) -> None:
        super().__init__()
        self.model = None
        self.save_path = save_path
        if trainset is None:
            self._load_model(load_path)
        else:
            self.dtrain = lgb.Dataset(trainset.X_train, label=trainset.y_train)
            self.deval = lgb.Dataset(trainset.X_val, label=trainset.y_val)
            self.X_test = trainset.X_test
            self.y_test = trainset.y_test

            """
            模型params参考：https://lightgbm.readthedocs.io/en/latest/Parameters.html
            """
            task_param = {
                'boosting': 'goss',  # 可选："gbdt", "rf", "dart", "goss"
                'metric': ''
            }
            self.num_class = len(set(trainset.y_train))
            if self.num_class == 2:
                task_param.update({
                    'objective': 'binary'
                })
            elif self.num_class > 2:
                task_param.update({
                    'objective': 'multiclass',
                    'num_class': self.num_class
                })
            else:
                raise Exception(f"Class num of training set ('{self.num_class}') < 2")
            tree_param = {
                'device_type': 'cpu',
                'learning_rate': 0.1,  # 学习率
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
        train_config = {
            'num_boost_round': 2000,  # 迭代轮次上限
            'valid_sets': [self.deval],
            'callbacks': [lgb.early_stopping(stopping_rounds=25)],  # 早停计数轮次
        }
        self.model = lgb.train(self.booster_params, self.dtrain, **train_config)
        if self.save_path is not None:
            self._save_model(self.save_path)

    def val(self):
        y_prob = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
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
        y_prob = self.model.predict(features, num_iteration=self.model.best_iteration)
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
        y_prob = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        if self.num_class > 2:
            y_pred = np.argmax(y_prob, axis=1)
        else:
            delta = 0.5 - binary_threshold
            y_pred = np.round(np.clip(y_prob + delta, 0, 1))
        res = confusion_matrix(self.y_test, y_pred) / len(self.y_test)
        return res