import numpy as np
from abc import ABC, abstractmethod

class LossFn(ABC):
    epsilon = 1e-12

    @abstractmethod
    def forward(self, exp_batch : np.ndarray, act_batch : np.ndarray, no_classes : int) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, exp_batch : np.ndarray, act_batch : np.ndarray, no_classes : int) -> np.ndarray:
        pass

    @staticmethod
    def one_hot_encoder(indexes : np.ndarray, no_classes : int):
        one_hot_encoded = np.zeros((indexes.size, no_classes))
        one_hot_encoded[np.arange(indexes.size), indexes] = 1

        return np.transpose(one_hot_encoded) 

class CrossEntropyLoss(LossFn):
    def forward(self, exp_batch, act_batch, no_classes):
        exp_batch = LossFn.one_hot_encoder(exp_batch, no_classes)
        act_batch = np.clip(act_batch, LossFn.epsilon, (1. - LossFn.epsilon))
        prob_dist = -np.mean(np.sum(exp_batch * np.log(act_batch), axis = 0, keepdims = True), axis = 0, keepdims = True)
        return prob_dist

    def backward(self, exp_batch, act_batch, no_classes):
        exp_batch = LossFn.one_hot_encoder(exp_batch, no_classes)
        return np.subtract(act_batch,  exp_batch)