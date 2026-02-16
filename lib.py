import numpy as np

class Quantumket:

    # 波矢容器, 功能：归一化，计算模方
    def __init__(self, state: np.ndarray, internal_labels = None, external_types = None):
        # state: 波矢数据，要求为numpy数组，且维度至少为1
        # internal_labels: 内部标签，用于标识波矢的内部结构，如自旋态、能级等，要求为列表或数组，长度与state的第一维相同
        # external_types: 外部类型，用于标识波矢的外部结构，如momentum、position.
        self.data = state
        if internal_labels is not None:
            if len(internal_labels) != state.shape[0]:
                raise ValueError("internal_labels的长度必须与state的第一维相同")
            else:
                self.internal_labels = internal_labels
        if state.ndim == 1 and external_types is not None:
            raise ValueError("当state为一维数组时，external_types必须为None")
        
    def norm2(self):
        # 计算波矢的模方
        return np.vdot(self.data, self.data.conj()).real

    def normalize(self):
        # 归一化波矢
        n2 = self.norm2()
        if n2 > 0:
            self.data /= np.sqrt(n2)
            return self
        else:
            raise ValueError("波矢的模方为零，无法归一化")

    def __str__(self):
        return f"Quantumket(internal_labels={getattr(self, 'internal_labels', None)}, external_types={getattr(self, 'external_types', None)} \n data={self.data})"
