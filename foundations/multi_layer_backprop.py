import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(W2, a1) + b2
        mse = (1/len(y_true)) * np.sum((z2 - y_true)**2)
        loss = float(np.round(mse, 4))

        dL_dz2 = (2/len(y_true)) * (z2 - y_true)
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2
        dL_da1 = dL_dz2 * W2
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1
        return {"loss": loss, "dW1": (np.round(dL_dW1, 4) + 0.0).tolist(), "db1": (np.round(dL_db1, 4).flatten() + 0.0).tolist(), "dW2": (np.round(dL_dW2, 4) + 0.0).tolist(), "db2": (np.round(dL_db2, 4).flatten() + 0.0).tolist()}
        # .tolist() to return in form of 2D list; .flatten().tolist() to return in form of 1D list
        # 0.0 was added to take care of the string matching being done for solution by platform; not actually required
