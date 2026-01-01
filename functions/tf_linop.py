import tensorflow as tf
class AOperator(tf.linalg.LinearOperator):
    def __init__(self, A_matvec, n, dtype=tf.float32):
        self._A_matvec = A_matvec
        self._n = n
        super().__init__(
            dtype=dtype,
            is_self_adjoint=True,
            is_positive_definite=True,
            is_square=True
        )

    def _shape(self):
        return tf.TensorShape([self._n, self._n])
        #return tf.TensorShape([self._n])

    def _shape_tensor(self):
        return tf.constant([self._n, self._n], dtype=tf.int32)
        #return tf.constant([self._n], dtype=tf.int32)
    def _matvec(self, x,adjoint=False):
        # x shape: [..., n]
        return self._A_matvec(x)
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if adjoint or adjoint_arg:
            raise NotImplementedError("Adjoint not supported")
        return self._A_matvec(x)
