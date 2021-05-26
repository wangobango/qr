from typing import Tuple
import numpy as np

class QR:
    def __init__(self) -> None:
        pass
    """
    Checks if caulcuated matricies fulfill QR decomposition conditions, that is:
    A = QR , where Q -> Q * Qt = I and R is a upper triangular matrix
    """
    def __check_condition(self, Q: np.matrix, R: np.matrix) -> bool:
        if not np.allclose(R, np.triu(R)):
            print("R matrix is not upper traingle.")
            return False
        I = np.identity(Q.shape[1])
        comparison = np.equal(np.matmul(np.transpose(Q), Q), I)
        if not comparison.all():
            print("Q matrix is not orthogonal.")
            return False
        return True
    """
    Checks if given matrix is 2d, m >= n and filled with real numbers
    """
    def __check_pre_conditions(self, matrix: np.ndarray) -> bool:
        if not matrix.shape[0] >= matrix.shape[1]:
            print("Matrix is m is lesser than n.")
            return False
        if len(matrix.shape) != 2:
            print("Matrix is not 2D.")
            return False
        if not np.isreal(matrix).all():
            print("Matrix doesn't containt all real numbers.")
            return False
        return True

    """
    Method that performs Householder Transformation QR, acceptcs 2D, real numbers matrix, that
    fulfills m >= n condition. Return Q and R matrices.
    """
    def perform_householder_QR(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if(self.__check_pre_conditions(matrix)):
            return self.__householder_qr(matrix)
        else:
            print("Incorrect type.")
            raise Exception

    def __householder(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v = matrix / (matrix[0] + np.copysign(np.linalg.norm(matrix), matrix[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v,tau

    def __householder_qr(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m,n = matrix.shape
        R = matrix.copy()
        Q = np.identity(m)

        for j in range(0, n):
            v, tau = self.__householder(R[j:, j, np.newaxis])

            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        return Q[:n].T, np.triu(R[:n])

    def __givens_rotation(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if b == 0:
            c = 1
            s = 0
        else:
            if np.abs(b) > np.abs(a):
                r = a / b
                s = 1 / np.sqrt(1 + r**2)
                c = s * r
            else:
                r = b / a
                c = 1 / np.sqrt(1 + r**2)
                s = c * r
        return c,s

    def __givens_qr(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = matrix.shape
        R = matrix
        Q = np.eye(m)
        G = np.zeros((2,2))
        for j in range(n):
            for i in reversed(range(j+1, m)):
                a, b = R[i-1, j], R[i, j]
                G = np.asarray([[a, b], [-b, a]]) / np.sqrt(a**2 + b**2)
                R[i-1:i+1, j] = G @ R[i-1:i+1, j]
                Q[i-1:i+1, :] = G @ Q[i-1:i+1, :]

        return Q, R

    """
    Method that performs Givens Rotation QR, acceptcs 2D, real numbers matrix, that
    fulfills m >= n condition. Return Q and R matrices.
    """
    def perform_givens_QR(self, matrix: np.ndarray) ->  Tuple[np.ndarray, np.ndarray]:
        if(self.__check_pre_conditions(matrix)):
            return self.__givens_qr(matrix)
        else:
            print("Incorrect type.")
            raise Exception



"""
Usage example
"""
if __name__ == "__main__":
    qr = QR()
    matrix = np.matrix('1 2 4; 5 6 7; 8 9 10')
    matrix = np.asarray(matrix)
    print(matrix)
    Q, R = qr.perform_householder_QR(matrix)
    print("Householder:")
    print(Q)
    print(R)
    print("Givens")
    Q, R = qr.perform_givens_QR(matrix)
    print(Q)
    print(R)
    # dla porównania
    print("Numpy")
    Q, R = np.linalg.qr(matrix)
    print(Q)
    print(R)

