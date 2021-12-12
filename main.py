from copy import deepcopy
from matplotlib import pyplot as plt
from typing import Tuple

import numpy as np
from scipy.linalg import solve

"""
    Class that performs qr decomposition. Use methods:
    perform_householder_QR,
    perform_givens_QR
    both accept np.ndarray that fulfils m >= n condition is 2d, and contains real numbers.
"""


class QR:
    def __init__(self) -> None:
        pass

    """
    Checks if calculated matrices fulfill QR decomposition conditions, that is:
    A = QR , where Q -> Q * Qt = I and R is a upper triangular matrix
    """

    def __check_condition(self, Q: np.matrix, R: np.matrix) -> bool:
        if not np.allclose(R, np.triu(R)):
            print("R matrix is not upper triangle.")
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
            print("Matrix doesn't contain all real numbers.")
            return False
        return True

    """
    Method that performs Householder Transformation QR, accepts 2D, real numbers matrix, that
    fulfills m >= n condition. Return Q and R matrices.
    """

    def perform_householder_QR(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.__check_pre_conditions(matrix):
            return self.__householder_qr(matrix)
        else:
            print("Incorrect type.")
            raise Exception

    def __householder(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v = matrix / (matrix[0] + np.copysign(np.linalg.norm(matrix), matrix[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v, tau

    def __householder_qr(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = matrix.shape
        R = matrix.copy()
        Q = np.identity(m)

        for j in range(0, n):
            v, tau = self.__householder(R[j:, j, np.newaxis])

            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        return Q[:n].T, np.triu(R[:n])

    def __givens_qr(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = matrix.shape
        R = matrix
        Q = np.eye(m)
        G = np.zeros((2, 2))
        for j in range(n):
            for i in reversed(range(j + 1, m)):
                a, b = R[i - 1, j], R[i, j]
                G = np.asarray([[a, b], [-b, a]]) / np.sqrt(a ** 2 + b ** 2)
                R[i - 1:i + 1, j] = G @ R[i - 1:i + 1, j]
                Q[i - 1:i + 1, :] = G @ Q[i - 1:i + 1, :]

        return Q.T, R

    """
    Method that performs Givens Rotation QR, accepts 2D, real numbers matrix, that
    fulfills m >= n condition. Return Q and R matrices.
    """

    def perform_givens_QR(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.__check_pre_conditions(matrix):
            return self.__givens_qr(matrix)
        else:
            print("Incorrect type.")
            raise Exception

    def solve_least_squares(self, A: np.ndarray, b: np.array, is_givens):
        if is_givens:
            Q, R = self.perform_givens_QR(A)
        else:
            Q, R = self.perform_householder_QR(A)
        x = solve(R, np.dot(Q.T, b))
        return x

    def __design_matrix(self, A: np.ndarray):
        return np.hstack((np.ones(A.shape[0]).reshape(-1, 1), A[:, :-1]))

    def fit_poly(self, A: np.ndarray, is_givens: bool):
        return self.solve_least_squares(
            np.dot(self.__design_matrix(A), self.__design_matrix(A).T),
            A[:, -1:].reshape(-1, 1),
            is_givens=is_givens)


"""
Usage example
"""
if __name__ == "__main__":
    qr = QR()
    matrix = np.matrix('0 0 0; 1 1 2; 1 2 4; 3 3 5; 5 6 7; 8 9 10')
    matrix = np.asarray(matrix)
    print(matrix)
    a = deepcopy(matrix)
    b = deepcopy(matrix)
    c = deepcopy(matrix)
    Q, R = qr.perform_householder_QR(a)
    print("Householder:")
    print(Q)
    print(R)
    print("Givens")
    Q, R = qr.perform_givens_QR(b)
    print(Q)
    print(R)
    # dla porównania
    print("Numpy")
    Q, R = np.linalg.qr(c)
    print(Q)
    print(R)
    print('solve least squares')
    print(matrix[:2, :-1])
    print(matrix[:2, -1:])
    print(qr.solve_least_squares(matrix[:2, :-1], matrix[:2, -1:], is_givens=True))

    def PolyCoefficients(x1,x2, coeffs):
        """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        o = len(coeffs)
        y = 0
        for i in range(o):
            y += coeffs[i] * x1 ** i + coeffs[i] * x2 ** i
        return y


    def f(a, b):
        return a + 2 * (b ** 2)


    x1 = np.asarray(np.random.randint(0,10,5))
    x2 = np.asarray(np.random.randint(0,10,5))
    y = f(x1, x2)
    mat = np.asarray([x1, x2, y])
    plt3d = plt.figure().gca(projection='3d')
    #xx,yy = np.meshgrid(np.linspace(0,10,11),np.linspace(0,10,11))
    xx, yy = np.meshgrid(range(10), range(10))
    plt3d.plot_surface(xx, yy, f(xx, yy), alpha=0.2)
    plt3d.plot_surface(xx, yy, PolyCoefficients(xx,yy, qr.fit_poly(mat.T, is_givens=True)), alpha=0.2)
    plt.show()
    print(qr.fit_poly(mat.T, is_givens=True))

    print( qr.fit_poly(mat.T, is_givens=True))