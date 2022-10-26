import random
import numpy as np
from math import ceil, log

def generate_matrix(rows, columns):
    matrix = np.array([[random.randint(0, 100) for _ in range(columns)] for _ in range(rows)])
    matrix = matrix.reshape(rows, columns)
    return matrix

def split_matrix(A):
    if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
        raise Exception('Only even matrices!')

    size = A.shape[0]
    mid = size // 2

    A11 = np.array([[A[i][j] for j in range(mid)] for i in range(mid)])
    A12 = np.array([[A[i][j] for j in range(mid)] for i in range(mid, size)])
    A21 = np.array([[A[i][j] for j in range(mid, size)] for i in range(mid)])
    A22 = np.array([[A[i][j] for j in range(mid, size)] for i in range(mid, size)])

    return A11, A12, A21, A22

def do_strassen_method(A, B):
    if A.shape[0] == 1 or B.shape[1] == 1:
        return A * B

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    P1 = do_strassen_method(A11, B12-B22)
    P2 = do_strassen_method(A11+A12, B22)
    P3 = do_strassen_method(A21+A22, B11)
    P4 = do_strassen_method(A22, B21-B11)
    P5 = do_strassen_method(A11+A22, B11+B22)
    P6 = do_strassen_method(A12-A22, B21+B22)
    P7 = do_strassen_method(A11-A21, B11+B12)

    C11 = P5+P4-P2+P6
    C12 = P1+P2
    C21 = P3+P4
    C22 = P5+P1-P3-P7

    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

class AdaptiveMomentEstimationMethod():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, b, dw, db):
        # calculate new increments
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        # calculate new biases
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * db

        # correct bias
        m_dw_corr = self.m_dw /(1 - self.beta1**t)
        m_db_corr = self.m_db / (1 - self.beta1**t)
        v_dw_corr = self.v_dw / (1 - self.beta2**t)
        v_db_corr = self.v_db / (1 - self.beta2**t)

        # update weights and biases
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))

        return w, b

def loss_function(m):
    return m**2 - 2 * m + 1

# take derivative
def grad_function(m):
    return 10*m - 7

def check_convergence(w0, w1):
    return w0 == w1

if __name__ == '__main__':
    rows_A = columns_A = rows_B = columns_B = 64
    A = generate_matrix(rows_A, columns_A)
    B = generate_matrix(rows_B, columns_B)
    print('First matrix:\n', A)
    print('\nSecond matrix:\n', B)
    print('\nResult:\n', do_strassen_method(A, B))

    w0 = 0
    b0 = 0
    t = 1
    converged = False
    adam = AdaptiveMomentEstimationMethod()

    while not converged:
        dw = grad_function(w0)
        db = grad_function(b0)
        w0_old = w0
        w0, b0 = adam.update(t, w = w0, b = b0, dw = dw, db = db)
        if check_convergence(w0, w0_old):
            print('Converged after ' + str(t) + ' iterations with weight =', w0)
            break
        else:
            t += 1
