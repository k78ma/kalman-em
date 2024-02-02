import numpy
import math
from matplotlib import pyplot as plt

class KalmanSmoother():
    def __init__(self):
        self._covariance = numpy.identity(8) * (1e-9)  # process noise covariance
        self._state = numpy.zeros(8)                                    # state vector
        self._J = numpy.identity(8)
        self._lag_one_covariance = numpy.identity(8)

    def run_smoother(self, forecasted_states, forecasted_covariances, corrected_states, corrected_covariances, jacobian, kalman_gain):
        N = forecasted_covariances.shape[0]

        self._state = numpy.zeros((N+1, 8))
        self._covariance = numpy.zeros((N+1, 8, 8))
        self._lag_one_covariance = numpy.zeros((N, 8, 8))
        self._J = numpy.zeros((N, 8, 8))

        self._state[-1, :] = corrected_states[-1, :]
        self._covariance[-1, :] = corrected_covariances[-1, :, :]

        C = numpy.zeros((5, 8))
        C[0, 0] = 1
        C[1, 1] = 1
        C[2, 3] = 1
        C[3, 4] = 1
        C[4, 5] = 1

        I = numpy.identity(8)

        self._lag_one_covariance[N-1, :, :] = numpy.matmul(numpy.matmul((I - numpy.matmul(kalman_gain, C)) , jacobian[N-1, :, :]), corrected_covariances[N-1, :, :])

        # print("initial lag one covariance")
        # print(numpy.matmul(numpy.matmul((I - numpy.matmul(kalman_gain, C)), jacobian[N - 1, :, :]), corrected_covariances[N - 2, :, :]))

        # print("corrected_covariances.shape(): " + str(corrected_covariances.shape))
        # print("forecasted_covariances.shape(): " + str(forecasted_covariances.shape))

        # input("PAUSE")

        self._J[N-1, :, :] = numpy.matmul(numpy.matmul(corrected_covariances[N-1, :, :], jacobian[N-1, :, :].T), numpy.linalg.inv(forecasted_covariances[N-1, :, :]))

        # print("N: " + str(N))
        for num in reversed(range(0,N)):
            self._state[num, :] = corrected_states[num, :] + numpy.matmul(self._J[num, :, :], (self._state[num+1, :] - forecasted_states[num, :]))
            self._covariance[num, :, :] = corrected_covariances[num, :, :] + numpy.matmul(numpy.matmul(self._J[num, :, :], (self._covariance[num+1, :, :] - forecasted_covariances[num, :, :])), self._J[num, :, :].T)
            if(num > 0):
                self._J[num - 1, :, :] = numpy.matmul(numpy.matmul(corrected_covariances[num-1, :, :], jacobian[num-1, :, :].T), numpy.linalg.inv(forecasted_covariances[num-1, :, :]))
                self._lag_one_covariance[num-1, :, :] = numpy.matmul(corrected_covariances[num, :, :], self._J[num-1, :, :].T) + numpy.matmul(numpy.matmul(self._J[num, :, :], (self._lag_one_covariance[num, :, :] - numpy.matmul(jacobian[num, :, :], corrected_covariances[num, :, :]))), self._J[num-1, :, :].T)
            # print("_lag_one_covariance index: " + str(num-1))
            # print(self._J[num - 1, :, :])

        # print("self._state.shape(): " + str(self._state.shape))
        # print("self._covariance.shape(): " + str(self._covariance.shape))
        # print("self._J.shape(): " + str(self._J.shape))
        # print("self._lag_one_covariance.shape(): " + str(self._lag_one_covariance.shape))
        # print("jacobian.shape(): " + str(jacobian.shape))
        # input("pause")

    def get_smoothed_states(self):
        return self._state
    def get_smoothed_covariances(self):
        return self._covariance
    def get_lag_one_covariance(self):
        return self._lag_one_covariance

if __name__ == '__main__':
    numpy.set_printoptions(precision=2)
    smoother = KalmanSmoother()
    smoother._state[2] = 0*math.pi/4

