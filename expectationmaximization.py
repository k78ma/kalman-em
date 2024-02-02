import numpy
import math
from kalmanfilter import KalmanFilter
from matplotlib import pyplot as plt

class ExpectationMaxization():
    def __init__(self, Ts):
        self._N = 0.0
        self._B1 = numpy.empty
        self._B2 = numpy.empty
        self._B3 = numpy.empty
        self._B4 = numpy.empty
        self._B5 = numpy.empty
        self._B6 = numpy.empty
        self._kalman_filter = KalmanFilter(Ts)

    def expectation(self, covariances, states, jacobians, feedbacks, lag_one_covariance):

        self._N = states.shape[0]
        self._get_B1(states, covariances)
        self._get_B2(states, jacobians, lag_one_covariance)
        self._get_B3(states, covariances)
        self._get_B4(feedbacks)
        self._get_B5(states, feedbacks)
        self._get_B6(states, feedbacks, covariances)

    def maximization(self):

        self._Q = (self._B1 - self._B2 - self._B2.T + self._B3)/(self._N-1)
        self._R = (self._B4 - self._B5 - self._B5.T + self._B6)/(self._N-1)

        # print("self._N-1")
        # print(self._N-1)
        #
        print("Q is positive definite: " + str(numpy.all(numpy.linalg.eigvals(self._Q) > 0)))
        # print(self._Q)
        # print(numpy.all(numpy.linalg.eigvals(self._Q) > 0))
        #
        print("R is positive definite: " + str(numpy.all(numpy.linalg.eigvals(self._R) > 0)))
        print(self._R)


    def _get_B1(self, states, covariances):

        N = states.shape[0]
        self._B1 = numpy.zeros((covariances.shape[1],covariances.shape[2]))
        count = 0
        for num in range(1, N): #every index except the first
            count = count + 1
            self._B1 = self._B1 + covariances[num, :, :] + numpy.outer(states[num, :], states[num,:].T)

        # print("self._B1")
        # print("count: " + str(count))
        # print(self._B1)

    def _get_B2(self, states, jacobians, lag_one_covariances):
        N = states.shape[0]
        self._B2 = numpy.zeros((lag_one_covariances.shape[1], lag_one_covariances.shape[2]))
        count = 0
        for num in range(1, N):
            count = count + 1
            self._kalman_filter.update_transfer_function(states[num-1, 2])
            self._kalman_filter.update_transfer_function_jacobian(states[num-1, 2], states[num-1, 3], states[num-1, 4], states[num-1, 6], states[num-1, 7])

            F = numpy.matmul(self._kalman_filter._transfer_function, states[num-1, :])
            phi = self._kalman_filter._transfer_function_jacobian

            #self._B2 = self._B2 + numpy.matmul(lag_one_covariances[num-1, :, :], jacobians[num-1, :, :].T) + numpy.outer(states[num, :], F.T)
            self._B2 = self._B2 + numpy.matmul(lag_one_covariances[num-1 , :, :], phi.T) + numpy.outer(states[num, :], F.T)

        # print("self._B2")
        # print("numpy.matmul(lag_one_covariances[num , :, :].shape: " + str(lag_one_covariances.shape))
        # print("count: " + str(count))
        # print(self._B2)

    def _get_B3(self, states, covariances):

        N = states.shape[0]
        self._B3 = numpy.zeros((covariances.shape[1], covariances.shape[2]))
        count = 0
        for num in range(0, N-1):
            count = count + 1
            self._kalman_filter.update_transfer_function(states[num, 2])
            self._kalman_filter.update_transfer_function_jacobian(states[num, 2], states[num, 3], states[num, 4], states[num, 6], states[num, 7])

            F = numpy.matmul(self._kalman_filter._transfer_function, states[num, :])
            phi = self._kalman_filter._transfer_function_jacobian
            self._B3 = self._B3 + numpy.outer(F, F.T) + numpy.matmul(numpy.matmul(phi, covariances[num, :, :]), phi.T)

        # print("self._B3")
        # print("count: " + str(count))
        # print(self._B3)

    def _get_B4(self, feedbacks):

        N = feedbacks.shape[0]
        count = 0
        self._B4 = numpy.zeros((feedbacks.shape[1],feedbacks.shape[1]))
        for num in range(0, N):
            count = count + 1
            self._B4 = self._B4 + numpy.outer(feedbacks[num, :], feedbacks[num,:].T)

        # print("self._B4")
        # print("count: " + str(count))
        # print(self._B4)

    def _get_B5(self, states, feedbacks):

        N = feedbacks.shape[0]
        count = 0
        self._B5 = numpy.zeros((feedbacks.shape[1],feedbacks.shape[1]))
        for num in range(0, N):
            count = count + 1
            H = numpy.array([states[num+1, 0], states[num+1, 1], states[num+1, 3], states[num+1, 4], states[num+1, 5]])
            #H = numpy.array([states[num, 0], states[num, 1], states[num, 3], states[num, 4], states[num, 5]])
            self._B5 = self._B5 + numpy.outer(feedbacks[num, :], H.T)

        # print("self._B5")
        # print("count: " + str(count))
        # print(self._B5)
        # print("self._B5.T")
        # print(self._B5.T)

    def _get_B6(self, states, feedbacks, covariances):

        N = states.shape[0]
        count = 0
        self._B6 = numpy.zeros((feedbacks.shape[1],feedbacks.shape[1]))
        C = numpy.zeros((5, 8))
        C[0, 0] = 1
        C[1, 1] = 1
        C[2, 3] = 1
        C[3, 4] = 1
        C[4, 5] = 1

        for num in range(0, N-1):
            count = count+1
            #H = numpy.array([states[num, 0], states[num, 1], states[num, 3], states[num, 4], states[num, 5]])
            H = numpy.array([states[num + 1, 0], states[num + 1, 1], states[num + 1, 3], states[num + 1, 4], states[num + 1, 5]])
            #B6_1 = covariances[num, :, :] + numpy.outer(states[num, :], states[num,:].T)
            # B6_1 = covariances[num, :, :]
            B6_1 = covariances[num + 1, :, :]
            self._B6 = self._B6 + numpy.matmul(numpy.matmul(C, B6_1), C.T) + numpy.outer(H, H.T)

        # print("self._B6")
        # print("count: " + str(count))
        # print(self._B6)


if __name__ == '__main__':
    numpy.set_printoptions(precision=2)
    EM = ExpectationMaxization()
