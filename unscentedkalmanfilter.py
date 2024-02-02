import filterpy.common

import numpy
import scipy.linalg
import math
from matplotlib import pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class UnscentedKalmanFilter():
    def __init__(self, Ts = 0.001):
        self._transfer_function = numpy.identity(8)                     # transfer matrix
        self._transfer_function_jacobian = self._transfer_function      # transfer function jacobian
        self._estimate_covariance = numpy.identity(8) * (1e-9)  # initial estmate covariance
        self._estimate_covariance[0, 0] = 1  # process noise covariance for yaw
        self._estimate_covariance[1, 1] = 1  # process noise covariance for yaw
        self._estimate_covariance[2, 2] = 1  # process noise covariance for yaw
        self._estimate_covariance[3, 3] = 1  # process noise covariance for yaw
        self._estimate_covariance[4, 4] = 1  # process noise covariance for yaw
        self._estimate_covariance[5, 5] = 1  # process noise covariance for yaw
        self._process_noise_covariance = numpy.identity(8) * (1e-6)  # process noise covariance
        self._state = numpy.zeros(8)                                    # state vector
        self._Ts = Ts                                                   # sample time
        self._H = numpy.zeros((5, 8))
        for x in range(0, 2):
            self._H[x, x] = 1

        for x in range(2, 5):
            self._H[x, x+1] = 1

        self._N = 8

        #Van Der Merwe paramters
        self._alpha = 0.1
        self._beta = 2
        self._kappa = 0
        self._lambda = (self._alpha*self._alpha*(self._N+self._kappa) - self._N)
        self._sigma_matrix = numpy.zeros([self._N, 2*self._N + 1])  # transfer matrix

        w = 1/(2*(self._N + self._lambda))

        self._mean_weights = numpy.ones(2*self._N+1)*w
        self._covariance_weights = numpy.ones(2*self._N+1)*w
        self._mean_weights[0] = self._lambda/(self._N+self._lambda)
        self._covariance_weights[0] = self._mean_weights[0] + (1 - self._alpha*self._alpha + self._beta)


        #initialize filterpy ukf to compare
        self.sigmas = MerweScaledSigmaPoints(8, alpha=self._alpha, beta=self._beta, kappa=self._kappa)
        self.ukf = UKF(dim_x=8, dim_z=3, fx=self.f_cv, hx=self.h_cv, dt=self._Ts, points=self.sigmas)
        self.ukf.x = numpy.array([0., 0., 0., 0., 0., 0., 0., 0.])
        self.ukf.R = numpy.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        self.ukf.Q = numpy.identity(8) * (1e-9)    # process noise covariance
        self.ukf.P = numpy.identity(8) * (1e-9)

    def update_transfer_function(self, state):
        yaw = state[2]
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        self._transfer_function[0, 3] = cy * self._Ts
        self._transfer_function[0, 4] = -sy * self._Ts
        self._transfer_function[0, 6] = 0.5 * self._transfer_function[0, 3] * self._Ts
        self._transfer_function[0, 7] = 0.5 * self._transfer_function[0, 4] * self._Ts
        self._transfer_function[1, 3] = sy * self._Ts
        self._transfer_function[1, 4] = cy * self._Ts
        self._transfer_function[1, 6] = 0.5 * self._transfer_function[1, 3] * self._Ts
        self._transfer_function[1, 7] = 0.5 * self._transfer_function[1, 4] * self._Ts
        self._transfer_function[2, 5] = self._Ts
        self._transfer_function[3, 6] = self._Ts
        self._transfer_function[4, 7] = self._Ts

    def correct(self, measurement, covariance):

        #1. Get measurement covariance matrix
        R = numpy.identity(covariance.size)
        for index in range(0, covariance.size):
            R[index, index] = covariance[index]

        #2. Calculate kalman gains
        B = numpy.linalg.inv(numpy.matmul(numpy.matmul(self._H, self._estimate_covariance), self._H.transpose()) + R)
        A = numpy.matmul(self._estimate_covariance, self._H.transpose())
        self._kalman_gain = numpy.matmul(A,B)

        #3. Correct the states

        e = (measurement - numpy.matmul(self._H, self._state))
        #print("error is: " + str(e))
        self._state = self._state + numpy.matmul(self._kalman_gain, e)

        #4. Update estimate error covariance
        I = numpy.identity(8)
        residual = I-numpy.matmul(self._kalman_gain, self._H)
        self._estimate_covariance = numpy.matmul(numpy.matmul(residual, self._estimate_covariance), residual.transpose()) + \
            numpy.matmul(numpy.matmul(self._kalman_gain, R), self._kalman_gain.transpose())

    def transform_sigma_matrix(self, sigma_matrix):
        transformed_sigma_matrix = numpy.zeros([self._N, 2 * self._N + 1])
        n_row, n_col = sigma_matrix.shape
        for i in range(0, n_col):
            transformed_sigma_matrix[:, i] = self.transform_state(sigma_matrix[:, i])

        return transformed_sigma_matrix

    def transform_state(self, state):
        self.update_transfer_function(state)
        state = numpy.matmul(self._transfer_function, state.transpose())
        #self.update_differential_transfer_function(state)
        #state = self._state + numpy.matmul(self._differential_transfer_function, state.transpose())
        return state

    def f_cv(self, x, dt):
        self.update_transfer_function(x)
        F = self._transfer_function
        return F @ x

    def h_cv(self, x):
        return x[[0, 1, 3, 4, 5]]

    def filterpy_predict(self):
        self.ukf.predict()

    def filterpy_update(self, measurement):
        self.ukf.update(measurement)

    def unscented_transform(self, x, mean_weights, covariance_weights, noise):

        n_rows, n_cols = x.shape
        mean = numpy.zeros(n_rows)
        covariance = numpy.zeros([n_rows, n_rows])

        #1. Transform mean
        for i in range(0, numpy.prod(mean_weights.shape)):
            mean = mean + mean_weights[i] * x[:, i]

        #2. Transform covariance
        for i in range(0, numpy.prod(covariance_weights.shape)):
            dx = x[:, i] - mean
            covariance = covariance + covariance_weights[i] * numpy.outer(dx, dx)

        covariance = covariance + noise

        return mean, covariance

    def predict_and_correct(self, measurement, covariance):

        #self.update_transfer_function(self._state)

        #2. Get measurement covariance matrix
        R = numpy.identity(covariance.size)
        for index in range(0, covariance.size):
            R[index, index] = covariance[index]

        #3. Get sigma points
        self.get_sigma_points()

        #4. Prediction Step
        transformed_sigma_matrix = self.transform_sigma_matrix(self._sigma_matrix)
        [prediction_mean, prediction_covariance] = self.unscented_transform(transformed_sigma_matrix, self._mean_weights, self._covariance_weights, self._process_noise_covariance)

        #5. Get measurement of transformed sigma points
        Z = numpy.matmul(self._H, transformed_sigma_matrix)
        [measurement_mean, measurement_covariance] = self.unscented_transform(Z, self._mean_weights, self._covariance_weights, R)

        cross_correlation = numpy.zeros([8, 5])

        #6. Get cross correlation between prediction and measurement
        for i in range(0, numpy.prod(self._covariance_weights.shape)):
            dy = transformed_sigma_matrix[:, i] - prediction_mean
            dz = Z[:, i] - measurement_mean
            cross_correlation = cross_correlation + self._covariance_weights[i] * numpy.outer(dy, dz)

        #7. Calculate Kalman Gain
        self._kalman_gain = numpy.matmul(cross_correlation, numpy.linalg.inv(measurement_covariance))

        #8. Correct
        e = (measurement - measurement_mean)
        self._state = prediction_mean + numpy.matmul(self._kalman_gain, e)
        self._estimate_covariance = prediction_covariance - numpy.matmul(numpy.matmul(self._kalman_gain, measurement_covariance), self._kalman_gain.transpose())

        print("error is: " + str(e))
        print("kalman gain is: " + str(self._kalman_gain))


    def get_sigma_points(self):

        self._sigma_matrix[:, 0] = self._state

        estimate_covariance_squared = scipy.linalg.sqrtm((self._N+self._lambda)*self._estimate_covariance)

        for i in range(1, self._N+1):
            self._sigma_matrix[:, i] = self._state + estimate_covariance_squared[i-1, :]

        for j in range(self._N+1, 2*self._N+1):
            self._sigma_matrix[:, j] = self._state - estimate_covariance_squared[j-1-self._N, :]

    def plot_filterpy_axis(self):
        self._plot_robot_axis(self.ukf.x)

    def plot_filter_axis(self):
        self._plot_robot_axis(self._state)

    def _plot_robot_axis(self, state):
        R = numpy.array([[math.cos(state[2]), -math.sin(state[2])],[math.sin(state[2]), math.cos(state[2])]])
        x_axis = numpy.array([5, 0])
        x_axis_rotated = R.dot(x_axis)
        y_axis = numpy.array([0, 5])
        y_axis_rotated = R.dot(y_axis)
        x_axis_x = [state[0], state[0] + x_axis_rotated[0]]
        x_axis_y = [state[1], state[1] + x_axis_rotated[1]]
        y_axis_x = [state[0], state[0] + y_axis_rotated[0]]
        y_axis_y = [state[1], state[1] + y_axis_rotated[1]]

        plt.xlim([-50, 50])
        plt.ylim([-50, 50])
        plt.plot(x_axis_x, x_axis_y)
        plt.plot(y_axis_x, y_axis_y)
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()

if __name__ == '__main__':
    numpy.set_printoptions(precision=2)
    filter = UnscentedKalmanFilter()
    filter._state[2] = 0*math.pi/4
    filter.update_transfer_function()
    print(filter._transfer_function)
