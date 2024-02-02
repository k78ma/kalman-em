import numpy
import math
from matplotlib import pyplot as plt

class KalmanFilter():
    def __init__(self, Ts = 0.001):
        self._transfer_function = numpy.identity(8)                     # transfer matrix
        self._transfer_function_jacobian = self._transfer_function      # transfer function jacobian
        self._estimate_covariance = numpy.identity(8) * (1e-9)  # initial estmate covariance
        # self._estimate_covariance[0, 0] = 1  # process noise covariance for yaw
        # self._estimate_covariance[1, 1] = 1  # process noise covariance for yaw
        # self._estimate_covariance[2, 2] = 1  # process noise covariance for yaw
        # self._estimate_covariance[3, 3] = 1  # process noise covariance for yaw
        # self._estimate_covariance[4, 4] = 1  # process noise covariance for yaw
        # self._estimate_covariance[5, 5] = 1  # process noise covariance for yaw
        self._process_noise_covariance = numpy.identity(8) * (1e-9)  # process noise covariance
        self._process_noise_covariance[0, 0] = 1e-4  # process noise covariance for yaw
        self._process_noise_covariance[1, 1] = 1e-4  # process noise covariance for yaw
        self._process_noise_covariance[2, 2] = 1e-4  # process noise covariance for yaw
        self._process_noise_covariance[3, 3] = 1e-4  # process noise covariance for yaw
        self._process_noise_covariance[4, 4] = 1e-4  # process noise covariance for yaw
        self._process_noise_covariance[5, 5] = 1e-4  # process noise covariance for yaw
        self._state = numpy.zeros(8)                                    # state vector
        self._cost = 0.0

        self._corrected_state =  self._state
        self._corrected_covariance = self._estimate_covariance

        self._Ts = Ts                                                   # sample time
        self._H = numpy.zeros((5, 8))
        for x in range(0, 2):
            self._H[x, x] = 1

        for x in range(2, 5):
            self._H[x, x+1] = 1

    def set_init_state(self, state0):
        self._state = state0

    def set_init_covariance(self, estimate_covariance0):
        self._estimate_covariance = estimate_covariance0
    def update_transfer_function(self, yaw):

        # yaw = self._state[2]
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        self._transfer_function[0, 3] =  cy * self._Ts
        self._transfer_function[0, 4] = -sy * self._Ts
        self._transfer_function[0, 6] =  0.5 * self._transfer_function[0, 3] * self._Ts
        self._transfer_function[0, 7] =  0.5 * self._transfer_function[0, 4] * self._Ts
        self._transfer_function[1, 3] =  sy * self._Ts
        self._transfer_function[1, 4] =  cy * self._Ts
        self._transfer_function[1, 6] =  0.5 * self._transfer_function[1, 3] * self._Ts
        self._transfer_function[1, 7] =  0.5 * self._transfer_function[1, 4] * self._Ts
        self._transfer_function[2, 5] =  self._Ts
        self._transfer_function[3, 6] =  self._Ts
        self._transfer_function[4, 7] =  self._Ts

    def project_covariance(self):
        #print(self._process_noise_covariance)
        self._estimate_covariance = numpy.matmul(numpy.matmul(self._transfer_function_jacobian, self._estimate_covariance), self._transfer_function_jacobian.T) + self._process_noise_covariance

    def update_transfer_function_jacobian(self, yaw, x_vel, y_vel, x_acc, y_acc):

        self._transfer_function_jacobian = numpy.copy(self._transfer_function)
        # yaw = self._state[2]
        # x_vel = self._state[3]
        # y_vel = self._state[4]
        # x_acc = self._state[6]
        # y_acc = self._state[7]
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        x_coeff = -sy
        y_coeff = -cy
        dFx_dY = (x_coeff * x_vel + y_coeff * y_vel)* self._Ts + (x_coeff * x_acc + y_coeff * y_acc) * 0.5 * self._Ts * self._Ts

        x_coeff = cy
        y_coeff = -sy
        dFy_dY = (x_coeff * x_vel + y_coeff * y_vel) * self._Ts + (x_coeff * x_acc + y_coeff * y_acc) * 0.5 * self._Ts * self._Ts

        self._transfer_function_jacobian[0, 2] = dFx_dY
        self._transfer_function_jacobian[1, 2] = dFy_dY

        #print("transfer_junction_jacobian: " + str(self._transfer_function_jacobian))

    def correct(self, measurement, R):

        # #1. Get measurement covariance matrix
        # R = numpy.identity(covariance.size)
        # for index in range(0, covariance.size):
        #     R[index, index] = covariance[index]

        #2. Calculate kalman gains
        sigma = numpy.matmul(numpy.matmul(self._H, self._estimate_covariance), self._H.transpose()) + R
        B = numpy.linalg.inv(sigma)
        A = numpy.matmul(self._estimate_covariance, self._H.transpose())
        self._kalman_gain = numpy.matmul(A,B)

        #3. Correct the states

        e = (measurement - numpy.matmul(self._H, self._state))
        # print("determinant(sigma): " + str(numpy.linalg.det(sigma)))
        # print("numpy.linalg.det(sigma)" + str(numpy.linalg.det(sigma)))
        # print("cost is: " + str(self._cost))
        # print("error is: " + str(e))
        self._cost = self._cost - (math.log(numpy.linalg.det(sigma)) + numpy.matmul(numpy.matmul(e.T, B), e))
        # self._cost = -self._cost

        # print("error is: " + str(e))
        self._state = self._state + numpy.matmul(self._kalman_gain, e)
        # print("state is: " + str(self._state))

        #4. Update estimate error covariance
        I = numpy.identity(8)
        residual = I-numpy.matmul(self._kalman_gain, self._H)
        self._estimate_covariance = numpy.matmul(numpy.matmul(residual, self._estimate_covariance), residual.transpose()) + \
            numpy.matmul(numpy.matmul(self._kalman_gain, R), self._kalman_gain.transpose())

    def predict(self):
        self.update_transfer_function(self._state[2])
        self.update_transfer_function_jacobian(self._state[2], self._state[3], self._state[4], self._state[6], self._state[7])
        self._state = numpy.matmul(self._transfer_function, self._state.transpose())
        self.project_covariance() #This simplies rotates and projects the covariance matrix

    def predict_and_correct(self, measurement, covariance):
        self.predict()
        self._forecasted_state = self._state
        self._forecasted_covariance = self._estimate_covariance

        self.correct(measurement, covariance)
        self._corrected_state = self._state
        self._corrected_covariance = self._estimate_covariance

    def plot_filter_axis(self):
        # self._plot_robot_axis(self._state)

        self._plot_robot_axis(self._corrected_state)

    def _plot_robot_axis(self, state):
        R = numpy.array([[math.cos(state[2]), -math.sin(state[2])], [math.sin(state[2]), math.cos(state[2])]])
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
        plt.pause(0.0001)
        plt.clf()

if __name__ == '__main__':
    numpy.set_printoptions(precision=2)
    filter = KalmanFilter()
    filter._state[2] = 0*math.pi/4
    filter.update_transfer_function()
    print(filter._transfer_function)

