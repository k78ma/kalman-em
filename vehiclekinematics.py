import numpy
import math
from matplotlib import pyplot as plt

class VehicleKinematicModel():
    def __init__(self, Ts = 0.001, x=0, y=0, yaw=0, distortion_yaw = math.pi/4):
        self._transfer_function = numpy.identity(8)  # transfer matrix
        self._state = numpy.zeros(8)                            #state vector
        self._dstate = numpy.zeros(8)
        self._state[0] = x
        self._state[1] = y
        self._state[2] = yaw
        self._Ts = Ts                                            #sample time
        self._distortion_yaw = distortion_yaw

        self._offset = numpy.zeros(5)
        self._covariance = numpy.ones(5)*(0.0001)
        self._covariance[0] = 0.01
        self._covariance[1] = 0.01
        self._z   = numpy.zeros(5)
        self._z_n = numpy.zeros(5)

    def forward_kinematic(self, forward_velocity, angular_velocity):

        self._state[3] = forward_velocity
        self._state[5] = angular_velocity

        yaw = self._state[2]
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

        self._state = numpy.matmul(self._transfer_function, self._state.transpose())

        # self._dstate[0] =  cy * forward_velocity * self._Ts + 0.5 * cy * forward_velocity * self._Ts * self._Ts
        # self._dstate[1] =  sy * forward_velocity * self._Ts + 0.5 * sy * forward_velocity * self._Ts * self._Ts
        # self._dstate[2] =  angular_velocity * self._Ts
        # self._dstate[3] =  forward_velocity
        # self._dstate[4] =  0 #sy * forward_velocity
        # self._dstate[5] =  angular_velocity
        # self._dstate[6] =  0 #ignore acceleration for now
        # self._dstate[7] =  0 #ignore acceleration for now
        #
        # self._state[0] = self._state[0] + self._dstate[0]
        # self._state[1] = self._state[1] + self._dstate[1]
        # self._state[2] = self._state[2] + self._dstate[2]
        # self._state[3] = self._dstate[3]
        # self._state[4] = self._dstate[4]
        # self._state[5] = self._dstate[5]
        # self._state[6] = self._dstate[6]
        # self._state[7] = self._dstate[7]

    def distort_position_feedback(self): #Used to rotate position feedback
        R = numpy.array([[math.cos(self._distortion_yaw), -math.sin(self._distortion_yaw)],[math.sin(self._distortion_yaw), math.cos(self._distortion_yaw)]])
        p = numpy.array([self._state[0], self._state[1]])
        p_rotated = R.dot(p)
        return p_rotated

    def get_feedback(self):
        p = self._get_position_feedback()
        v = self._get_velocity_feedback()
        a = self._get_acceleration_feedback()
        z = numpy.append(p, v)
        z_n = numpy.zeros(5)
        for index, signal in enumerate(z):
            z_n[index] = self._add_white_noise(signal, index)

        self._z = numpy.column_stack((self._z, z))
        self._z_n = numpy.column_stack((self._z_n, z_n))

        return z_n

    def get_covariance(self):
        return self._covariance

    def _get_position_feedback(self):
        p_rotated = self.distort_position_feedback()
        p = p_rotated
        # p = numpy.append(p_rotated, self._state[2])
        #add noise and offset
        return p

    def _get_velocity_feedback(self):
        v = numpy.array([self._state[3], self._state[4], self._state[5]])
        # add noise and offset
        return v

    def _get_acceleration_feedback(self):
        a = numpy.array([self._state[6], self._state[7]])
        # add noise and offset
        return a

    def _add_white_noise(self, signal, index):
        noise = numpy.random.normal(0, math.sqrt(self._covariance[index]), 1)
        return signal+noise

    def _plot_robot_axis(self):
        R = numpy.array([[math.cos(self._state[2]), -math.sin(self._state[2])],[math.sin(self._state[2]), math.cos(self._state[2])]])
        x_axis = numpy.array([1, 0])
        x_axis_rotated = R.dot(x_axis)
        y_axis = numpy.array([0, 1])
        y_axis_rotated = R.dot(y_axis)
        x_axis_x = [self._state[0], self._state[0] + x_axis_rotated[0]]
        x_axis_y = [self._state[1], self._state[1] + x_axis_rotated[1]]
        y_axis_x = [self._state[0], self._state[0] + y_axis_rotated[0]]
        y_axis_y = [self._state[1], self._state[1] + y_axis_rotated[1]]

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(x_axis_x, x_axis_y)
        plt.plot(y_axis_x, y_axis_y)
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()

if __name__ == '__main__':
    numpy.set_printoptions(suppress=True)
    Ts = 0.001
    N = 10000
    model = VehicleKinematicModel(Ts, 0, 0, 0)
    for x in range(0, N):
        model.forward_kinematic(1, 0.001)
        z = model.get_feedback()
        model._plot_robot_axis()

    t = numpy.linspace(0, N, N+1)*0.001
    #print(t)

    plt.plot(t, model._z_n[3,:])
    plt.plot(t, model._z[3,:])
    plt.show()