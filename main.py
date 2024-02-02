import numpy
import math
from matplotlib import pyplot as plt
from kalmanfilter import KalmanFilter
from kalmansmoother import KalmanSmoother
from unscentedkalmanfilter import UnscentedKalmanFilter
from vehiclekinematics import VehicleKinematicModel
from expectationmaximization import ExpectationMaxization
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

if __name__ == '__main__':

    forecasted_state = numpy.empty
    forecasted_covariances = numpy.empty

    corrected_state = numpy.empty
    corrected_covariances = numpy.empty

    feedback = numpy.empty

    jacobian = numpy.empty

    numpy.set_printoptions(precision=3, suppress=True)
    Ts = 0.05
    N = 300

    EM = ExpectationMaxization(Ts)
    #R = numpy.empty
    #filter = UnscentedKalmanFilter(Ts)

    state0 = numpy.zeros(8)
    estimate_covariance0 = numpy.identity(8) * (1e-9)  # initial estmate covariance

    v_desired = 10.0
    w_desired = 0.5
    a = 100
    a_w = 5.0

    #model = VehicleKinematicModel(Ts, 0, 0, math.pi / 4, math.pi / 2)
    model = VehicleKinematicModel(Ts, 0, 0, 0, 0.0)
    true_state = numpy.empty

    for i in range(0, 20):

        z_filt = numpy.zeros(8)
        filter = KalmanFilter(Ts)
        v = 10.0
        w = 0.5
        # print(state0)
        # print(estimate_covariance0)

        filter.set_init_state(state0)
        filter.set_init_covariance(estimate_covariance0)
        smoother = KalmanSmoother()

        if i > 0:
            R = EM._R
            filter._process_noise_covariance = EM._Q

        for x in range(0, N):
            if i == 0:
                model.forward_kinematic(v, w)
                v = min([v + a*Ts, v_desired])
                w = min([w + a_w * Ts, w_desired])
                z = model.get_feedback()
                sigma = model.get_covariance()
            else:
                z = model._z_n[:,x]

            if not 'R' in locals():
                R = numpy.identity(sigma.size)*1e-2
                # for index in range(0, sigma.size):
                #     R[index, index] = sigma[index]

                print(R)

            if x == 0:
                jacobian = filter._transfer_function_jacobian.reshape(1, 8, 8)
                corrected_covariances = filter._corrected_covariance.reshape(1, 8, 8)
                corrected_state = filter._corrected_state
                if i == 0:
                    true_state = model._state
            else:
                jacobian = numpy.vstack([jacobian, filter._transfer_function_jacobian.reshape(1, 8, 8)])
                corrected_covariances = numpy.vstack([corrected_covariances, filter._corrected_covariance.reshape(1, 8, 8)])
                corrected_state = numpy.vstack([corrected_state, filter._corrected_state])

            filter.predict_and_correct(z, R)

            if i ==0:
                true_state = numpy.vstack([true_state, model._state])

            if x == 0:
                forecasted_covariances = filter._forecasted_covariance.reshape(1, 8, 8)
                forecasted_state = filter._forecasted_state
                feedback = z
            else:
                forecasted_covariances = numpy.vstack([forecasted_covariances, filter._forecasted_covariance.reshape(1, 8, 8)])
                forecasted_state = numpy.vstack([forecasted_state, filter._forecasted_state])
                kalman_gain = filter._kalman_gain #Store the last kalman gain
                feedback = numpy.vstack([feedback, z])

            #filter.plot_filter_axis()
            z_filt = numpy.column_stack((z_filt, filter._state))


        print("cost: " + str(filter._cost))

        corrected_covariances = numpy.vstack([corrected_covariances, filter._corrected_covariance.reshape(1, 8, 8)])
        corrected_state = numpy.vstack([corrected_state, filter._corrected_state])

        smoother.run_smoother(forecasted_state, forecasted_covariances, corrected_state, corrected_covariances, jacobian, kalman_gain)
        state0 = smoother.get_smoothed_states()[0,:]
        estimate_covariance0 = smoother.get_smoothed_covariances()[0, :, :]

        EM.expectation(smoother.get_smoothed_covariances(), smoother.get_smoothed_states(), jacobian, feedback, smoother.get_lag_one_covariance())
        EM.maximization()

        t = numpy.linspace(0, N, N + 1) * 0.001
        fig, axs = plt.subplots(5, 1)
        #plt.plot(t, model._z_n[0, :], label='measurement', marker='o')
        axs[0].plot(t, true_state[:, 0], label='true value')
        axs[0].plot(t, model._z_n[0, :], label='measured')
        # axs[0].plot(t, smoother.get_smoothed_states()[:, 0] , label='Kalman Smoothed' )
        axs[0].plot(t, z_filt[0, :], label='filtered value')
        axs[0].set(ylabel="X [m]")

        axs[1].plot(t, true_state[:, 1], label='true value')
        axs[1].plot(t, model._z_n[1, :], label='measured')
        # axs[1].plot(t, smoother.get_smoothed_states()[:, 1], label='Kalman Smoothed')
        axs[1].plot(t, z_filt[1, :], label='filtered value')
        axs[1].set(ylabel="Y [m]")

        # axs[2].plot(t, true_state[:, 2], label='true value')
        # axs[2].plot(t, smoother.get_smoothed_states()[:, 2], label='Kalman Smoothed')
        # axs[2].plot(t, z_filt[2, :], label='filtered value')

        # axs[2].plot(t, true_state[:, 2], label='true value')
        # # axs[2].plot(t, model._z[2, :], label='measured')
        # # axs[2].plot(t, smoother.get_smoothed_states()[:, 2], label='Kalman Smoothed')
        # axs[2].plot(t, z_filt[2, :], label='filtered value')

        axs[2].plot(t, true_state[:, 3], label='true value')
        axs[2].plot(t, model._z_n[2, :], label='measured')
        # axs[3].plot(t, smoother.get_smoothed_states()[:, 3], label='Kalman Smoothed')
        axs[2].plot(t, z_filt[3, :], label='filtered value')
        axs[2].set(ylabel="X velocity [m/s]")

        axs[3].plot(t, true_state[:, 4], label='true value')
        axs[3].plot(t, model._z_n[3, :], label='measured')
        # axs[4].plot(t, smoother.get_smoothed_states()[:, 4], label='Kalman Smoothed')
        axs[3].plot(t, z_filt[4, :], label='filtered value')
        axs[3].set(ylabel="Y velocity [m/s]")

        axs[4].plot(t, true_state[:, 5], label='true value')
        axs[4].plot(t, model._z_n[4, :], label='measured')
        # axs[5].plot(t, smoother.get_smoothed_states()[:, 5], label='Kalman Smoothed')
        axs[4].plot(t, z_filt[5, :], label='filtered value')
        axs[4].set(xlabel = "Time [s]", ylabel="Yaw velocity [rad/s]")

        plt.legend()
        plt.show()

        # print(EM._Q)
        # print(EM._R)

        R = EM._R
        filter._process_noise_covariance = EM._Q

    # for x in range(0, N*2):
    #     model.forward_kinematic(-10.0, -0.2)
    #     z = model.get_feedback()
    #     sigma = model.get_covariance()
    #     print("covariance: " + str(sigma))
    #     filter.predict_and_correct(z, sigma)
    #     filter.plot_filter_axis()
    #     z_filt = numpy.column_stack((z_filt, filter._state))

