import numpy as np
from numpy.linalg import inv
from numpy import dot
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math
import seaborn


class BasicMovement:
    def __init__(self, maxSpeed, maxRotation, covariance, measureFunction):
        self.maxSpeed = maxSpeed
        self.maxRotation = maxRotation
        self.measureFunction = measureFunction
        self.covariance = np.atleast_2d(covariance)

    #  Input the real state
    def move(self, state, covariance=None, command=None):
        command = self.__choose_command(state) if command is None else command
        noise = self.__get_noise(covariance)
        idealMove = self.exact_move(state, command)
        realMove = self.__noisy_move(state, idealMove, noise)
        newState = state + realMove
        return clipState(newState), command

    def __choose_command(self, state):
        speed = self.maxSpeed * np.random.rand()
        rotation = (np.random.rand() * 2 - 1) * self.maxRotation
        return [speed, rotation]

    def exact_move(self, state, command):
        speed, rotation = command
        angle = state[2]
        deltaX = speed * math.cos(angle)
        deltaY = speed * math.sin(angle)

        move = np.zeros_like(state)
        move[:3, 0] = [deltaX, deltaY, rotation]
        return move

    def __noisy_move(self, state, idealMove, noise):
        noisyMove = idealMove[:3] + noise
        noisySpeed, _ = self.measureFunction(noisyMove[:3], np.zeros_like(noise)[:2])
        noisyRotation = noisyMove[2]

        maxs = [self.maxSpeed, self.maxRotation]
        mins = [0, -self.maxRotation]
        correctedCommand = np.clip([noisySpeed, noisyRotation], mins, maxs)
        return self.exact_move(state, correctedCommand)

    def __noisy_move2(self, state, idealMove, noise):
        noisyMove = np.zeros_like(state)
        noisyMove[:3] = idealMove[:3] + noise
        return noisyMove

    def __get_noise(self, covariance):
        covariance = self.covariance if covariance is None else covariance
        noise = np.random.multivariate_normal(np.zeros(covariance.shape[0]), covariance, 1).T
        return noise


class BasicMeasurement:
    def __init__(self, covariance, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize=0, detectionCone=0):
        self.covariance = np.atleast_2d(covariance)
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.measureFunction = measureFunction
        self.gradMeasureFunction = gradMeasureFunction
        self.detectionSize = detectionSize
        self.detectionCone = detectionCone

    #  Input the real state
    def measure(self, state):
        dim = state.shape[0]
        dimR = self.robotFeaturesDim
        dimE = self.envFeaturesDim
        rState = state[:dimR]
        envState = state[dimR:]
        nbLandmark = (dim - dimR) / dimE

        mes = np.zeros(nbLandmark * dimE).reshape(nbLandmark, dimE)
        landmarkIds = np.zeros(nbLandmark)
        j = 0

        for i, landmark in enumerate(envState.reshape((nbLandmark, dimE, 1))):
            diffNorm, diffAngle = self.measureFunction(rState, landmark)
            angleOk = (abs(clipAngle(diffAngle, True)) < self.detectionCone / 2.) or (self.detectionCone is 0)
            distanceOk = (diffNorm < self.detectionSize) or (self.detectionSize is 0)

            if distanceOk and angleOk:
                mes[j] = [diffNorm, diffAngle]
                landmarkIds[j] = i
                j += 1

        mes = mes[:j]
        landmarkIds = landmarkIds[:j]
        mes = np.array(mes) + self.__get_noise(mes)
        return mes, landmarkIds

    def __get_noise(self, mes):
        noise = np.random.multivariate_normal(np.zeros(self.covariance.shape[0]), self.covariance, mes.shape[0])
        return noise


class EIFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, int(ldmIndex))
        return self.H, self.b

    def __motion_update(self, command, U):
        previousMeanState = self.estimate()
        meanStateChange = self.motionModel.exact_move(previousMeanState, command)
        newMeanState = clipState(previousMeanState + meanStateChange)

        # TO IMPROVE
        angle = previousMeanState[2, 0]  # TO IMPROVE
        gradMeanMotion = np.zeros_like(self.H)  # TO IMPROVE
        gradMeanMotion[2, 0:2] = command[0] * np.array([-math.sin(angle), math.cos(angle)])  # TO IMPROVE

        IA = np.eye(self.H.shape[0]) + gradMeanMotion  # TO IMPROVE
        sigma = dot(dot(IA, inv(self.H)), IA.T) + dot(dot(self.S, U), self.S.T)
        self.H = inv(sigma)
        self.b = dot((newMeanState).T,  self.H)

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.estimate()
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.array(ldmMes).reshape(len(ldmMes), 1)
        zM = np.array(meanMes).reshape(len(ldmMes), 1)
        C = gradMeanMes

        mesError = (z - zM)
        mesError[1, 0] = clipAngle(mesError[1, 0], force=True)
        mesError += dot(C.T, mu)
        mesError[1, 0] = clipAngle(mesError[1, 0])
        self.H += dot(dot(C, self.invZ),  C.T)
        self.b += dot(dot(mesError.T, self.invZ), C.T)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(rMeanState, ldmMeanState, realIndex)
        return meanMes, gradMeanMes

    def estimate(self, H=None, b=None):
        H = self.H if H is None else H
        b = self.b if b is None else b
        return clipState(dot(b, inv(H)).T)


# class Robot:
#     def __init__(self, maxSpeed, maxRotation, detectionSize, detectionCone, featuresDim, covarianceMotion, covarianceMeasurements):
#


def measureFunction(rState, landmark):
    rDim = 3
    diff = landmark - rState[:rDim-1]
    diffNorm = np.linalg.norm(diff)

    angle = rState[rDim-1, 0]
    diffAngle = math.atan2(diff[1], diff[0]) - angle
    diffAngle = clipAngle(diffAngle)

    return diffNorm, diffAngle


def gradMeasureFunction(rState, landmark, ldmIndex):
    rDim = 3
    eDim = 2
    diff = (rState[:rDim-1] - landmark).flatten()
    diffNorm = np.linalg.norm(diff)

    grad = np.zeros(dimension * 2).reshape(dimension, 2)
    grad[:rDim-1, 0] = diff / diffNorm
    grad[ldmIndex:ldmIndex + eDim, 0] = -grad[:rDim-1, 0]
    grad[:rDim-1, 1] = np.array([-diff[1], diff[0]]) / (diffNorm**2)
    grad[ldmIndex:ldmIndex + eDim, 1] = -grad[:rDim-1, 1]
    grad[rDim-1, 1] = -1

    return grad


def clipAngle(angle, force=False):
    if clip or force:
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle


def clipState(state):
    if clip:
        state[2, 0] = clipAngle(state[2, 0])
    return state

clip = False


if __name__ == '__main__':

    T = 500  # Number of timesteps
    nbLandmark = 64
    maxSpeed = 5
    maxRotation = 45 * math.pi / 180  # 45  # en radians

    # Robot Detection Parameters
    detectionSize = 40
    detectionCone = 0 # 180 * math.pi / 180  # en radians

    # Dimension Constants
    robotFeaturesDim = 3
    envFeaturesDim = 2
    commandsDim = 2
    mesDim = 2
    dimension = robotFeaturesDim + nbLandmark * envFeaturesDim

    # Covariances for motions and measurements
    covarianceMotion = np.eye(robotFeaturesDim)
    covarianceMotion[0, 0] = 1 ** 2  # motion noise variance X
    covarianceMotion[1, 1] = 1 ** 2  # motion noise variance Y
    covarianceMotion[2, 2] = (5 * math.pi / 180) ** 2  # motion noise variance Angle

    covarianceMeasurements = np.eye(mesDim)
    covarianceMeasurements[0, 0] = 1 ** 2  # measurement noise variance distance
    covarianceMeasurements[1, 1] = (5 * math.pi / 180) ** 2  # motion noise variance Angle


    ## ----------------------------------------------------------------------
    ## Simulation initialization

    ## -------------------
    ## State Definition

    # Real robot state
    state = np.zeros((dimension, 1))

    x = np.linspace(-200, 200, np.sqrt(nbLandmark))
    y = np.linspace(-200, 200, np.sqrt(nbLandmark))
    xv, yv = np.meshgrid(x, y)
    state[robotFeaturesDim:, 0] = np.vstack([xv.ravel(), yv.ravel()]).ravel(order="F")
    # state[robotFeaturesDim:] = np.random.rand(nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1) * 300 - 150


    # Basic and EIF estimator for robot state
    mu = state.copy()
    mu[robotFeaturesDim:] += np.random.normal(0, covarianceMeasurements[0, 0], nbLandmark * envFeaturesDim).reshape(nbLandmark * envFeaturesDim, 1)
    muEIF = mu.copy()

    ## --------------------
    ## Models Definition

    motionModel = BasicMovement(maxSpeed, maxRotation, covarianceMotion, measureFunction)
    measurementModel = BasicMeasurement(covarianceMeasurements, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize, detectionCone)
    eif = EIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)


    mus_simple = np.zeros((T, dimension))
    mus_eif = np.zeros((T, dimension))
    states = np.zeros((T, dimension))

    mus_simple[0] = np.squeeze(mu)
    mus_eif[0] = np.squeeze(muEIF)
    states[0] = np.squeeze(state)


    # LOG Initial state
    print("BEFORE")
    print("EIF estimate :")
    print(muEIF)
    print("Real state :")
    print(state)
    print('\n')

    for t in range(1, T):
        print("\nIteration %d" % t)
        state, motionCommand = motionModel.move(state)
        measures, landmarkIds = measurementModel.measure(state)

        mu += motionModel.exact_move(mu, motionCommand)

        eif.update(measures, landmarkIds, motionCommand, covarianceMotion)
        muEIF = eif.estimate()

        mus_simple[t] = np.squeeze(mu)
        mus_eif[t] = np.squeeze(muEIF)
        states[t] = np.squeeze(state)


    # LOG Final state
    print('\n')
    print('AFTER')
    print("EIF estimate :")
    print(muEIF)
    print("Real state :")
    print(state)
    print("Final Error :")
    print(state - muEIF)
    print("Final Max Error : %f" % max(state-muEIF))
    print("Final Norm Error : %f" % np.linalg.norm(state-muEIF))

    landmarks = state[robotFeaturesDim:].reshape(nbLandmark, 2)
    plt.figure()
    ax = plt.gca()
    for x, y in landmarks:
        ax.add_artist(Circle(xy=(x, y),
                      radius=detectionSize,
                      alpha=0.3))
    plt.scatter(landmarks[:, 0], landmarks[:, 1])

    plt.plot(states[:, 0], states[:, 1])
    plt.plot(mus_simple[:, 0], mus_simple[:, 1])
    plt.plot(mus_eif[:, 0], mus_eif[:, 1])

    plt.legend(['Real position', 'Simple estimate', 'EIF estimate'])
    plt.title("{0} landmarks".format(nbLandmark))
    plt.show()
