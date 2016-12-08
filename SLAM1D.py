import numpy as np
from numpy.linalg import inv
from numpy import dot
from matplotlib import pyplot as plt
import seaborn

class BasicMovement:
    def __init__(self, maxSpeed, covariance, robotFeaturesDim):
        self.maxSpeed = maxSpeed
        self.covariance = np.atleast_2d(covariance)
        self.robotFeaturesDim = robotFeaturesDim

    #  Input the real state
    def move(self, state, covariance=None, command=None):
        move = self.__choose_command(state) if command is None else command
        noise = self.__get_noise(state, covariance)
        newState = state + move + noise
        return newState, move

    def __choose_command(self, state):  # TO CHANGE FOR 2D
        dim = self.robotFeaturesDim
        way = 1 - 2 * np.random.randint(2)
        speed = self.maxSpeed * np.random.rand()
        newCommand = np.zeros_like(state)
        newCommand[:dim] = way * speed  # TO CHANGE FOR 2D
        return newCommand

    def __get_noise(self, state, covariance):
        dim = self.robotFeaturesDim
        noise = np.zeros_like(state)
        covariance = self.covariance if covariance is None else covariance
        noise[:dim] = np.random.multivariate_normal(np.zeros(dim), covariance, 1).T
        return noise


class BasicMeasurement:
    def __init__(self, covariance, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize=0):
        self.covariance = np.atleast_2d(covariance)
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.measureFunction = measureFunction
        self.gradMeasureFunction = gradMeasureFunction
        self.detectionSize = detectionSize

    #  Input the real state
    def measure(self, state):
        dim = state.shape[0]
        dimR = self.robotFeaturesDim
        dimE = self.envFeaturesDim
        rState = state[:dimR]
        envState = state[dimR:]
        nbLandmark = (dim - dimR) / dimE

        mes = np.zeros(nbLandmark)
        landmarkIds = np.zeros(nbLandmark)
        j = 0
        for i, landmark in enumerate(envState.reshape((nbLandmark, dimE, 1))):
            if (np.linalg.norm(rState - landmark) < self.detectionSize) or (self.detectionSize is 0):
                mes[j] = self.measureFunction(rState, landmark)
                landmarkIds[j] = int(i)
                j += 1
        mes = mes[:j]
        landmarkIds = landmarkIds[:j]

        mes = np.array(mes)
        noise = self.__get_noise(mes)
        mes += noise  # TO CHANGE for 2D ---------------------------------------------
        return mes.reshape((len(mes), 1)), landmarkIds

    def __get_noise(self, mes):
        noise = np.squeeze(np.random.multivariate_normal(np.zeros(self.envFeaturesDim), self.covariance, len(mes)))
        return noise


class EIFModel:
    def __init__(self, dimension, robotFeaturesDim, envFeaturesDim, motionModel, mesModel, covMes, muInitial):
        self.robotFeaturesDim = robotFeaturesDim
        self.envFeaturesDim = envFeaturesDim
        self.dimension = dimension

        self.H = np.eye(dimension)
        self.b = dot(muInitial.T, self.H)  # np.zeros((1, dimension))
        self.S = np.zeros(dimension * robotFeaturesDim).reshape((dimension, robotFeaturesDim))
        self.S[:robotFeaturesDim] = np.eye(robotFeaturesDim)
        self.invZ = inv(covMes)
        self.motionModel = motionModel
        self.mesModel = mesModel

    def update(self, measures, landmarkIds, command, U):
        self.__motion_update(command, U)
        # print(eif.estimate())
        for ldmIndex, ldmMes in zip(landmarkIds, measures):
            self.__measurement_update(ldmMes, ldmIndex)
        # print(eif.estimate())
        return self.H, self.b

    def __motion_update(self, command, U):
        previousMeanState = self.estimate()
        _, meanStateChange = self.motionModel.move(previousMeanState, command=command)
        self.H = inv(inv(self.H) + dot(dot(self.S, U), self.S.T))
        self.b = dot((previousMeanState + meanStateChange).T,  self.H)

    def __measurement_update(self, ldmMes, ldmIndex):
        mu = self.estimate()
        meanMes, gradMeanMes = self.__get_mean_measurement_params(mu, ldmIndex)

        z = np.atleast_2d(ldmMes)
        zM = np.atleast_2d(meanMes)
        C = gradMeanMes

        # print("Safety Check : %r" % (meanMes == dot(C.T, mu))[0, 0])
        # print("mu:")
        # print(mu)
        # print("z : %f" % z)
        # print("zM : %f" % zM)
        self.H += dot(dot(C, self.invZ),  C.T)
        self.b += dot(dot((z - zM + dot(C.T, mu)).T, self.invZ), C.T)

    def __get_mean_measurement_params(self, mu, ldmIndex):
        realIndex = self.robotFeaturesDim + ldmIndex * self.envFeaturesDim
        ldmMeanState = mu[realIndex: realIndex + self.envFeaturesDim]
        rMeanState = mu[:self.robotFeaturesDim]

        meanMes = self.mesModel.measureFunction(rMeanState, ldmMeanState)
        gradMeanMes = self.mesModel.gradMeasureFunction(mu, ldmIndex)
        return meanMes, gradMeanMes

    def estimate(self, H=None, b=None):
        H = self.H if H is None else H
        b = self.b if b is None else b
        return dot(b, inv(H)).T


measureFunction = lambda rState, landmark: np.sign(landmark[0, 0] - rState[0, 0]) * np.linalg.norm(rState - landmark)
def gradMeasureFunction(state, ldmIndex):
    grad = np.zeros_like(state)
    grad[0] = -1
    grad[ldmIndex+1] = 1
    return grad

T = 10000  # Number of timesteps
nbLandmark = 10
maxSpeed = 3
robotFeaturesDim = 1
envFeaturesDim = 1
dimension = robotFeaturesDim + nbLandmark * envFeaturesDim

# Detection parameters
spaceBetween = 100
detectionSize = 15

covarianceMotion = np.eye(robotFeaturesDim) * 10  # motion noise variance
covarianceMeasurements = np.eye(envFeaturesDim) * 10  # measurement noise variance

motionModel = BasicMovement(maxSpeed, covarianceMotion, robotFeaturesDim)
measurementModel = BasicMeasurement(covarianceMeasurements, robotFeaturesDim, envFeaturesDim, measureFunction, gradMeasureFunction, detectionSize)
state = np.zeros((dimension, 1))  # Real robot state
state[1:] = np.arange(0, nbLandmark * spaceBetween, spaceBetween).reshape(nbLandmark, 1)
# state[1] = -100

mu = np.zeros_like(state)  # Estimated robot state basic
# mu[1] = 50
# mu[1:nbLandmark+1] = np.arange(0, nbLandmark * spaceBetween, spaceBetween).reshape(nbLandmark, 1)
mu = state.copy()
mu[1:] += np.random.normal(0, covarianceMeasurements, nbLandmark).reshape(nbLandmark, 1)


muEIF = np.zeros_like(state)  # Estimated robot state using EIF Algorithm
# muEIF[1] = 50
# muEIF[1:nbLandmark+1] = np.arange(0, nbLandmark * spaceBetween, spaceBetween).reshape(nbLandmark, 1)
muEIF = mu.copy()


eif = EIFModel(dimension, robotFeaturesDim, envFeaturesDim, motionModel, measurementModel, covarianceMeasurements, mu)

mus_simple = np.zeros((T, dimension))
mus_eif = np.zeros((T, dimension))
states = np.zeros((T, dimension))

mus_simple[0] = np.squeeze(mu)
mus_eif[0] = np.squeeze(muEIF)
states[0] = np.squeeze(state)

print("BEFORE")
print("EIF estimate :")
print(muEIF)
print("Real state :")
print(state)
print('\n')

for t in range(1, T):
    state, motionCommand = motionModel.move(state)
    measures, landmarkIds = measurementModel.measure(state)

    H, b = eif.update(measures, landmarkIds, motionCommand, covarianceMotion)

    mu += motionCommand
    # + measurement !
    muEIF = eif.estimate()

    mus_simple[t] = np.squeeze(mu)
    mus_eif[t] = np.squeeze(muEIF)
    states[t] = np.squeeze(state)


print('\n')
print('AFTER')
print("EIF estimate :")
print(muEIF)
print("Real state :")
print(state)

plt.figure()
plt.plot(states[:, 0])
plt.plot(mus_simple[:, 0])
plt.plot(mus_eif[:, 0])
plt.legend(['Real position', 'Simple estimate', 'EIF estimate'])
plt.title("{0} landmarks".format(nbLandmark))
plt.show()
