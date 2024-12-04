import numpy as np
import matplotlib.pyplot as plt


def getBinomial(k):
    i = int(np.random.binomial(2 * k, 0.5))
    j = (2 * k) - i
    return i, j


def GillespieSSA_1():
    stateVector = np.zeros(
        200)  # A vector to hold the number of particles in each state, we assume that a particle in state 201 or larger will not be generated in such a short timeframe
    binnedData = np.zeros(
        (1002,
         200))  # This keeps track of the state vectors at discretised bins of time (width 0.01 time units) alongside initial and final states.
    stateVector[1] = 1
    binnedData[0] = stateVector
    t = 0  # A count for the time value
    N = 1  # A count for the total number of particles, note when a reaction occurs. One new particle is created

    while t < 10:  # iterate the algorithm for 10 time units
        r1 = np.random.random()  # generate 2 uniformly random numbers in the interval (0,1), r1 and r2
        r2 = np.random.random()
        tau = np.log(1 / r1) / N  # determines time until next reaction
        k = 0

        tempSum = 0  # temporary value used to find k satisfying 4d in Gillspie SSA in lecture notes
        # This for loop find the reaction to occur at time: t + tau
        checkValue = N * r2  # defined to reduce number of calculations performed
        for k, Nk in enumerate(stateVector):
            if (tempSum <= checkValue) and (checkValue < (tempSum + Nk)):  # checks to see if reaction k occurs
                break
            else:
                tempSum += Nk

        i, j = getBinomial(k)  # obtains i,j according to binomial distribution
        # update stateVector

        stateVector[k] -= 1
        stateVector[i] += 1
        stateVector[j] += 1

        # update t, N, and binnedData
        t += tau
        N += 1
        bin = int(np.floor(100 * t)) + 1 # determines which bin to place stateVector in
        if bin <= 1000:  # catch case for t > 10
            binnedData[bin] = stateVector
        else:
            binnedData[1001] = stateVector
    return binnedData


def padData(data):
    for i in range(np.shape(data)[0]):
        if not np.any(data[i]):
            data[i] = data[i-1]
    return data

def endStateHistogram(data):
    endState = np.zeros(20)
    for i in range(1000):
        for j in range(20):
            endState[j] += data[i, 1001, j] / 1000

    return endState

def moment(stateVec, l):
    N = np.sum(stateVec)
    M = 0
    for k,Nk in enumerate(stateVec):
        M += (k**l) * Nk
    return (M/N)

def getMomentTimeSeries(data,l):
    timeSeries = np.zeros(1001)
    for i in range(1000):
        for j in range (1,1002):
            timeSeries[j-1] += moment(data[i,j],l)/1000
    return timeSeries

def s1Iters():
    allBinnedData = np.zeros((1000,1002,200))
    for i in range(1000):
        if i % 50 == 0:
            print(i)
        allBinnedData[i] = padData(GillespieSSA_1())

    endStateHistogram(allBinnedData)
    moment1 = getMomentTimeSeries(allBinnedData,1)
    moment2 = getMomentTimeSeries(allBinnedData,2)
    times = np.arange(0,10.01,0.01)
    theoreticalM2 = times + 1

    plt.plot(times,moment1)
    plt.axhline(1)
    plt.show()

    plt.plot(times,moment2)
    plt.plot(times,theoreticalM2)
    plt.show()





s1Iters()