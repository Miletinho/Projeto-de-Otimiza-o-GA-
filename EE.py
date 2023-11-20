import numpy as np
from functions import *
from utils import _EPSILON, _POP_SIZE, constants

def evaluateSigma(population, ps):
    for i in population:
        d = len(i) - 1 
        c = np.random.uniform(0.8, 1)
        if ps > _EPSILON:
            i[d] /= c  # ampliar a busca -> exploration
        elif ps < _EPSILON:
            i[d] *= c  # concentrar a busca -> explotation

def calculateMutationSuccess(fitnessTotal, newFitnessTotal, mutations):
    mutationSuccess = 0
    for i in range(mutations):
        if newFitnessTotal[i] > fitnessTotal[i]:    
            mutationSuccess += 1
    return mutationSuccess

def findSolutionPart2(population, generation, func, total=False):
    parents = selection(population, type="tournament", func=func)
    fitnessTotal = evalFitness(population, func)
    newGeneration, fit = genChildren(parents, population, type="es", func=func)
    mutations = len(newGeneration)

    mutationSuccess = calculateMutationSuccess(fitnessTotal, fit, mutations)
    popWithFitness = sortByFitness(newGeneration, func)
    print(popWithFitness[0][1])
    count = 0
    if popWithFitness[0][1] <= constants[func][3] and popWithFitness[0][1] >= (-constants[func][3]):
        for i in fit:
            if i <= constants[func][3] and i >= (-constants[func][3]):
                print("fit!!", i)
                count+=1
        if total and count == _POP_SIZE:
            return [[], mutationSuccess, mutations, fit, count]
        elif not total:
            return [[], mutationSuccess, mutations, fit, count]

    return [newGeneration, mutationSuccess, mutations, fit, 0]