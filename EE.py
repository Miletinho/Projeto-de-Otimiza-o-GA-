import numpy as np
from functions import *
from utils import _D, _EPSILON
import random 


#population = initPopulation()

def evaluateSigma(population, ps):
    for i in population:
        c = np.random.uniform(0.8, 1)
        if ps > _EPSILON:
            i[_D] /= c  # ampliar a busca -> exploration
        elif ps < _EPSILON:
            i[_D] *= c  # concentrar a busca -> explotation


# def populationMutation(newGeneration):
#     for i in range(0, _POP_SIZE):
#         rand = random.uniform(0, 1)
#         if rand < _MP:
#             mutatedChromosome = mutationES(newGeneration[i])
#             newGeneration[i] = mutatedChromosome
#     newGeneration = np.array(newGeneration)
#     return newGeneration

def calculateMutationSuccess(fitnessTotal, newFitnessTotal, mutations):
    mutationSuccess = 0
    for i in range(mutations):
        if newFitnessTotal[i] > fitnessTotal[i]:    
            mutationSuccess += 1
    return mutationSuccess

def findSolutionPart2(population, generation):
    parents = selection(population, type="random")
    fitnessTotal = evalFitness(population)
    newGeneration = genChildren(parents, population, type="es")
    newFitnessTotal = evalFitness(newGeneration)
    mutations = len(newGeneration)

    mutationSuccess = calculateMutationSuccess(fitnessTotal, newFitnessTotal, mutations)
    # print(mutationSuccess)
    popWithFitness = sortByFitness(newGeneration)
    
    print(popWithFitness[0][1])
    if popWithFitness[0][1] < 0.001:    
        count = 0
        solutions = []
        while popWithFitness[count][1] < 0.001:
            solutions.append(popWithFitness[count][0])
            count += 1
        print(f"Solution(s) has been found in {generation}th generation:") 
        print(solutions)
        return [[], mutationSuccess, mutations]

    return [newGeneration, mutationSuccess, mutations]
