import numpy as np
from functions import *
from utils import _D, _EPSILON, _MP, _POP_SIZE
from random import gauss
import random 


#population = initPopulation()

def mutation(chromosome, sigma):
    # randomly pick a gene to mutate from the chromosome
    globalTal = 1/np.sqrt(2*_D)
    localTal = 1/np.sqrt(2*np.sqrt(_D))
    newSigma = sigma * np.exp(globalTal*gauss(0,1) + localTal*gauss(0,1))
    
    for i in range(_D): 
        chromosome[i] += newSigma*gauss(0,1)

    return chromosome


def evaluateSigma(sigma, ps):
    c = np.random.uniform(0.8, 1)
    if ps > _EPSILON:
        return sigma/c  # ampliar a busca -> exploration
    elif ps < _EPSILON:
        return sigma*c  # concentrar a busca -> explotation

    return sigma


def populationMutation(newGeneration, sigma):
    for i in range(0, _POP_SIZE):
        rand = random.uniform(0, 1)
        if rand < _MP:
            mutatedChromosome = mutation(newGeneration[i], sigma)
            newGeneration[i] = mutatedChromosome
    newGeneration = np.array(newGeneration)
    return newGeneration

def calculateMutationSuccess(fitnessTotal, newFitnessTotal, mutations):
    mutationSuccess = 0
    for i in range(mutations):
        if newFitnessTotal[i] > fitnessTotal[i]:
            mutationSuccess += 1
    return mutationSuccess

def findSolutionPart2(population, generation, sigma):
    fitnessTotal = evalFitness(population)

    newGeneration = populationMutation(population, sigma)

    newFitnessTotal = evalFitness(newGeneration)

    mutations = len(newGeneration)

    mutationSuccess = calculateMutationSuccess(fitnessTotal, newFitnessTotal, mutations)

    popWithFitness = sortByFitness(population)
    
    print(popWithFitness[0][1])
    if popWithFitness[0][1] < 0.001:
        count = 0
        solutions = []
        while popWithFitness[count][1] < 0.001:
            solutions.append(popWithFitness[count][0])
            count += 1
        print(f"Solution(s) has been found in {generation}th generation:") 
        print(solutions)
        return []

    return [newGeneration, mutationSuccess, mutations]
