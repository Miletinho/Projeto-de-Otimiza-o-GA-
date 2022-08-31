from audioop import reverse
from calendar import c
from mimetypes import init
import random
from urllib import response
import numpy as np
from utils import _A, _B, _C, _D, _START, _END, _POP_SIZE, _MP, _CP

def ackleyFitnessFunction(x):
    sum1 = 0
    sum2 = 0
    for xi in x:
        sum1 += xi**2
        sum2 += np.cos(_C*xi)

    fitness = -_A*np.exp(-_B * np.sqrt(sum1/_D)) - np.exp(sum2/_D) + _A + np.exp(1)

    return fitness
     
    
def generateChromosome():
    chromosome = []
    for i in range(_D):
        chromosome.append(random.uniform(_START,_END))

    return chromosome

    
def initPopulation():
    initialpopulation = []

    for i in range(_POP_SIZE):
        initialpopulation.append(generateChromosome())

    return initialpopulation

def evalFitness(population):
    fitness = []

    for x in population:
        ackleyFit = ackleyFitnessFunction(x)
        fitness.append(ackleyFit)

    return fitness

def getFit(parent):
    return parent[1]

def sortByFitness(individuals):
    individualsWithFitness = []
    
    for i in individuals:
        individualsWithFitness.append([i, ackleyFitnessFunction(i)])  
    
    individualsWithFitness.sort(key=getFit, reverse=False)

    return individualsWithFitness

def pickParents(population):
    parents = []
    # as each pair of parents generates 2 children, the number of pairs will be equal to half of the population size
    numberOfPairs = int(len(population)/2)
    for i in range(numberOfPairs):
        parentsChoice = random.sample(population, 5)

        parentsWithFitness = sortByFitness(parentsChoice)

        parentsWithFitness = parentsWithFitness[:2]
        
        for p in parentsWithFitness:
            parents.append(p[0])

    return parents


def crossover(parents):
    # randomly generates a number between 0 and 1 that indicates whether or not there is a crossover
    r = random.uniform(0,1)
    children = []
    if r < _CP:
        # generate parents 
        [parent1, parent2] = parents
        crossPoint = random.randint(1,_D-1)
        
        # create children by switching parents parts
        child1 = np.append(parent1[:crossPoint], parent2[crossPoint:_D])
        child2 = np.append(parent2[:crossPoint], parent1[crossPoint:_D])
        children = np.array([child1, child2])
        return children
    return parents   

def mutation(chromosome):

    for i in range(_D):
        # randomly generates a number between 0 and 1 for each gene on a chromosome that indicates whether or not there is a mutation
        r = random.uniform(0,1)
        if r < _MP:
            chromosome[i] = random.uniform(_START, _END)

    return chromosome

def genChildren(parents, population):
    newGeneration = []
    
    for i in range(0, _POP_SIZE, 2):
        children = crossover([parents[i], parents[i+1]])
        for c in children:
            c = mutation(c)
            population.append(c)
        
    population = sortByFitness(population)
    
    population = population[:_POP_SIZE]

    for p in population:
        newGeneration.append(p[0])

    return newGeneration

def findSolution(population, generation):
    popWithFitness = sortByFitness(population)
    print(popWithFitness[0][1])
    if popWithFitness[0][1] < 0.0001:
        count = 0
        solutions = []
        while popWithFitness[count][1] < 0.0001:
            solutions.append(popWithFitness[count][0])
            count += 1
        print(f"Solution(s) has been found in {generation}th generation:") 
        print(solutions)
        return []
    else:
        parents = pickParents(population)
        newGeneration = genChildren(parents, population)

    return newGeneration
    