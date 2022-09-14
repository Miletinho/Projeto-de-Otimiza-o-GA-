from hashlib import new
import random
import time 
import numpy as np
from random import random as rand
from utils import _A, _B, _C, _D, _START, _END, _POP_SIZE, _MP, _CP,_MP_ES,_LOCAL_THAO,_GLOBAL_THAO, _ROUND,_FITNESS

def ackleyFitnessFunction(x):
    sum1 = 0
    sum2 = 0
    for i in range(_D):
        sum1 += x[i]**2
        sum2 += np.cos(_C*x[i])

    fitness = -_A*np.exp(-_B * np.sqrt(sum1/_D)) - np.exp(sum2/_D) + _A + np.exp(1)

    return fitness
     
    
def generateChromosome():
    chromosome = []
    for i in range(_D):
        chromosome.append(rand() * (_END - _START) + _START)

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

def rankingParents(population):
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

def getFitnessProbability(population):
    fitness = evalFitness(population)
    fitnessProbability = []

    totalFitness = 0
    for i in fitness:
        totalFitness += 1/(i+1)
    
    for i in fitness:
        if i == fitness[-1]:
            fitnessProbability.append(1-sum(fitnessProbability))
        else:
            fitnessProbability.append(1/(i+1)/totalFitness)

    return fitnessProbability

def rouletteParents(population):
    fitnessProbability = getFitnessProbability(population)
    parents = []
    for i in range(_POP_SIZE):
        r = random.choices(population, weights=fitnessProbability, k=1)
        parents.append(r[0])

    return parents

def calcParentFit(parents):
    parentFit = abs(ackleyFitnessFunction(parents))
    return parentFit

def pickParents(population):
    parents = []
    for k in range(_POP_SIZE):
        firstParent = random.choice(population)
        secondParent  = random.choice(population)
        firstParentFit = calcParentFit(firstParent)
        secondParentFit = calcParentFit(secondParent)
        if secondParentFit < firstParentFit:
            parents.append(secondParent)
        else:
            parents.append(firstParent)
    return parents

def randomSelection(population):
    parents = []
    for i in range(_POP_SIZE//2):
        chosen1, chosen2 = random.sample(population, 2)
        parents.append(chosen1)
        parents.append(chosen2)
    return parents

def selection(population, type='ranking'):
    if type == 'ranking': 
        return rankingParents(population)
    elif type == 'pickParents':
        return pickParents(population)
    elif type == 'roulette':
        return rouletteParents(population)
    else:
        return randomSelection(population)



def onePointCrossover(parents):
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

def intermediateRecombination(parents):
    # randomly generates a number between 0 and 1 that indicates whether or not there is a crossover
    r = random.uniform(0,1)
    children = []
    if r < _CP:
        # generate parents 
        parent1, parent2 = parents
        child1 = []
        child2 = []
        for g in range(_D):
            ratio = rand()
            child1.append(parent1[g] + ratio * (parent2[g] - parent1[g]))
            ratio = rand()
            child2.append(parent2[g] + ratio * (parent1[g] - parent2[g]))

        if len(parent1) == (_D + 1):
            child1.append(parent1[_D])
            child2.append(parent2[_D])

        children = np.array([child1, child2])
        return children
    return parents

def crossover(parents, type="onePoint"):

    if type == 'onePoint':
        return onePointCrossover(parents)
    else:
        return intermediateRecombination(parents)


def randomMutation(chromosome):
    for i in range(_D):
        # randomly generates a number between 0 and 1 for each gene on a chromosome that indicates whether or not there is a mutation
        r = random.uniform(0,1)
        if r < _MP:
            chromosome[i] = random.uniform(_START, _END)
    return chromosome

def gaussianMutation(chromosome):
    for i in range(_D):
        r = random.uniform(0,1)
        if r < _MP:
            chromosome[i] = chromosome[i] + random.gauss(0,1)
    return chromosome


def mutationES(chromosome):
    # randomly pick a gene to mutate from the chromosome
    for i in range(_D): 
        r = random.uniform(0,1)
        chromosome[_D] *= np.exp(_GLOBAL_THAO*np.random.normal(0,1) + _LOCAL_THAO*np.random.normal(0,1))
        if r < _MP_ES:
            chromosome[i] += chromosome[_D]*np.random.normal(0,1)

    return chromosome

def mutation(chromosome, type='random'):

    if type == 'random':
        return randomMutation(chromosome)
    elif type == 'gauss':
        return gaussianMutation(chromosome)
    elif type == 'es':
        return mutationES(chromosome)

def rouletteSelection(population):
    fitnessProbability = getFitnessProbability(population)
    survivors = []
    wasChosen = [False for i in range(2*_POP_SIZE)]
    for i in range(_POP_SIZE):
        index = random.choices(range(2*_POP_SIZE), weights=fitnessProbability, k=1) 
        while(wasChosen[index[0]]):
            index = random.choices(range(2*_POP_SIZE), weights=fitnessProbability, k=1)
        survivors.append(population[index[0]])
        wasChosen[index[0]] = True
        

    return survivors

def genChildren(parents, population, type='gauss', roulette=False):
    newGeneration = []
    fit = []
    for i in range(0, _POP_SIZE, 2):
        children = crossover([parents[i], parents[i+1]], type="intermediate")
        for c in children:
            c = mutation(c, type=type)
            population.append(c)
        
    population = sortByFitness(population)

    if not roulette:
        population = population[:_POP_SIZE]
    
    for p in population:
        newGeneration.append(p[0])
        fit.append(p[1])
    if roulette:
        newGeneration = rouletteSelection(newGeneration)

    return newGeneration, fit

def findSolutionPart1(population, generation):
    popWithFitness = sortByFitness(population)
    print(round(popWithFitness[0][1],_ROUND))
    newfit = []
    if round(popWithFitness[0][1],_ROUND) < _FITNESS:
        #print(solutions)
        for i in popWithFitness:
            newfit.append(i[1])
        return [],newfit
    else:
        parents = selection(population, type="random")
        newGeneration,fit = genChildren(parents, population)
    return newGeneration,fit
    

def getTime(startTime):
    execTime = round(time.time() - startTime, 3)
    if execTime > 60:
        print("Tempo de execução: ", round(execTime/60, 3), " minutos")
    else:
        print("Tempo de execução: ", execTime, " segundos")
    return execTime