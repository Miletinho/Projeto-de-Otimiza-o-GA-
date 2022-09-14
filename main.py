import sys
import time
from statistics import mean
from functions import *
from utils import _EXECUTIONS, _FITNESS, _MAX_GENERATIONS,_POP_SIZE,_ROUND
from EE import *
import numpy as np


def evaluateExecutions(allGen, allFitness, counter, execTime):
    meanGen = np.average(allGen)
    stdGen = np.std(allGen)
    meanFitness = np.average(allFitness)
    stdFitness = np.std(allFitness)
    nConvergence = sum(counter)
    meanConvergence = np.average(counter)
    meanExecTime = np.average(execTime)
    return [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime]

def printEvaluation(meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime):
    print("Em que iteração o algoritmo convergiu, em média: ", round(meanGen, 3))
    print("Desvio Padrão de em quantas iterações o algoritmo convergiu: ", round(stdGen, 3))
    print("Fitness médio alcançado nas 3 execuções : ", round(meanFitness, 3))
    print("Desvio padrão dos Fitness alcançados nas 3 execuções: ", round(stdFitness, 3))
    print("Em quantas execuções o algoritmo convergiu: ", str(min(nConvergence, 3)) + "/3")
    print("Número de indivíduos que convergiram: ", nConvergence)
    print("Número de indivíduos que convergiram por execução, em média: ", round(meanConvergence, 3))
    print("Tempo médio de execução das 3 execuções: ", round(meanExecTime, 3), " segundos")

def main(totalConvergence, option):
    gen = 0
    population = initPopulation()
    allGenerationsFitness = []
    startTime = time.time()
    if option == "1":
        while gen < _MAX_GENERATIONS:
            [population, fit] = findSolutionPart1(population, gen)
            allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
            if population == []:
                count = 0
                for i in fit:
                    # print(i)
                    if i < _FITNESS:
                        count+=1
                if totalConvergence:
                    if count == _POP_SIZE:
                        print("Number of Generations to reach total convergence: ", gen+1)
                        executionTime = getTime(startTime)
                        return [gen,mean(allGenerationsFitness),count, executionTime]
                else:
                    print("Number of Generations to reach individual convergence: ", gen+1)
                    print(count,f"Solution(s) has been found in {gen}th generation:") 
                    executionTime = getTime(startTime)
                    print("Count: ", count)
                    return [gen, mean(allGenerationsFitness), count, executionTime] 
                break
            gen+=1
        executionTime = getTime(startTime)
        return [gen, mean(allGenerationsFitness), 0, executionTime] 
    else:
        sigma = 1

        for i in range(_POP_SIZE):
            population[i].append(sigma)

        while gen < _MAX_GENERATIONS:

            numberOfMutations = 0
            numberOfMutationSuccess = 0
            [population, mutationSuccess, mutations,fit] = findSolutionPart2(population, gen)
            allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
            numberOfMutationSuccess += mutationSuccess
            numberOfMutations += mutations
            # ---- 1/5 da regra de sucesso:
            #   -> se mais de 1/5 das mutações levar a uma melhora, a força da mutação é aumentada (sigma=sigma/c), se == 1/5, mantém (sigma = sigma), se não é diminuída (sigma = sigma*c).
            # ps é a % de mutações com sucesso
            if gen % 5 == 0:
                ps = numberOfMutationSuccess/numberOfMutations
                # print(ps)
                evaluateSigma(population, ps)
                
            print("\n")
            print(f"*****{gen}th Generation*****")
            if population == []:
                count = 0
                for i in fit:
                    if round(i,_ROUND) < _FITNESS:
                        # print("fit:", i)
                        count+=1
                if totalConvergence:
                    if count == _POP_SIZE:
                        print("Number of Generations to reach total convergence: ", gen+1)
                        executionTime = getTime(startTime)
                        return [gen,mean(allGenerationsFitness),count, executionTime]
                else:
                    print("Number of Generations to reach individual convergence: ", gen+1)
                    print(count,f"Solution(s) has been found in {gen}th generation:") 
                    executionTime = getTime(startTime)
                    return [gen, mean(allGenerationsFitness), count, executionTime] 
                break
            gen+=1
    
    if gen == _MAX_GENERATIONS and population != []:
        print("Solution not found")

    executionTime = getTime(startTime)
    return [gen, mean(allGenerationsFitness), 0, executionTime]

if __name__ == "__main__":

    option = input("""
                Qual parte utilizar?
                    1. Parte 1
                    2. Parte 2
                Opção:""")
    option = input("""
                Convergência?
                    1. Total
                    2. Individual
                Opção:""")
    n=0
    generations = []
    fitness = [] 
    counts = []
    times = []
    # meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime
    while n < _EXECUTIONS:
        [gen, fit, count, executionTime] = main(False, option)
        generations.append(gen)
        fitness.append(fit)
        counts.append(count)
        times.append(executionTime)
        n+=1
    
    [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime] = evaluateExecutions(generations, fitness, counts, times)
    printEvaluation(meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime)