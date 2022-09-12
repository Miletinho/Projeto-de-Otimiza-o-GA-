import time
from statistics import mean
from functions import *
from utils import _MAX_GENERATIONS,_POP_SIZE
from EE import *
import itertools
import numpy as np
import pandas as pd
import sys
import random
import decimal


def main(totalConvergence):
    option = input("""
                    Qual parte utilizar?
                        1. Parte 1
                        2. Parte 2
                    Opção:""")
    gen = 0
    population = initPopulation()
    allGenerationsFitness = []
    if option == "1":
        while gen < _MAX_GENERATIONS:
            [population, fit] = findSolutionPart1(population, gen)
            allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
            if population == []:
                count = 0
                for i in fit:
                    print(i)
                    if round(i,2) < 0.01:
                        count+=1
                if totalConvergence:
                    if count == 2000:
                        print("Number of Generations to reach total convergence: ", gen+1)
                        return [gen,mean(allGenerationsFitness),count]
                else:
                    print("Number of Generations to reach individual convergence: ", gen+1)
                    print(count,f"Solution(s) has been found in {gen}th generation:") 
                    return [gen, mean(allGenerationsFitness), count] 
                break
            gen+=1
        return [gen, mean(allGenerationsFitness), 0] 
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
                
            #print("\n")
            #print(f"*****{gen}th Generation*****")
            if population == []:
                count = 0
                for i in fit:
                    print(i)
                    if round(i,2) < 0.01:
                        count+=1
                if totalConvergence:
                    if count == 2000:
                        print("Number of Generations to reach total convergence: ", gen+1)
                        return [gen,mean(allGenerationsFitness),count]
                else:
                    print("Number of Generations to reach individual convergence: ", gen+1)
                    print(count,f"Solution(s) has been found in {gen}th generation:") 
                    return [gen, mean(allGenerationsFitness), count] 
                break
            gen+=1
        
    if gen == _MAX_GENERATIONS and population != []:
        print("Solution not found")
    return [gen,]

if __name__ == "__main__":
    
    
    interation = main(False)
    #getTime(startTime)
        