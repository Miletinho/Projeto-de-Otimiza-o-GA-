import time
from functions import *
from utils import _MAX_GENERATIONS,_POP_SIZE
from EE import *


population = initPopulation()


if __name__ == "__main__":
	option = input("""
	Qual parte utilizar?
		1. Parte 1
		2. Parte 2
	Opção:
	""")
	gen = 0

	#startTime = time.time()
 
	if option == "1":

		while gen < _MAX_GENERATIONS:
			population = findSolutionPart1(population, gen)
			print("\n")
			print(f"*****{gen}th Generation*****")
			if population == []:
				break
			gen+=1
		
	else:
		sigma = 1

		for i in range(_POP_SIZE):
			population[i].append(sigma)

		while gen < _MAX_GENERATIONS:

			numberOfMutations = 0
			numberOfMutationSuccess = 0
			[population, mutationSuccess, mutations] = findSolutionPart2(population, gen)
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
				break
			gen+=1
		
	if gen == _MAX_GENERATIONS and population != []:
		print("Solution not found")

	#getTime(startTime)
		