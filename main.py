from functions import *
from utils import _MAX_GENERATIONS


population = initPopulation()

if __name__ == "__main__":
	gen = 0
	while gen < _MAX_GENERATIONS:
		population = findSolution(population, gen)
		print("\n")
		print(f"*****{gen}th Generation*****")
		if population == []:
			break
		gen+=1
    
	if gen == _MAX_GENERATIONS and population != []:
		print("Solution not foud")
		