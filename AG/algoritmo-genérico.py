import numpy as np, random, time, operator, pandas as pd, matplotlib.pyplot as plt

# Gene
class City:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def distance(self, city):
		xDis = abs(self.x - city.x)
		yDis = abs(self.y - city.y)
		distance = np.sqrt((xDis ** 2) + (yDis ** 2))
		return distance
	
	def __repr__(self):
		return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
	def __init__(self, route):
		self.route = route
		self.distance = 0
		self.fitness= 0.0
	
	def routeDistance(self):
		if self.distance ==0:
			pathDistance = 0
			for i in range(0, len(self.route)):
				fromCity = self.route[i]
				toCity = None
				if i + 1 < len(self.route):
					toCity = self.route[i + 1]
				else:
					toCity = self.route[0]
				pathDistance += fromCity.distance(toCity)
			self.distance = pathDistance
		return self.distance
	
	def routeFitness(self):
		if self.fitness == 0:
			self.fitness = 1 / float(self.routeDistance())
		return self.fitness


# ===================== Declaracao de funcoes ==================

# Cria rota aleatoria entre as cidades
def createRoute(cityList):
	route = random.sample(cityList, len(cityList))
	return route

# gera populacao inicial
def initialPopulation(popSize, cityList):
	population = []

	for i in range(0, popSize):
		population.append(createRoute(cityList))
	return population

# Retorna a distancia total entre a populacao de cidades
def rankRoutes(population):
	fitnessResults = {}
	for i in range(0,len(population)):
		fitnessResults[i] = Fitness(population[i]).routeFitness()
	return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

# Retorna melhores resultados de uma populacao (cromossomo)
def selection(popRanked, eliteSize):
	selectionResults = []
	df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
	df['cum_sum'] = df.Fitness.cumsum()
	df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
	
	for i in range(0, eliteSize):
		selectionResults.append(popRanked[i][0])
	for i in range(0, len(popRanked) - eliteSize):
		pick = 100*random.random()
		for i in range(0, len(popRanked)):
			if pick <= df.iat[i,3]:
				selectionResults.append(popRanked[i][0])
				break
	return selectionResults

# selecao de pais para proxima geracao de filhos
def matingPool(population, selectionResults):
	matingpool = []
	for i in range(0, len(selectionResults)):
		index = selectionResults[i]
		matingpool.append(population[index])
	return matingpool

# Criando filhos a partir dos pais
def breed(parent1, parent2):
	child = []
	childP1 = []
	childP2 = []
	
	geneA = int(random.random() * len(parent1))
	geneB = int(random.random() * len(parent1))
	
	startGene = min(geneA, geneB)
	endGene = max(geneA, geneB)

	for i in range(startGene, endGene):
		childP1.append(parent1[i])
		
	childP2 = [item for item in parent2 if item not in childP1]

	child = childP1 + childP2
	return child


def breedPopulation(matingpool, eliteSize):
	children = []
	length = len(matingpool) - eliteSize
	pool = random.sample(matingpool, len(matingpool))

	for i in range(0,eliteSize):
		children.append(matingpool[i])
	
	for i in range(0, length):
		child = breed(pool[i], pool[len(matingpool)-i-1])
		children.append(child)
	return children

# mutacao de um cromossomo
def mutate(individual, mutationRate):
	for swapped in range(len(individual)):
		if(random.random() < mutationRate):
			swapWith = int(random.random() * len(individual))
			
			city1 = individual[swapped]
			city2 = individual[swapWith]
			
			individual[swapped] = city2
			individual[swapWith] = city1
	return individual


# mutacoes em uma populacao
def mutatePopulation(population, mutationRate):
	mutatedPop = []
	
	for ind in range(0, len(population)):
		mutatedInd = mutate(population[ind], mutationRate)
		mutatedPop.append(mutatedInd)
	return mutatedPop


# Gerar proxima geracao de cromossomos
def nextGeneration(currentGen, eliteSize, mutationRate):

	# Gera distancia da populacao atual
	popRanked = rankRoutes(currentGen)
	# Captura os melhores resultados
	selectionResults = selection(popRanked, eliteSize)
	# gerando pais
	matingpool = matingPool(currentGen, selectionResults)
	# gerando filhos
	children = breedPopulation(matingpool, eliteSize)
	nextGeneration = mutatePopulation(children, mutationRate)
	
	return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
	pop = initialPopulation(popSize, population)
	print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
	
	for i in range(0, generations):
		pop = nextGeneration(pop, eliteSize, mutationRate)
	
	print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
	bestRouteIndex = rankRoutes(pop)[0][0]
	bestRoute = pop[bestRouteIndex]
	return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
	pop = initialPopulation(popSize, population)
	progress = []
	progress.append(1 / rankRoutes(pop)[0][1])
	
	for i in range(0, generations):
		pop = nextGeneration(pop, eliteSize, mutationRate)
		progress.append(1 / rankRoutes(pop)[0][1])
	
	plt.plot(progress)
	plt.ylabel('Distance')
	plt.xlabel('Generation')
	plt.show()


def tuple2array(tuple):
	array = []
	for i in range(0, len(tuple)):
		subarray = []
		subarray.append(tuple[i].x)
		subarray.append(tuple[i].y)
		array.append(subarray)
	return array
# ===================== FIM DAS FUNCOES ===============================

# iniciando tempo de codigo
start_time = time.time()

cities = []
# for i in range(0,15):
# 	cities.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

cities = [City(x=60, y=200), City(x=180, y=200), City(x=80, y=180), City(x=140, y=180),
City(x=20, y=60), City(x=100, y=160), City(x=200, y=160), City(x=140, y=140),City(x=40, y=120), 
City(x=100, y=120), City(x=180, y=100), City(x=60, y=80), City(x=120, y=80), City(x=180, y=60), 
City(x=20, y=40)]

# Teste de distancia inicial igual a rota [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# print(Fitness(cities).routeDistance())

best = geneticAlgorithm(population=cities, popSize=100, eliteSize=20, mutationRate=0.01, generations=100)

# finalizando tempo de codigo
ending_time = time.time()

best2array = tuple2array(best)

print("Tempo de algoritmo")
print(ending_time - start_time)
print("Melhor rota final")
for i in range(0, len(best2array)):
	print(best[i])

plt.plot(
	[best2array[i % 15][0] for i in range(16)], 
	[best2array[i % 15][1] for i in range(16)], 'xb-');
plt.show()