import numpy as np
import random
import imageio
from PIL import Image
from deap import base
from deap import creator
from deap import tools

#open an image to be painted
to_paint = Image.open('pic.jpg')


def genetic_painter(png, generations, init_population):
    #convert the image to a 2d list of binaries (list of chromosomes)
    def png_to_chromosomes(png):
        png = png.convert('1')
        png = np.asarray(png)
        return png.tolist()

    png = png_to_chromosomes(png)

    painting = []
    preliminary = [[] for _ in range(generations)]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, len(png[0]))
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # define the fitness function to be maximize

    def objective(individual, row):
        fitness = 0
        for column in range(len(individual)):
            if bool(individual[column]) == png[row][column]:
                fitness += 1
        return fitness,

    toolbox.register("evaluate", objective)
    # set the crossover operator
    toolbox.register("mate", tools.cxUniform)
    # set a mutation operator with a probability to flip bit in chromosome
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    # operator to select individuals for breeding such that each individual of the current generation
    # is replaced by the fittest of three (tournament size equals 3) individuals that are picked randomly
    toolbox.register("select", tools.selTournament, tournsize=3)

    # so the optimization begins
    def main():

        print(len(png), " rows")
        for row in range(len(png)):
            rows = [row for _ in range(len(png))]
            # we create an initial population of n individuals where each individual is a list of booleans
            # such that each gene stands for an either black or white pixel
            pop = toolbox.population(n=init_population)
            # CXPB  is the probability with which two individuals breed
            # MUTPB is the probability for mutating an individual
            CXPB, MUTPB = 0.5, 0.2
            # we assess the entire population
            fitnesses = list(map(toolbox.evaluate, pop, rows))

            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # variable g that keeps track of the number of generations
            g = 0

            # the evolution begins
            while g < generations:
                # a new generation
                g = g + 1
                # individuals to form the next generation are selected
                offspring = toolbox.select(pop, len(pop))
                # we copy the selected individuals for later use
                offspring = list(map(toolbox.clone, offspring))
                # algorithm applies crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    # breed two individuals with probability CXPB
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2, indpb=0.3)
                        # fitness values of the children must be recalculated later
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    # algorithm mutates an individual with probability MUTPB
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # the individuals with an invalid fitness are assessed
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind, rows)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                best_in_gen = tools.selBest(pop, 1)[0]
                preliminary[g-1].append(list(map(lambda x: bool(x), best_in_gen)))
                # The population is entirely replaced by the offspring
                pop[:] = offspring

            best_ind = tools.selBest(pop, 1)[0]

            print("Row {}. The best individual's fitness is {:0.2f}".format(row,(best_ind.fitness.values[0])/len(png[0])))
            # the best of each generation is stored to be later used in the slideshow
            painting.append(list(map(lambda x: bool(x), best_ind)))
        return painting

    if __name__ == "__main__":
        # array of bits is converted to unicode and saved as .png images
        painting = np.asarray(main()).astype('uint8')*255
        image = Image.fromarray(painting)
        image.save('result.png')

        images = []
        frames = []
        # making a .gif out of images
        for img in range(init_population):
            preliminary[img] = np.asarray(preliminary[img]).astype('uint8')*255
            images.append(Image.fromarray(preliminary[img]))
            images[img].save('{}.png'.format(img))

        for image in range(init_population):
            frames.append(imageio.imread('{}.png'.format(image)))

        imageio.mimsave('movie.gif', frames)

#the function takes a target image, number of generations and a population's size as initial parameters
genetic_painter(to_paint, 600, 100)