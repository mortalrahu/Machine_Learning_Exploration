"""
This script contains a genetic algorithm to find the optimal number of moves to solve the game for a given board
"""

from tile_game import Game, get_positive_integer
import random
import copy
import time


class GeneticAlgorithmPlayer:
    """
    A class that implements a genetic algorithm to find the optimal number of moves to solve the tile game.

    Attributes:
        game (Game): The game instance.
        population_size (int): The number of individuals in the population.
        generations (int): The number of generations to evolve.
        mutation_rate (float): The probability of mutation.
        chromosome_length (int): The length of each chromosome (sequence of moves).

    Methods:
        initialize_population(): Initializes the population with random chromosomes.
        fitness(chromosome): Evaluates the fitness of a chromosome.
        select_parents(): Selects parents for reproduction based on fitness.
        crossover(parent1, parent2): Produces offspring by crossing over two parents.
        mutate(chromosome): Mutates a chromosome with a certain probability.
        run(): Runs the genetic algorithm to find the optimal sequence of moves.
    """

    def __init__(self, game, population_size=20, generations=50, mutation_rate=0.01, chromosome_length=10):
        """
        Initializes the genetic algorithm player with given parameters.

        Args:
            game (Game): The game instance.
            population_size (int): The number of individuals in the population.
            generations (int): The number of generations to evolve.
            mutation_rate (float): The probability of mutation.
            chromosome_length (int): The length of each chromosome (sequence of moves).
        """
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.chromosome_length = chromosome_length
        self.colors = list(range(1, self.game.m + 1))

    def initialize_population(self):
        """
        Initializes the population with random chromosomes.

        Returns:
            list: A list of randomly initialized chromosomes.
        """
        population = []
        for _ in range(self.population_size):
            chromosome = [random.choice(self.colors) for _ in range(self.chromosome_length)]
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        """
        Evaluates the fitness of a chromosome by simulating the game with the given sequence of moves.

        Args:
            chromosome (list): The chromosome to evaluate.

        Returns:
            int: The fitness score (number of moves required to complete the game).
        """
        game_copy = copy.deepcopy(self.game)
        for move_count, color in enumerate(chromosome, start=1):
            game_copy.perform_move(color)
            if game_copy.is_game_over(print_off=True):
                return len(chromosome) - move_count  # Fewer moves = higher fitness
        return 0  # Did not complete the game

    def select_parents(self, population):
        """
        Selects parents for reproduction based on fitness.

        Args:
            population (list): The population of chromosomes.

        Returns:
            tuple: Two selected parents.
        """
        weights = [self.fitness(chromosome) for chromosome in population]
        parent1 = random.choices(population, weights=weights, k=1)[0]
        parent2 = random.choices(population, weights=weights, k=1)[0]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        """
        Produces offspring by crossing over two parents.

        Args:
            parent1 (list): The first parent chromosome.
            parent2 (list): The second parent chromosome.

        Returns:
            list: The offspring chromosome.
        """
        crossover_point = random.randint(1, self.chromosome_length - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, chromosome):
        """
        Mutates a chromosome with a certain probability.

        Args:
            chromosome (list): The chromosome to mutate.
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.choice(self.colors)

    def run(self):
        """
        Runs the genetic algorithm to find the optimal sequence of moves.

        Returns:
            list: The best chromosome found.
        """
        population = self.initialize_population()
        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population)
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent2, parent1)
                self.mutate(offspring1)
                self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])
            population = new_population
        best_chromosome = max(population, key=self.fitness)
        return best_chromosome


if __name__ == '__main__':

    start_time = time.time()

    n = get_positive_integer("Enter board size n: ")
    m = get_positive_integer("Enter number of colors m: ")
    print("Default Population Size is 20")
    population_size = get_positive_integer("Enter number the population size: ")
    print("Default generations is 50")
    generations = get_positive_integer("Enter number the generations: ")
    print("Default mutation rate is 0.01")
    mutation_rate = float(input("Enter number the mutation rate: "))
    print("Default chromosome length is 10")
    chromosome_length = get_positive_integer("Enter length of the chromosome: ")

    game = Game(n, m)
    ga_player = GeneticAlgorithmPlayer(game, population_size=population_size, generations=generations, mutation_rate=mutation_rate, chromosome_length=chromosome_length)
    best_moves = ga_player.run()
    print("Best sequence of moves:", best_moves)
    for move in best_moves:
        game.perform_move(move)
        game.print_board()
        if game.is_game_over(print_off=True):
            break

    end_time = time.time()

    gen_algo_time = end_time - start_time
    print(f"Game time: {gen_algo_time} seconds")
