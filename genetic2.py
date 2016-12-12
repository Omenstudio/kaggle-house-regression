import copy
import operator
import random
from collections import defaultdict

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

from data_prepocessor import DataPreprocessor
from genetic import GeneticAlgorithm, GeneticFunctions


class GuessText(GeneticFunctions):
    def __init__(self, target_text, limit=200, size=400, prob_crossover=0.9, prob_mutation=0.3):
        self.target = 0
        self.counter = 0

        self.limit = limit
        self.size = size
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation

        self.all = [x for x in range(335)]

        self.dp = DataPreprocessor()
        self.dp.read_all()

        self.ws = defaultdict(int)
        self.best = (0, [])

        print("Ready to evaluate")
        pass

    # GeneticFunctions interface impls
    def probability_crossover(self):
        return self.prob_crossover

    def probability_mutation(self):
        return self.prob_mutation

    def initial(self):
        """
        Начальные хромосомы (выборки параметров)
        :return:
        """
        ans = []
        for i in range(100):
            n = random.randint(0, 334)
            d = []
            for j in range(n):
                d.append(random.randint(0, 334))
            d = list(set(d))
            ans.append(d)
        return ans

    def fitness(self, chromo):
        """
        Оценочная функция!
        :param chromo:
        :return:
        """
        model = GradientBoostingRegressor()
        outputs = self.dp.train_outputs
        inputs_temp = np.array(list(zip(*self.dp.train_inputs)))
        inputs = []
        for i in chromo:
            inputs.append(inputs_temp[i])
        inputs = np.array(inputs)
        inputs = np.array(list(zip(*inputs)))

        if len(inputs) == 0:
            return 0

        res = cross_val_score(model, inputs, outputs, cv=KFold(n_splits=10))
        ans = res.mean()*1000
        if ans < -1000:
            ans = -1000
        if ans > self.best[0]:
            self.best = (ans, copy.copy(chromo))
        return ans

    def check_stop(self, fits_populations):
        """
        Функция останоки
        :param fits_populations:
        :return:
        """
        self.counter += 1
        if True or self.counter % 1 == 0:
            fits = [f for f, ch in fits_populations]
            best = max(fits)
            worst = min(fits)
            avg = sum(fits) / len(fits)
            print("[G %3d] score=(%4d, %4d, %4d)" % (self.counter, best, avg, worst))
            # self.print_ws()
        if self.counter % 10 == 0:
            print(self.best)
        return self.counter >= self.limit

    def crossover(self, parents):
        """
        Размножение
        :param parents:
        :return:
        """
        father, mother = parents

        self.make_weights(father)
        self.make_weights(mother)


        random.shuffle(father)
        random.shuffle(mother)
        slicer_father = random.randint(1, len(father)-1)
        slicer_mother = random.randint(1, len(mother)-1)
        child1 = father[:slicer_father] + mother[slicer_mother:]
        child2 = father[slicer_father:] + mother[:slicer_mother]

        child1 = list(set(child1))
        child2 = list(set(child2))

        # print(mother, father)
        # print(child1, child2)
        # exit()

        return child1, child2

    def mutation(self, chromosome):
        mutated = copy.deepcopy(chromosome)
        random.shuffle(mutated)
        mutated = sorted(mutated, key=lambda m: self.ws[m])
        mutated.reverse()

        slicer = random.randint(int(len(mutated)/2), len(mutated)-1)
        slicer2 = len(mutated) - slicer + random.randint(5, 100)

        mutated = mutated[:slicer]
        random.shuffle(self.all)

        sorted_x = sorted(self.ws.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        mutated_additional = [k for k, v in sorted_x]
        mutated_additional = mutated_additional[:slicer2*2]
        random.shuffle(mutated_additional)
        mutated += mutated_additional[:slicer2]

        mutated = list(set(mutated))
        return mutated

    # internals
    def tournament(self, fits_populations):
        """
        Выбирает лучших родителей для размножения
        :param fits_populations:
        :return:
        """

        target_len = (len(fits_populations) + 1) / 3
        index = random.randint(0, int(target_len))
        d = sorted(fits_populations)
        d.reverse()
        return d[index][1]

    def make_weights(self, mas):
        for x in mas:
            self.ws[x] += 1
        pass

    def print_ws(self):
        sorted_x = sorted(self.ws.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        print(sorted_x)


    def parents(self, fits_populations):
        """
        Генератор "родителей" для размножения
        :param fits_populations:
        :return:
        """
        while True:
            father = self.tournament(fits_populations)
            mother = self.tournament(fits_populations)
            yield (father, mother)
        pass



    pass


def main():
    print('Started')
    GeneticAlgorithm(GuessText("Hello World!")).run()


main()
