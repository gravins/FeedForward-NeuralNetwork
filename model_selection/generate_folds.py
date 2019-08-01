import random


class KFolds:
    """
    This class represents a list of folds, each of that consist of a different set
    of training index and test/validation index
    """
    def __init__(self, folds):
        self.folds = folds

    class Folds:
        """
        This class is an instance of a fold. The fold contain the index for the
        training and for the test/validation
        """
        def __init__(self, training_indexes, test_indexes):
            self.train_indexes = training_indexes
            self.test_indexes = test_indexes

    @staticmethod
    def cross_validation_folds(data_indexes, folds_number, shuffle=True, random_seed=42):
        """
        This function implements the splitting among the indexes in order to perform
        a k-fold cross validation
        """
        if shuffle:
            random.seed(random_seed)
            random.shuffle(data_indexes)

        folds_dim = len(data_indexes) / folds_number
        folds = [data_indexes[int(folds_dim * i):int(folds_dim * (i + 1))] for i in range(folds_number)]

        cross_validation = []
        for i in range(folds_number):
            training_index = []
            for j, f in enumerate(folds):
                if j != i:
                    training_index += f

            cross_validation.append(
                KFolds.Folds(training_indexes=training_index, test_indexes=folds[i]))

        return KFolds(cross_validation)

    @staticmethod
    def double_cross_validation_folds(data_indexes, external_folds_dim=1, internal_folds_dim=1, shuffle=True, random_seed=42):
        """
        This function implements the splitting among the indexes in order to perform
        a double cross validation
        """
        if shuffle:
            random.seed(random_seed)
            random.shuffle(data_indexes)

        double_cross_validation = KFolds.cross_validation_folds(data_indexes, external_folds_dim)
        for e in double_cross_validation.folds:
            e.train_indexes = KFolds.cross_validation_folds(e.train_indexes, internal_folds_dim)

        return double_cross_validation
