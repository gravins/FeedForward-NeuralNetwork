import numpy as np
from preprocessing.enums import TaskType


class Task:

    # static ID used for task name which increments for every new task
    idCounter = 0

    def __init__(self, inputs, targets, folds, name=None):
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)
        self.task_type = TaskType.regression
        self.folds = folds
        self.name = "Task" + str(Task.idCounter) if name is None else name
        Task.idCounter += 1 
