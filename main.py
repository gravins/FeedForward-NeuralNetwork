from test.test import Test1, Test2, Test3, Test4, Test5, StepByStepExample, LevelSetTest, TestErrorFunctions
from test.monk_test import MonkTest, MonkTest2, MonkTest3
from test.MLcup_test import MicheliDataset, MicheliModelSelection
from time import time


if __name__ == '__main__':
    start = time()

    #Test1.run()
    #Test2.run()
    #Test3.run()
    #Test4.run()
    #Test5.run()
    #StepByStepExample.run()
    #LevelSetTest.run()
    #TestErrorFunctions.run()
    #MonkTest.run()
    #MonkTest2.run()
    #MonkTest3.run()
    MicheliDataset.run()
    #MicheliDataset.run_nested()
    #MicheliModelSelection.run()

    m, s = divmod(time() - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02f end" % (h, m, s))
