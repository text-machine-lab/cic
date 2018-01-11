import unittest2

from cic.datasets.mnist import MNISTTestSet


class MNISTDatasetsTest(unittest2.TestCase):
    def test_mnist_split(self):
        # Test that split dataset elements point back to the original dataset
        mnist_test_set = MNISTTestSet()
        mnist_1, mnist_2 = mnist_test_set.split(fraction=0.7)
        for index in range(len(mnist_1)):
            each_split_example = mnist_1[index]
            assert isinstance(each_split_example, dict)
            each_pointer_index = mnist_1.indices[index]
            each_test_example = mnist_test_set[each_pointer_index]
            for key in each_split_example:
                assert (each_test_example[key] == each_split_example[key]).all()