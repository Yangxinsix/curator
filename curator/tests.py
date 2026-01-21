import unittest

from tests import test_feature_stack


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromModule(test_feature_stack)
