import unittest
from secdtplus import entity2_secdt

class TestMO(unittest.TestCase):
    def test_init(self):
        mo = entity2_secdt.ModelOwner()
        self.assertEqual()