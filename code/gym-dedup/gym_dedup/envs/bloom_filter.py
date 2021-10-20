import math
#pip install the following:
import mmh3
from bitarray import bitarray


class bloom_filter:

    def __init__(self, num_elements, fp_prob):
        """
        n: number of elements to be inserted into Bloom filter
        fp_prob: desired false positive probability
        """
        self.n = num_elements
        self.fp_prob = fp_prob

        # Get length of bloom filter m
        self.m = int(-(self.n * math.log(self.fp_prob))/(math.log(2)**2))
        # Keep track of number of unset bits  to determine if all bits in the filter are set
        self.num_zero = self.m

        # Get optimal number of hash functions
        self.hash_count = int((self.m/self.n) * math.log(2))
        # Initialize bit array and make all bits 0
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)

    """ Add an item to the filter"""
    def add(self, item):
        for i in range(self.hash_count):
            # get hash function results
            res = mmh3.hash(str(item), i)
            # take down 1 for number of unset bits if it's newly set
            if self.bit_array[res] == False:
                self.num_zero = self.num_zero - 1
            # set corresponding bits True
            self.bit_array[res] = True

    """ Check if an items exists in filter"""
    def does_exist(self, item):
        for i in range(self.hash_count):
            # get hash function results
            res = mmh3.hash(str(item), i)
            # check if any of resutling bits is False. If so, item is not present in filter
            # Else, the item may be present in filter
            if self.bit_array[res] == False:
                return False
            return True

    """ Check if all bits are set in filter"""
    def is_full(self):
        return self.num_zero == 0

    """ Reset the filter"""
    def reset(self):
        self.bit_array.setall(0)
        self.num_zero = self.m