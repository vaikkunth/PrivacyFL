from pyspark import SparkContext

sc = SparkContext("local", "First App")


class Goo:
    def __init__(self, x):
        self.x = x
        self.new_val = 0

    def funcA(self):
        self.new_val = 1
        return self.x

goo1 = Goo(1)

def mapper(goo_num):
    if goo_num == 1:
        print('ID of goo1 in method call is {} \n'.format(id(goo1))) # not the same!!
        goo1.funcA()
        print('Goo1 new val is {}'.format(goo1.new_val))
        return None

print('ID of goo1 before Spark call is {} \n'.format(id(goo1)))
WORKS = sc.parallelize([1]).map(mapper).collect()
print('ID of goo1 after Spark call is {} \n'.format(id(goo1)))
print('Goo1 new val is {} \n'.format(goo1.new_val))
