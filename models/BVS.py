import os
from.area import Area

"""
General Idea of the core idea to create a BVS architecture

"""


class BVS:

    # Class variables
    max_line_display = 150

    def __init__(self, v1=None, v2=None, v4=None, it=None):
        self.Areas = []

        if v1 is not None:
            self.v1 = Area(v1)
            self.Areas.append(self.v1)
        else:
            self.v1 = v1

        if v2 is not None:
            self.v2 = Area(v2)
        else:
            self.v2 = None

        if v4 is not None:
            self.v4 = Area(v4)
        else:
            self.v4 = None

        if it is not None:
            self.it = Area(it)
        else:
            self.it = None

    def fit(self):
        print("fit model!")

    def summary(self):
        columns, rows = os.get_terminal_size(0)

        # print v1 summary
        if self.v1 is not None:
            print("V1: ", self.v1.type)
            self.v1.summary()
        else:
            print("V1: ", self.v1)
        print("-" * min(columns, self.max_line_display))

        # print v2 summary
        if self.v2 is not None:
            print("V2")
            self.v2.summary()
        else:
            print("V2: ", self.v2)
        print("-" * min(columns, self.max_line_display))

        # print v4 summary
        if self.v4 is not None:
            print("V4")
            self.v4.summary()
        else:
            print("V4: ", self.v4)
        print("-" * min(columns, self.max_line_display))

        # print IT summary
        if self.it is not None:
            print("IT")
            self.it.summary()
        else:
            print("IT: ", self.it)
        print("-" * min(columns, self.max_line_display))

