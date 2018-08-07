"""Implementation of HRED model https://arxiv.org/pdf/1605.06069.pdf"""
import arcadian

class HRED(arcadian.gm.GenericModel):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build(self):
        pass