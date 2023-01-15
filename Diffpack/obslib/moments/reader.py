#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
from tools.reader import _READER
from tools.config import conf


class READER(_READER):

    def __init__(self):
        pass

    def modify_table(self, tab):
        pass
        # tab=self.apply_cuts(tab,k)
        return tab


