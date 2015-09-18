import unittest
import os
import sys
from src.pulses.Plotting import plot, PlottingNotPossibleException
from src.pulses.TablePulseTemplate import TablePulseTemplate
from src.pulses.SequencePulseTemplate import SequencePulseTemplate

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

class GenericPlottingTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup test data
        self.square = TablePulseTemplate()
        self.square.add_entry('up', 'v', 'hold')
        self.square.add_entry('down', 0, 'hold')
        self.square.add_entry('length', 0)

        self.mapping1 = {}
        self.mapping1['up'] = lambda ps: ps['uptime']
        self.mapping1['down'] = lambda ps: ps['uptime'] + ps['length']
        self.mapping1['v'] = lambda ps: ps['voltage']
        self.mapping1['length'] = lambda ps: ps['pulse-length'] * 0.5

        self.outer_parameters = ['uptime', 'length', 'pulse-length', 'voltage']

        self.parameters = {}
        self.parameters['uptime'] = 5
        self.parameters['length'] = 10
        self.parameters['pulse-length'] = 100
        self.parameters['voltage'] = 10

        self.sequence = SequencePulseTemplate([(self.square, self.mapping1)], self.outer_parameters)

    def test_plotting(self):
        pass
        #plot(self.sequence,self.parameters)
        
    def test_exceptions(self):
        
        a = PlottingNotPossibleException(self.square)
        self.assertIsInstance(a.__str__(), str)