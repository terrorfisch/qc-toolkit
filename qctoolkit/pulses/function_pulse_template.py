import logging
from typing import Dict, List, Set,  Optional
import numbers
import numpy as np
from typing import Any

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

from .parameters import ParameterDeclaration, Parameter
from .pulse_template import PulseTemplate, MeasurementWindow
from .sequencing import InstructionBlock, Sequencer
from .sequence_pulse_template import ParameterNotProvidedException
from .instructions import Waveform

logger = logging.getLogger(__name__)

__all__ = ["FunctionPulseTemplate"]

class FunctionPulseTemplate(PulseTemplate):
    """Defines a pulse via a time-domain expression.

    FunctionPulseTemplate stores the expression and its external parameters. The user must provide
    two things: one expression that calculates the length of the pulse from the external parameters
    and the time-domain pulse shape itself as a expression. The external parameters are derived from
    the expressions themselves.
    Like other PulseTemplates the FunctionPulseTemplate can be declared to be a measurement pulse.

    The independent variable in the expression is called 't' and is given in units of nano-seconds.
    """

    def __init__(self, expression: str, duration_expression: str, measurement: bool=False) -> None:
        super().__init__()
        self.__expression = Expression(expression)
        self.__duration_expression = Expression(duration_expression)
        self.__is_measurement_pulse = measurement # type: bool
        self.__parameter_names = set(self.__duration_expression.variables() + self.__expression.variables()) - set(['t'])

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return a set of all parameter declaration objects of this TablePulseTemplate."""
        return set()

    def get_pulse_length(self, parameters):
        """Return the length of this pulse for the given parameters."""
        missing = self.__parameter_names - set(parameters.keys())
        for m in missing:
            raise ParameterNotProvidedException(m)
        return self.__duration_expression.evaluate(parameters)

    def get_measurement_windows(self, parameters: Optional[Dict[str, Parameter]] = {}) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate.
       
        A ExpressionPulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether the measurement_pulse flag was given during construction.
        """
        if not self.__is_measurement_pulse:
            return
        else:
            return [(0, self.get_pulse_length(parameters))]

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       instruction_block: InstructionBlock) -> None:
        instantiated = FunctionWaveform(parameters,self.__expression,self.__duration_expression)
        waveform = sequencer.register_waveform(instantiated)
        instruction_block.add_instruction_exec(waveform)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, 'Condition']) -> bool:
        return any(parameters[name].requires_stop for name in parameters.keys() if (name in self.parameter_names) and not isinstance(parameters[name], numbers.Number))

    def get_serialization_data(self, serializer: Serializer) -> None:
        root = dict()
        root['type'] = 'FunctionPulseTemplate'
        root['parameter_names'] = self.__parameter_names
        root['duration_expression'] = self.__duration_expression.string
        root['expression'] = self.__expression.string
        root['measurement'] = self.__is_measurement_pulse
        return root

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs):
        return FunctionPulseTemplate(kwargs['expression'], kwargs['duration_expression'], kwargs['Measurement'])

class FunctionWaveform(Waveform):

    def __init__(self, parameters:Dict[str,Parameter],expression,duration_expression) -> None:
        super().__init__()
        self.__expression = expression
        self.__duration_expression = duration_expression
        self.__parameters = parameters
        
    @property
    def _compare_key(self) -> Any:
        return self.__expression

    @property
    def duration(self) -> float:
        return self.__duration_expression.evaluate(self.__parameters)

    def sample(self, sample_times: np.ndarray, first_offset: float=0) -> np.ndarray:
        # TODO: implement this
        
        voltages = np.empty_like(sample_times)
        sample_times -= (sample_times[0] - first_offset)
        
        #voltages[sample_times] = self.__expression(self.__parameters.update("t":sample_times))
        
        #for entry1, entry2 in zip(self.__table[:-1], self.__table[1:]): # iterate over interpolated areas
        #    indices = np.logical_and(sample_times >= entry1.t, sample_times <= entry2.t)
        #    voltages[indices] = entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices]) # evaluate interpolation at each time
        return voltages