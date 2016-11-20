"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qctoolkit.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse.
    - AtomicPulseTemplate: PulseTemplate that does imply any control flow disruptions and can be
        directly translated into a waveform.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Set, Optional

from qctoolkit.serialization import Serializable

from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.sequencing import SequencingElement, InstructionBlock

__all__ = ["MeasurementWindow", "PulseTemplate", "AtomicPulseTemplate", "DoubleParameterNameException",  "concatenate_name_mappings"]


MeasurementWindow = Tuple[str, float, float]
def concatenate_name_mappings(d1: Dict[str,str],d2 : Dict[str, str],*args) -> Dict[str,str]:
    """
    Concatenate the mappings of the given parameters. Mappings further to the right are applied later
    and overwrite earlier ones.
    :param d1:
    :param d2:
    :param args:
    :return:
    """
    result = d1.copy()
    for k, v in d2.items():
        if v in d1:
            result[k] = result[v]
        else:
            result[k] = v
    if args:
        return concatenate_name_mappings(result, args[0])
    else:
        return result


class PulseTemplate(Serializable, SequencingElement, metaclass=ABCMeta):
    """A PulseTemplate represents the parametrized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate and achieved by invoking the sequencing process.
    """

    def __init__(self, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)

    @abstractproperty
    def parameter_names(self) -> Set[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""

    @abstractproperty
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """The set of ParameterDeclaration objects detailing all parameters required to instantiate
        this PulseTemplate.
        """

    @abstractproperty
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """

    @abstractproperty
    def num_channels(self) -> int:
        """Returns the number of hardware output channels this PulseTemplate defines."""


    def __matmul__(self, other) -> 'SequencePulseTemplate':
        """This method enables us to use the @-operator (intended for matrix multiplication) for
         concatenating pulses"""
        from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
        # check if parameter names of the subpulses clash, otherwise construct a default mapping
        double_parameters = self.parameter_names & other.parameter_names # intersection
        if double_parameters:
            # if there are parameter name conflicts, throw an exception
            raise DoubleParameterNameException(self, other, double_parameters)
        else:
            subtemplates = [(self, {p:p for p in self.parameter_names}, {}),
                            (other, {p:p for p in other.parameter_names}, {})]
            external_parameters = self.parameter_names | other.parameter_names # union
            return SequencePulseTemplate(subtemplates, external_parameters)


class AtomicPulseTemplate(PulseTemplate):
    """A PulseTemplate that does not imply any control flow disruptions and can be directly
    translated into a waveform.

    Implies that no AtomicPulseTemplate object is interruptable.
    """

    def __init__(self, identifier: Optional[str]=None):
        super().__init__(identifier=identifier)

    def is_interruptable(self) -> bool:
        return False

    @abstractmethod
    def build_waveform(self, parameters: Dict[str, Parameter]) -> Optional['Waveform']:
        """Translate this AtomicPulseTemplate into a waveform according to the given parameteres.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
            Waveform object represented by this AtomicPulseTemplate object or None, if this object
                does not represent a valid waveform.
        """

    @abstractmethod
    def get_measurement_windows(self, parameters: Dict[str, Parameter]=None) -> List[MeasurementWindow]:
        """
        :param parameters:
        :return:
        """

    def build_sequence(self,
                       sequencer: 'Sequencer',
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       window_mapping: Dict[str, str],
                       instruction_block: InstructionBlock) -> None:
        waveform = self.build_waveform(parameters)
        if waveform:
            windows = self.get_measurement_windows(parameters)

            for entry in windows:
                if entry[0] in window_mapping:
                    entry[0] = window_mapping[entry[0]]

            instruction_block.add_instruction_exec(waveform,windows)




class DoubleParameterNameException(Exception):

    def __init__(self, templateA: PulseTemplate, templateB: PulseTemplate, names: Set[str]) -> None:
        super().__init__()
        self.templateA = templateA
        self.templateB = templateB
        self.names = names

    def __str__(self) -> str:
        return "Cannot concatenate pulses '{}' and '{}' with a default parameter mapping. " \
               "Both define the following parameter names: {}".format(
            self.templateA, self.templateB, ', '.join(self.names)
        )

