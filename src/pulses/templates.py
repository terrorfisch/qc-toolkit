from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Tuple, Union, Dict, List, Set,  Optional, NamedTuple, Callable, Any, Iterable
import numbers
import logging
import copy

"""RELATED THIRD PARTY IMPORTS"""
import numpy as np

"""LOCAL IMPORTS"""
from .Parameter import ParameterDeclaration, Parameter, ParameterNotProvidedException
from .Sequencer import SequencingElement, Sequencer
from .Expressions import Expression
from .Serializer import Serializer, Serializable
from .Instructions import InstructionBlock, Waveform, WaveformTable
from .Interpolation import InterpolationStrategy, LinearInterpolationStrategy, HoldInterpolationStrategy, JumpInterpolationStrategy


"""EXPORTS"""
__all__ = ["PulseTemplate",
           "BranchPulseTemplate",
           "FunctionPulseTemplate",
           "LoopPulseTemplate",
           "RepetitionPulseTemplate",
           "SequencePulseTemplate",
           "TablePulseTemplate",
           
           "ParameterNotInPulseTemplateException",
           "ParameterNotIntegerException",
           "MissingParameterDeclarationException",
           "MissingMappingException",
           "UnnecessaryMappingException",
           "RuntimeMappingError",
           "ParameterValueIllegalException",
           
           "MeasurementWindow",
           "TableValue",
           "TableEntry",
           "clean_entries",
           "TableWaveform"]


logger = logging.getLogger(__name__)

MeasurementWindow = Tuple[float, float]

TableValue = Union[float, ParameterDeclaration]
TableEntry = NamedTuple("TableEntry", [('t', TableValue), ('v', TableValue), ('interp', InterpolationStrategy)])



""" Templates """
class PulseTemplate(Serializable, SequencingElement, metaclass = ABCMeta):
    """A PulseTemplate represents the parameterized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate.
    """

    def __init__(self, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)

    @abstractproperty
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""

    @abstractproperty
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return the set of ParameterDeclarations."""

    @abstractmethod
    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""

    @abstractproperty
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""

# a subtemplate consists of a pulse template and mapping functions for its "internal" parameters
Subtemplate = Tuple[PulseTemplate, Dict[str, str]]

class BranchPulseTemplate(PulseTemplate):
    """Conditional branching in a pulse.
    
    A BranchPulseTemplate is a PulseTemplate
    with different structures depending on a certain condition.
    It defines refers to an if-branch and an else-branch, which
    are both PulseTemplates.
    When instantiating a pulse from a BranchPulseTemplate,
    both branches refer to concrete pulses. If the given
    condition evaluates to true at the time the pulse is executed,
    the if-branch, otherwise the else-branch, is chosen for execution.
    This allows for alternative execution 'paths' in pulses.
    
    Both branches must be of the same length.
    """
    def __init__(self) -> None:
        super().__init__()
        self.else_branch = None
        self.if_branch = None
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()
    
    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        raise NotImplementedError()

    @property
    def parameter_declarations(self) -> Set[str]:
        """Return the set of ParameterDeclarations."""
        raise NotImplementedError()

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        """Return True, if this PulseTemplate contains points at which it can halt if interrupted."""
        raise NotImplementedError()

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
        
    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        # TODO: implement this
        dt = 1
        instantiated = self.__expression(dt, **parameters)
        instantiated = self.get_entries_instantiated(parameters)
        instantiated = tuple(instantiated)
        waveform = sequencer.register_waveform(instantiated)
        instruction_block.add_instruction_exec(waveform)

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool: 
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

class LoopPulseTemplate(PulseTemplate):
    """Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which is repeated
    during execution as long as a certain condition holds.
    """
    
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        raise NotImplementedError()

    @property
    def parameter_declarations(self) -> Set[str]:
        """Return the set of ParameterDeclarations."""
        raise NotImplementedError()

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        raise NotImplementedError()


class RepetitionPulseTemplate(PulseTemplate):

    def __init__(self, body: PulseTemplate, repetition_count: Union[int, ParameterDeclaration], identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.__body = body
        self.__repetition_count = repetition_count

    @property
    def body(self) -> PulseTemplate:
        return self.__body

    @property
    def repetition_count(self) -> Union[int, ParameterDeclaration]:
        return self.__repetition_count

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>".format(self.__repetition_count, self.__body)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return self.__body.is_interruptable

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock):
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            if not repetition_count.check_parameter_set_valid(parameters):
                raise ParameterValueIllegalException(repetition_count, parameters[repetition_count.name])
            repetition_count = repetition_count.get_value(parameters)
            if not repetition_count.is_integer():
                raise ParameterNotIntegerException(self.__repetition_count.name, repetition_count)

        for i in range(0, int(repetition_count), 1):
            sequencer.push(self.__body, parameters, instruction_block)

    def requires_stop(self, parameters: Dict[str, Parameter]):
        if isinstance(self.__repetition_count, ParameterDeclaration):
            if parameters[self.__repetition_count.name].requires_stop:
                return True
        return False

    def get_serialization_data(self, serializer: Serializer):
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = serializer._serialize_subpulse(repetition_count)
        return dict(
            type=serializer.get_type_identifier(self),
            body=serializer._serialize_subpulse(self.__body),
            repetition_count=repetition_count
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    repetition_count: Dict[str, Any],
                    body: Dict[str, Any],
                    identifier: Optional[str]=None):
        body = serializer.deserialize(body)
        if isinstance(repetition_count, dict):
            repetition_count = serializer.deserialize(repetition_count)
        return RepetitionPulseTemplate(body, repetition_count, identifier=identifier)

class SequencePulseTemplate(PulseTemplate):
    """A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group smaller
    PulseTemplates (subtemplates) into on larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate allows to specify a mapping of
    parameter declarations from its subtemplates, enabling
    renaming and mathematical transformation of parameters.
    The default behavior is to exhibit the union of parameter
    declarations of all subtemplates. If two subpulses declare
    a parameter with the same name, it is mapped to both. If the
    declarations define different minimal and maximal values, the
    more restricitive is chosen, if possible. Otherwise, an error
    is thrown and an explicit mapping must be specified.
    ^outdated
    """

    def __init__(self, subtemplates: List[Subtemplate], external_parameters: List[str], identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.__parameter_names = frozenset(external_parameters)
        # convert all mapping strings to expressions
        for i, (template, mappings) in enumerate(subtemplates):
            subtemplates[i] = (template, {k: Expression(v) for k, v in mappings.items()})

        for template, mapping_functions in subtemplates:
            # Consistency checks
            open_parameters = template.parameter_names
            mapped_parameters = set(mapping_functions.keys())
            missing_parameters = open_parameters - mapped_parameters
            for m in missing_parameters:
                raise MissingMappingException(template, m)
            unnecessary_parameters = mapped_parameters - open_parameters
            for m in unnecessary_parameters:
                raise UnnecessaryMappingException(template, m)

            for key, mapping_function in mapping_functions.items():
                mapping_function = mapping_functions[key]
                required_externals = set(mapping_function.variables())
                non_declared_externals = required_externals - self.__parameter_names
                if non_declared_externals:
                    raise MissingParameterDeclarationException(template, non_declared_externals.pop())

        self.subtemplates = subtemplates  # type: List[Subtemplate]
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return set([ParameterDeclaration(name) for name in self.__parameter_names])

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        raise NotImplemented() # will be computed by Sequencer

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool):
        self.__is_interruptable = new_value

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool:
        if not self.subtemplates:
            return False

        # obtain first subtemplate
        (template, mapping_functions) = self.subtemplates[0]

        # collect all parameters required to compute the mappings for the first subtemplate
        external_parameters = set()
        for mapping_function in mapping_functions.values():
            external_parameters = external_parameters | set([parameters[x] for x in mapping_function.variables()])

        # return True only if none of these requires a stop
        return any([p.requires_stop for p in external_parameters])

    def __map_parameter(self, mapping_function: str, parameters: Dict[str, Parameter]) -> Parameter:
        external_parameters = mapping_function.variables()
        external_values = {name: float(parameters[name]) for name in external_parameters}
        return mapping_function.evaluate(external_values)

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        # detect missing or unnecessary parameters
        missing = self.parameter_names - set(parameters)
        for m in missing:
            raise ParameterNotProvidedException(m)

        # push subtemplates to sequencing stack with mapped parameters
        for template, mappings in reversed(self.subtemplates):
            inner_parameters = {name: self.__map_parameter(mapping_function, parameters) for (name, mapping_function) in mappings.items()}
            sequencer.push(template, inner_parameters, instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['external_parameters'] = sorted(list(self.parameter_names))
        data['is_interruptable'] = self.is_interruptable

        subtemplates = []
        for (subtemplate, mapping_functions) in self.subtemplates:
            mapping_functions_strings = {k: m.string for k, m in mapping_functions.items()}
            subtemplate = serializer._serialize_subpulse(subtemplate)
            subtemplates.append(dict(template=subtemplate, mappings=copy.deepcopy(mapping_functions_strings)))
        data['subtemplates'] = subtemplates

        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    is_interruptable: bool,
                    subtemplates: Iterable[Dict[str, Union[str, Dict[str, Any]]]],
                    external_parameters: Iterable[str],
                    identifier: Optional[str]=None) -> 'SequencePulseTemplate':
        subtemplates = \
            [(serializer.deserialize(d['template']),
             {k: m for k, m in d['mappings'].items()}) for d in subtemplates]

        template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        template.is_interruptable = is_interruptable
        return template

class TableWaveform(Waveform):

    def __init__(self, waveform_table: WaveformTable) -> None:
        if len(waveform_table) < 2:
            raise ValueError("The given WaveformTable has less than two entries.")
        super().__init__()
        self.__table = waveform_table

    @property
    def _compare_key(self) -> Any:
        return self.__table

    @property
    def duration(self) -> float:
        return self.__table[-1].t

    def sample(self, sample_times: np.ndarray, first_offset: float=0) -> np.ndarray:
        sample_times -= (sample_times[0] - first_offset)
        voltages = np.empty_like(sample_times)
        for entry1, entry2 in zip(self.__table[:-1], self.__table[1:]): # iterate over interpolated areas
            indices = np.logical_and(sample_times >= entry1.t, sample_times <= entry2.t)
            voltages[indices] = entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices]) # evaluate interpolation at each time
        return voltages


class TablePulseTemplate(PulseTemplate):
    """Defines a pulse via linear interpolation of a sequence of (time,voltage)-pairs.
    
    TablePulseTemplate stores a list of (time,voltage)-pairs (the table) which is sorted
    by time and uniquely define a pulse structure via interpolation of voltages of subsequent
    table entries.
    TablePulseTemplate provides methods to declare parameters which may be referred to instead of
    using concrete values for both, time and voltage. If the time of a table entry is a parameter
    reference, it is sorted into the table according to the first value of default, minimum or maximum
    which is defined (not None) in the corresponding ParameterDeclaration. If none of these are defined,
    the entry is placed at the end of the table.
    A TablePulseTemplate may be flagged as representing a measurement pulse, meaning that it defines a
    measurement window.
    """

    def __init__(self, measurement=False, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.__identifier = identifier
        self.__entries = [] # type: List[TableEntry]
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = measurement# type: bool
        self.__interpolation_strategies = {'linear': LinearInterpolationStrategy(),
                                           'hold': HoldInterpolationStrategy(), 
                                           'jump': JumpInterpolationStrategy()
                                          }

    @staticmethod
    def from_array(times: np.ndarray, voltages: np.ndarray, measurement=False):
        """Static constructor to build a TablePulse from numpy arrays.

        Args:
            times: 1D numpy array with time values
            voltages: 1D numpy array with voltage values

        Returns:
            TablePulseTemplate with the given values, hold interpolation everywhere and no free parameters.
        """
        res = TablePulseTemplate(measurement=measurement)
        for t, v in zip(times, voltages):
            res.add_entry(t, v, interpolation='hold')
        return res

    def add_entry(self,
                  time: Union[float, str, ParameterDeclaration], 
                  voltage: Union[float, str, ParameterDeclaration], 
                  interpolation: str = 'hold') -> None:
        """Add an entry to this TablePulseTemplate.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name or a ParameterDeclaration object.
        The following constraints hold:
        - If a non-existing parameter declaration is referenced (via string), it is created without min, max and default values.
        - Parameter declarations for the time domain may not be used multiple times. Else a ValueError is thrown.
        - ParameterDeclaration objects for the time domain may not refer to other ParameterDeclaration objects as min or max values. Else a ValueError is thrown.
        - If a ParameterDeclaration is provided, its min and max values will be set to its neighboring values if they were not set previously or would exceed neighboring bounds.
        - Each entries time value must be greater than its predecessor's, i.e.,
            - if the time value is a float and the previous time value is a float, the new value must be greater
            - if the time value is a float and the previous time value is a parameter declaration, the new value must not be smaller than the maximum of the parameter declaration
            - if the time value is a parameter declaration and the previous time value is a float, the new values minimum must be no smaller
            - if the time value is a parameter declaration and the previous time value is a parameter declaration, the new minimum must not be smaller than the previous minimum
              and the previous maximum must not be greater than the new maximum
        """

        # Check if interpolation value is valid
        if interpolation not in self.__interpolation_strategies.keys():
            raise ValueError("Interpolation strategy not implemented. Allowed values: {}.".format(', '.join(self.__interpolation_strategies.keys())))
        else:
            interpolation = self.__interpolation_strategies[interpolation]

        # If this is the first entry, there are a number of cases we have to check
        if not self.__entries:
            # if the first entry has a time that is either > 0 or a parameter declaration, insert a start point (0, 0)
            if not isinstance(time, numbers.Real) or time > 0:
                #self.__entries.append(TableEntry(0, 0, self.__interpolation_strategies['hold'])) # interpolation strategy for first entry is disregarded, could be anything
                last_entry = TableEntry(0, 0, self.__interpolation_strategies['hold'])
            # ensure that the first entry is not negative
            elif isinstance(time, numbers.Real) and time < 0:
                raise ValueError("Time value must not be negative, was {}.".format(time))
            elif time == 0:
                last_entry = TableEntry(-1, 0, self.__interpolation_strategies['hold'])
        else:
            last_entry = self.__entries[-1]


        # Handle time parameter/value
        # first case: time is a real number
        if isinstance(time, numbers.Real):
            if isinstance(last_entry.t, ParameterDeclaration):
                # set maximum value of previous entry if not already set
                if last_entry.t.max_value == float('+inf'):
                    last_entry.t.max_value = time

                if time < last_entry.t.absolute_max_value:
                    raise ValueError("Argument time must be no smaller than previous time parameter declaration's" \
                                     " maximum value. Parameter '{0}', Maximum {1}, Provided: {2}."
                                     .format(last_entry.t.name, last_entry.t.absolute_max_value, time))

            # if time is a real number, ensure that is it not less than the previous entry
            elif time <= last_entry.t:
                raise ValueError("Argument time must be greater than previous time value {0}, was: {1}!".format(last_entry.t, time))

        # second case: time is a string -> Create a new ParameterDeclaration and continue third case
        elif isinstance(time, str):
            time = ParameterDeclaration(time)

        # third case: time is a ParameterDeclaration
        # if time is (now) a ParameterDeclaration, verify it, insert it and establish references/dependencies to previous entries if necessary
        if isinstance(time, ParameterDeclaration):
            if time.name in self.__voltage_parameter_declarations:
                raise ValueError("Cannot use already declared voltage parameter '{}' in time domain.".format(time.name))
            if time.name not in self.__time_parameter_declarations:
                if isinstance(time.min_value, ParameterDeclaration):
                    raise ValueError("A ParameterDeclaration for a time parameter may not have a minimum value reference" \
                                     " to another ParameterDeclaration object.")
                if isinstance(time.max_value, ParameterDeclaration):
                    raise ValueError("A ParameterDeclaration for a time parameter may not have a maximum value reference" \
                                     " to another ParameterDeclaration object.")

                # make a (shallow) copy of the ParameterDeclaration to ensure that it can't be changed from outside the Table
                time = ParameterDeclaration(time.name, min=time.min_value, max=time.max_value, default=time.default_value)
                # set minimum value if not previously set
                # if last_entry.t is a ParameterDeclaration, its max_value field will be set accordingly by the min_value setter,
                #  ensuring a correct boundary relationship between both declarations 
                if time.min_value == float('-inf'):
                    time.min_value = last_entry.t

                # Check dependencies between successive time parameters
                if isinstance(last_entry.t, ParameterDeclaration):
                    
                    if last_entry.t.max_value == float('inf'):
                        last_entry.t.max_value = time

                    if time.absolute_min_value < last_entry.t.absolute_min_value:
                        raise ValueError("Argument time's minimum value must be no smaller than the previous time" \
                                         "parameter declaration's minimum value. Parameter '{0}', Minimum {1}, Provided {2}."
                                         .format(last_entry.t.name, last_entry.t.absolute_min_value, time.min_value))
                    if time.absolute_max_value < last_entry.t.absolute_max_value:
                        raise ValueError("Argument time's maximum value must be no smaller than the previous time" \
                                         " parameter declaration's maximum value. Parameter '{0}', Maximum {1}, Provided {2}."
                                         .format(last_entry.t.name, last_entry.t.absolute_max_value, time.max_value))
                else:
                    if time.min_value < last_entry.t:
                        raise ValueError("Argument time's minimum value {0} must be no smaller than the previous time value {1}."
                                         .format(time.min_value, last_entry.t))
            else:
                raise ValueError("A time parameter with the name {} already exists.".format(time.name))


        # Handle voltage parameter/value
        # construct a ParameterDeclaration if voltage is a parameter name string
        if isinstance(voltage, str):
            voltage = ParameterDeclaration(voltage)
            
        # if voltage is (now) a ParameterDeclaration, make use of it
        if isinstance(voltage, ParameterDeclaration):
            # check whether a ParameterDeclaration with the same name already exists and, if so, use that instead
            # such that the same object is used consistently for one declaration
            if voltage.name in self.__voltage_parameter_declarations:
                voltage = self.__voltage_parameter_declarations[voltage.name]
            elif (voltage.name in self.__time_parameter_declarations or
                        (isinstance(time, ParameterDeclaration) and voltage.name == time.name)):
                    raise ValueError("Argument voltage <{}> must not refer to a time parameter declaration.".format(voltage.name))
            
        # no special action if voltage is a real number

        # add declaration if necessary
        if isinstance(time, ParameterDeclaration):
            self.__time_parameter_declarations[time.name] = time
        if isinstance(voltage, ParameterDeclaration):
            self.__voltage_parameter_declarations[voltage.name] = voltage
        # in case we need a time 0 entry previous to the new entry
        if not self.__entries and (not isinstance(time, numbers.Real) or time > 0):
                self.__entries.append(last_entry)
        # finally, add the new entry to the table 
        self.__entries.append(TableEntry(time, voltage, interpolation))
        
    @property
    def entries(self) -> List[TableEntry]:
        """Return an immutable copies of this TablePulseTemplate's entries."""
        return copy.deepcopy(self.__entries)

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return set(self.__time_parameter_declarations.keys()) | set(self.__voltage_parameter_declarations.keys())

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return a set of all parameter declaration objects of this TablePulseTemplate."""
        return set(self.__time_parameter_declarations.values()) | set(self.__voltage_parameter_declarations.values())

    def get_measurement_windows(self, parameters: Optional[Dict[str, Parameter]] = {}) -> List[MeasurementWindow]: # TODO: not very robust
        """Return all measurement windows defined in this PulseTemplate.
        
        A TablePulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """

        if not self.__is_measurement_pulse:
            return []
        else:
            instantiated_entries = self.get_entries_instantiated(parameters)
            return [(0, instantiated_entries[-1].t)]
    
    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def get_entries_instantiated(self, parameters: Dict[str, Parameter]) -> List[TableEntry]:
        """Return a list of all table entries with concrete values provided by the given parameters.
        """
        instantiated_entries = [] # type: List[TableEntry]
        for entry in self.__entries:
            time_value = None # type: float
            voltage_value = None # type: float
            # resolve time parameter references
            if isinstance(entry.t, ParameterDeclaration):
                parameter_declaration = entry.t # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameters[parameter_declaration.name])
                
                time_value = parameter_declaration.get_value(parameters)
            else:
                time_value = entry.t
            # resolve voltage parameter references only if voltageParameters argument is not None, otherwise they are irrelevant
            if isinstance(entry.v, ParameterDeclaration):
                parameter_declaration = entry.v # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameters[parameter_declaration.name])
                
                voltage_value= parameter_declaration.get_value(parameters)
            else:
                voltage_value = entry.v
            
            instantiated_entries.append(TableEntry(time_value, voltage_value, entry.interp))
            
        # ensure that no time value occurs twice
        previous_time = -1
        for (time, _, _) in instantiated_entries:
            if time <= previous_time:
                raise Exception("Time value {0} is smaller than the previous value {1}.".format(time, previous_time))
            previous_time = time
            
        return instantiated_entries

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        instantiated = self.get_entries_instantiated(parameters)
        if instantiated:
            instantiated = clean_entries(instantiated)
            waveform = TableWaveform(tuple(instantiated))
            sequencer.register_waveform(waveform)
            instruction_block.add_instruction_exec(waveform)

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool: 
        return any(parameters[name].requires_stop for name in parameters.keys() if (name in self.parameter_names) and not isinstance(parameters[name], numbers.Number))

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['is_measurement_pulse'] = self.__is_measurement_pulse
        data['time_parameter_declarations'] = [serializer._serialize_subpulse(self.__time_parameter_declarations[key]) for key in sorted(self.__time_parameter_declarations.keys())]
        data['voltage_parameter_declarations'] = [serializer._serialize_subpulse(self.__voltage_parameter_declarations[key]) for key in sorted(self.__voltage_parameter_declarations.keys())]
        entries = []
        for (time, voltage, interpolation) in self.__entries:
            if isinstance(time, ParameterDeclaration):
                time = time.name
            if isinstance(voltage, ParameterDeclaration):
                voltage = voltage.name
            entries.append((time, voltage, str(interpolation)))
        data['entries'] = entries
        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    time_parameter_declarations: Iterable[Any],
                    voltage_parameter_declarations: Iterable[Any],
                    entries: Iterable[Any],
                    is_measurement_pulse: bool,
                    identifier: Optional[str]=None) -> 'TablePulseTemplate':
        time_parameter_declarations = {declaration['name']: serializer.deserialize(declaration) for declaration in time_parameter_declarations}
        voltage_parameter_declarations = {declaration['name']: serializer.deserialize(declaration) for declaration in voltage_parameter_declarations}

        template = TablePulseTemplate(is_measurement_pulse, identifier=identifier)

        for (time, voltage, interpolation) in entries:
            if isinstance(time, str):
                time = time_parameter_declarations[time]
            if isinstance(voltage, str):
                voltage = voltage_parameter_declarations[voltage]
            template.add_entry(time, voltage, interpolation=interpolation)

        return template




def clean_entries(entries: List[TableEntry]) -> List[TableEntry]:
    """ Checks if two subsequent values have the same voltage value. If so, the second is redundant and removed in-place."""
    if not entries:
        return []
    length = len(entries)
    if length < 3: # for less than 3 points all are necessary
       return entries
    for index in range(length-2, 1, -1):
        previous_step = entries[index - 1]
        step = entries[index]
        next_step = entries[index + 1]
        if step.v == previous_step.v:
            if step.v == next_step.v:
                entries.pop(index)
    return entries

""" Exceptions """
class ParameterNotInPulseTemplateException(Exception):
    """Indicates that a provided parameter was not declared in a PulseTemplate."""

    def __init__(self, name: str, pulse_template: PulseTemplate) -> None:
        super().__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self) -> str:
        return "Parameter {1} not found in pulse template {2}".format(self.name, self.pulse_template)

class ParameterNotIntegerException(Exception):

    def __init__(self, parameter_name: str, parameter_value: float) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def __str__(self) -> str:
        return "The parameter <{}> must have an integral value (was <{}>) for use as repetition count.".format(
            self.parameter_name, self.parameter_value)


class MissingParameterDeclarationException(Exception):

    def __init__(self, template: PulseTemplate, missing_delcaration: str) -> None:
        super().__init__()
        self.template = template
        self.missing_declaration = missing_delcaration

    def __str__(self) -> str:
        return "A mapping for template {} requires a parameter '{}' which has not been declared as an external" \
               " parameter of the SequencePulseTemplate.".format(self.template, self.missing_declaration)


class MissingMappingException(Exception):

    def __init__(self, template, key) -> None:
        super().__init__()
        self.key = key
        self.template = template

    def __str__(self) -> str:
        return "The template {} needs a mapping function for parameter {}". format(self.template, self.key)


class UnnecessaryMappingException(Exception):

    def __init__(self, template, key):
        super().__init__()
        self.template = template
        self.key = key

    def __str__(self) -> str:
        return "Mapping function for parameter '{}', which template {} does not need".format(self.key, self.template)

class RuntimeMappingError(Exception):
    def __init__(self, template, subtemplate, outer_key, inner_key):
        self.template = template
        self.subtemplate = subtemplate
        self.outer_key = outer_key
        self.inner_key = inner_key

    def __str__(self):
        return ("An error occurred in the mapping function from {} to {}."
                " The mapping function for inner parameter '{}' requested"
                " outer parameter '{}', which was not provided.").format(self.template, self.subtemplate, self.inner_key, self.outer_key)

class ParameterValueIllegalException(Exception):
    """Indicates that the value provided for a parameter is illegal, i.e., is outside the parameter's bounds or of wrong type."""

    def __init__(self, parameter_declaration: ParameterDeclaration, parameter: Parameter) -> None:
        super().__init__()
        self.parameter = parameter
        self.parameter_declaration = parameter_declaration

    def __str__(self) -> str:
        return "The value {0} provided for parameter {1} is illegal (min = {2}, max = {3})".format(
            float(self.parameter), self.parameter_declaration.name, self.parameter_declaration.min_value,
            self.parameter_declaration.max_value)