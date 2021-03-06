import sys
import unittest
import importlib

from typing import List
from copy import copy, deepcopy

import numpy as np

from tests.hardware.dummy_modules import import_package


class DummyTaborProgramClass:
    def __init__(self, segments=None, segment_lengths=None,
                 sequencer_tables=None, advanced_sequencer_table=None, waveform_mode=None):
        self.program = None
        self.device_properties = None
        self.channels = None
        self.markers = None

        self.segment_lengths = segment_lengths
        self.segments = segments

        self.sequencer_tables = sequencer_tables
        self.advanced_sequencer_table = advanced_sequencer_table
        self.waveform_mode = waveform_mode

        self.created = []

    def __call__(self, program, device_properties, channels, markers):
        self.program = program
        self.device_properties = device_properties
        self.channels = channels
        self.markers = markers

        class DummyTaborProgram:
            def __init__(self, class_obj: DummyTaborProgramClass):
                self.sampled_segments_calls = []
                self.class_obj = class_obj
                self.waveform_mode = class_obj.waveform_mode
            def sampled_segments(self, sample_rate, voltage_amplitude, voltage_offset, voltage_transformation):
                self.sampled_segments_calls.append((sample_rate, voltage_amplitude, voltage_offset, voltage_transformation))
                return self.class_obj.segments, self.class_obj.segment_lengths
            def get_sequencer_tables(self):
                return self.class_obj.sequencer_tables
            def get_advanced_sequencer_table(self):
                return self.class_obj.advanced_sequencer_table
        self.created.append(DummyTaborProgram(self))
        return self.created[-1]


class TaborDummyBasedTest(unittest.TestCase):
    to_unload = ['pytabor', 'pyvisa', 'visa', 'teawg', 'qctoolkit', 'tests.pulses.sequencing_dummies']
    backup_modules = dict()

    @classmethod
    def unload_package(cls, package_name):
        modules_to_delete = [module_name for module_name in sys.modules if module_name.startswith(package_name)]

        for module_name in modules_to_delete:
            del sys.modules[module_name]

    @classmethod
    def backup_package(cls, package_name):
        cls.backup_modules[package_name] = [(module_name, module)
                                            for module_name, module in sys.modules.items()
                                            if module_name.startswith(package_name)]

    @classmethod
    def restore_packages(cls):
        for package, module_list in cls.backup_modules.items():
            for module_name, module in module_list:
                sys.modules[module_name] = module

    @classmethod
    def setUpClass(cls):
        for u in cls.to_unload:
            cls.backup_package(u)

        for u in cls.to_unload:
            cls.unload_package(u)

        import_package('pytabor')
        import_package('pyvisa')
        import_package('teawg')

    @classmethod
    def tearDownClass(cls):
        for u in cls.to_unload:
            cls.unload_package(u)

        cls.restore_packages()

    def setUp(self):
        from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation
        self.instrument = TaborAWGRepresentation('main_instrument',
                                                 reset=True,
                                                 paranoia_level=2,
                                                 mirror_addresses=['mirror_instrument'])
        self.instrument.main_instrument.visa_inst.answers[':OUTP:COUP'] = 'DC'
        self.instrument.main_instrument.visa_inst.answers[':VOLT'] = '1.0'
        self.instrument.main_instrument.visa_inst.answers[':FREQ:RAST'] = '1e9'
        self.instrument.main_instrument.visa_inst.answers[':VOLT:HV'] = '0.7'

    @property
    def awg_representation(self):
        return self.instrument

    @property
    def channel_pair(self):
        return self.awg_representation.channel_pair_AB

    def reset_instrument_logs(self):
        for device in self.instrument.all_devices:
            device.logged_commands = []
            device._send_binary_data_calls = []
            device._download_adv_seq_table_calls = []
            device._download_sequencer_table_calls = []

    def assertAllCommandLogsEqual(self, expected_log: List):
        for device in self.instrument.all_devices:
            self.assertEqual(device.logged_commands, expected_log)


class TaborAWGRepresentationDummyBasedTests(TaborDummyBasedTest):
    def test_send_cmd(self):
        self.reset_instrument_logs()

        self.instrument.send_cmd('bleh', paranoia_level=3)

        self.assertAllCommandLogsEqual([((), dict(paranoia_level=3, cmd_str='bleh'))])

        self.instrument.send_cmd('bleho')
        self.assertAllCommandLogsEqual([((), dict(paranoia_level=3, cmd_str='bleh')),
                                        ((), dict(cmd_str='bleho', paranoia_level=None))])

    def test_trigger(self):
        self.reset_instrument_logs()
        self.instrument.trigger()

        self.assertAllCommandLogsEqual([((), dict(cmd_str=':TRIG', paranoia_level=None))])

    def test_paranoia_level(self):
        self.assertEqual(self.instrument.paranoia_level, self.instrument.main_instrument.paranoia_level)
        self.instrument.paranoia_level = 30
        for device in self.instrument.all_devices:
            self.assertEqual(device.paranoia_level, 30)

    def test_enable(self):
        self.reset_instrument_logs()
        self.instrument.enable()

        expected_commands = [':ENAB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)


class TaborChannelPairTests(TaborDummyBasedTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from qctoolkit.hardware.awgs.tabor import TaborChannelPair, TaborProgramMemory, TaborSegment, TaborSequencing
        from qctoolkit.pulses.table_pulse_template import TableWaveform
        from qctoolkit.pulses.interpolation import HoldInterpolationStrategy
        from qctoolkit.hardware.program import Loop

        from tests.pulses.sequencing_dummies import DummyWaveform

        from qctoolkit.hardware.util import make_combined_wave

        cls.DummyWaveform = DummyWaveform
        cls.TaborChannelPair = TaborChannelPair
        cls.TaborProgramMemory = TaborProgramMemory
        cls.TableWaveform = TableWaveform
        cls.HoldInterpolationStrategy = HoldInterpolationStrategy
        cls.Loop = Loop
        cls.TaborSegment = TaborSegment
        cls.make_combined_wave = staticmethod(make_combined_wave)
        cls.TaborSequencing = TaborSequencing

    def test_copy(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        with self.assertRaises(NotImplementedError):
            copy(channel_pair)
        with self.assertRaises(NotImplementedError):
            deepcopy(channel_pair)

    def test_init(self):
        with self.assertRaises(ValueError):
            self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 3))

    def test_free_program(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        with self.assertRaises(KeyError):
            channel_pair.free_program('test')

        program = self.TaborProgramMemory(np.array([1, 2], dtype=np.int64), None)

        channel_pair._segment_references = np.array([1, 3, 1, 0])
        channel_pair._known_programs['test'] = program
        self.assertIs(channel_pair.free_program('test'), program)

        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 0]))


    def test_upload_exceptions(self):

        wv = self.TableWaveform(1, [(0, 0.1, self.HoldInterpolationStrategy()),
                                    (192, 0.1, self.HoldInterpolationStrategy())])

        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        program = self.Loop(waveform=wv)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2, 3), (5, 6), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (5, 6, 'a'), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x,))

        channel_pair._known_programs['test'] = self.TaborProgramMemory(np.array([0]), None)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x, lambda x: x))

    def test_upload(self):
        segments = np.array([1, 2, 3, 4, 5])
        segment_lengths = np.array([0, 16, 0, 16, 0], dtype=np.uint16)

        segment_references = np.array([1, 1, 2, 0, 1], dtype=np.uint32)

        w2s = np.array([-1, -1, 1, 2, -1], dtype=np.int64)
        ta = np.array([True, False, False, False, True])
        ti = np.array([-1, 3, -1, -1, -1])

        to_restore = sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram
        my_class = DummyTaborProgramClass(segments=segments, segment_lengths=segment_lengths)
        sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram = my_class
        try:
            program = self.Loop(waveform=self.DummyWaveform(duration=192))

            channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
            channel_pair._segment_references = segment_references

            def dummy_find_place(segments_, segement_lengths_):
                self.assertIs(segments_, segments)
                self.assertIs(segment_lengths, segement_lengths_)
                return w2s, ta, ti

            def dummy_upload_segment(segment_index, segment):
                self.assertEqual(segment_index, 3)
                self.assertEqual(segment, 2)

            def dummy_amend_segments(segments_):
                np.testing.assert_equal(segments_, np.array([1, 5]))
                return np.array([5, 6], dtype=np.int64)

            channel_pair._find_place_for_segments_in_memory = dummy_find_place
            channel_pair._upload_segment = dummy_upload_segment
            channel_pair._amend_segments = dummy_amend_segments

            channel_pair.upload('test', program, (1, None), (None, None), (lambda x: x, lambda x: x))

            self.assertIs(my_class.program, program)

            # the other references are increased in amend and upload segment method
            np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 3, 0, 1]))

            self.assertEqual(len(channel_pair._known_programs), 1)
            np.testing.assert_equal(channel_pair._known_programs['test'].waveform_to_segment,
                                    np.array([5, 3, 1, 2, 6], dtype=np.int64))
            self.assertIs(channel_pair._known_programs['test'].program, my_class.created[0])

        finally:
            sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram = to_restore

    def test_find_place_for_segments_in_memory(self):
        def hash_based_on_dir(ch):
            hash_list = []
            for d in dir(ch):
                o = getattr(ch, d)
                if isinstance(o, np.ndarray):
                    hash_list.append(hash(o.tobytes()))
                else:
                    try:
                        hash_list.append(hash(o))
                    except TypeError:
                        pass
            return hash(tuple(hash_list))

        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        # empty
        segments = np.asarray([-5, -6, -7, -8, -9])
        segment_lengths = 192 + np.asarray([32, 16, 64, 32, 16])

        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, True, True, True])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # all new segments
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 16, 0], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 1, 1, 2, 1], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, True, True, True])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # some known segments
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 64, 0, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, -7, 5, -9], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 1, 1, 2, 1, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, 3, -1, 5])
        self.assertEqual(ta.tolist(), [True, True, False, True, False])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # insert some segments with same length
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 64, 0, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 0, 1, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, False, False, True, True])
        self.assertEqual(ti.tolist(), [-1, 1, 3, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # insert some segments with smaller length
        channel_pair._segment_capacity = 192 + np.asarray([0, 80, 32, 64, 96, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 1, 0, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, False, False, True])
        self.assertEqual(ti.tolist(), [-1, -1, 4, 1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # mix everything
        segments = np.asarray([-5, -6, -7, -8, -9, -10, -11])
        segment_lengths = 192 + np.asarray([32, 16, 64, 32, 16, 0, 0])

        channel_pair._segment_capacity = 192 + np.asarray([0, 80, 32, 64, 32, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, -8, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1,    -1,   -1,    4,     -1,     -1, -1])
        self.assertEqual(ta.tolist(),  [False, True, False, False, True, True, True])
        self.assertEqual(ti.tolist(),  [1,     -1,   3,     -1,    -1,   -1,   -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

    def test_upload_segment(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = channel_pair._segment_capacity.copy()

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        segment = self.TaborSegment(np.ones(192+16, dtype=np.uint16), np.zeros(192+16, dtype=np.uint16), None, None)
        segment_binary = segment.get_as_binary()
        with self.assertRaises(ValueError):
            channel_pair._upload_segment(3, segment)

        with self.assertRaises(ValueError):
            channel_pair._upload_segment(0, segment)

        channel_pair._upload_segment(2, segment)
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 16, 16, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, hash(segment), 4], dtype=np.int64))

        expected_commands = [':INST:SEL 1', ':INST:SEL 1', ':INST:SEL 1',
                             ':TRAC:DEF 3, 208',
                             ':TRAC:SEL 3',
                             ':TRAC:MODE COMB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)

        expected_send_binary_data_log = [(':TRAC:DATA', segment_binary, None)]
        for device in self.instrument.all_devices:
            np.testing.assert_equal(device._send_binary_data_calls, expected_send_binary_data_log)

    def test_amend_segments_flush(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.main_instrument.paranoia_level = 0
        self.instrument.main_instrument.logged_commands = []
        self.instrument.main_instrument.logged_queries = []
        self.instrument.main_instrument._send_binary_data_calls = []
        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 16, 16, 32], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [self.TaborSegment(0*data, 1*data, None, None),
                    self.TaborSegment(1*data, 2*data, None, None)]

        channel_pair._amend_segments(segments)

        expected_references = np.array([1, 2, 0, 1, 1, 1], dtype=np.uint32)
        expected_capacities = 192 + np.array([0, 16, 32, 32, 0, 0], dtype=np.uint32)
        expected_lengths = 192 + np.array([0, 16, 16, 32, 0, 0], dtype=np.uint32)
        expected_hashes = np.array([1, 2, 3, 4, hash(segments[0]), hash(segments[1])], dtype=np.int64)

        np.testing.assert_equal(channel_pair._segment_references, expected_references)
        np.testing.assert_equal(channel_pair._segment_capacity, expected_capacities)
        np.testing.assert_equal(channel_pair._segment_lengths, expected_lengths)
        np.testing.assert_equal(channel_pair._segment_hashes, expected_hashes)

        expected_commands = [':INST:SEL 1',
                             ':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 3,208']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)
        #self.assertEqual(expected_log, instrument.main_instrument.logged_commands)

        expected_download_segment_calls = [(expected_capacities, ':SEGM:DATA', None)]
        np.testing.assert_equal(self.instrument.main_instrument._download_segment_lengths_calls, expected_download_segment_calls)

        expected_bin_blob = self.make_combined_wave(segments)
        expected_send_binary_data_log = [(':TRAC:DATA', expected_bin_blob, None)]
        np.testing.assert_equal(self.instrument.main_instrument._send_binary_data_calls, expected_send_binary_data_log)

    def test_amend_segments_iter(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [self.TaborSegment(0*data, 1*data, None, None),
                    self.TaborSegment(1*data, 2*data, None, None)]

        indices = channel_pair._amend_segments(segments)

        expected_references = np.array([1, 2, 0, 1, 1, 1], dtype=np.uint32)
        expected_capacities = 192 + np.array([0, 16, 32, 32, 0, 0], dtype=np.uint32)
        expected_lengths = 192 + np.array([0, 0, 16, 16, 0, 0], dtype=np.uint32)
        expected_hashes = np.array([1, 2, 3, 4, hash(segments[0]), hash(segments[1])], dtype=np.int64)

        np.testing.assert_equal(channel_pair._segment_references, expected_references)
        np.testing.assert_equal(channel_pair._segment_capacity, expected_capacities)
        np.testing.assert_equal(channel_pair._segment_lengths, expected_lengths)
        np.testing.assert_equal(channel_pair._segment_hashes, expected_hashes)

        np.testing.assert_equal(indices, np.array([4, 5], dtype=np.int64))

        expected_commands = [':INST:SEL 1',
                             ':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 5,192',
                             ':TRAC:DEF 6,192']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)

        expected_download_segment_calls = []
        for device in self.instrument.all_devices:
            self.assertEqual(device._download_segment_lengths_calls, expected_download_segment_calls)

        expected_bin_blob = self.make_combined_wave(segments)
        expected_send_binary_data_log = [(':TRAC:DATA', expected_bin_blob, None)]
        for device in self.instrument.all_devices:
            np.testing.assert_equal(device._send_binary_data_calls, expected_send_binary_data_log)

    def test_cleanup(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.instrument._send_binary_data_calls = []

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        channel_pair.cleanup()
        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 1], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 0, 16, 16], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, 3, 4], dtype=np.int64))

        channel_pair._segment_references = np.array([1, 2, 0, 1, 0], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16, 0], dtype=np.uint32)
        channel_pair._segment_hashes = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        channel_pair.cleanup()
        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 1], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 0, 16, 16], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, 3, 4], dtype=np.int64))

    def test_remove(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        calls = []

        program_name = 'test'
        def dummy_free_program(name):
            self.assertIs(name, program_name)
            calls.append('free_program')

        def dummy_cleanup():
            calls.append('cleanup')

        channel_pair.cleanup = dummy_cleanup
        channel_pair.free_program = dummy_free_program

        channel_pair.remove(program_name)
        self.assertEqual(calls, ['free_program', 'cleanup'])

    def test_change_armed_program_single_sequence(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.reset_instrument_logs()

        advanced_sequencer_table = [(2, 1, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        w2s = np.array([2, 5, 3, 1])

        expected_sequencer_table = [(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]
        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=self.TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = self.TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [expected_sequencer_table]]

        for device in self.instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)

    def test_change_armed_program_single_waveform(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.reset_instrument_logs()

        advanced_sequencer_table = [(1, 1, 0)]
        sequencer_tables = [[(10, 0, 0)]]
        w2s = np.array([4])

        expected_sequencer_table = [(10, 5, 0), (1, 1, 0), (1, 1, 0)]
        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=self.TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = self.TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (1, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [expected_sequencer_table]]

        for device in self.instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)

    def test_change_armed_program_advanced_sequence(self):
        channel_pair = self.TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.instrument._send_binary_data_calls = []

        self.reset_instrument_logs()

        advanced_sequencer_table = [(2, 1, 0), (3, 2, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)],
                            [(4, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        wf_idx2seg_idx = np.array([2, 5, 3, 1])

        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]
        expected_sequencer_tables = [[(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)],
                                     [(4, 6, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=self.TaborSequencing.ADVANCED)(None, None, None, None)

        channel_pair._known_programs['test'] = self.TaborProgramMemory(wf_idx2seg_idx, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (3, 3, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in expected_sequencer_tables]

        for device in self.instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)
