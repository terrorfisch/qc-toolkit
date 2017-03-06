import unittest


def get_pulse():
    from qctoolkit.pulses import TablePulseTemplate as TPT, SequencePulseTemplate as SPT, RepetitionPulseTemplate as RPT

    ramp = TPT(identifier='ramp', channels={'out', 'trigger'})
    ramp.add_entry(0, 'start', channel='out')
    ramp.add_entry('duration', 'stop', 'linear', channel='out')

    ramp.add_entry(0, 1, channel='trigger')
    ramp.add_entry(0, 'duration', 'hold', channel='trigger')

    ramp.add_measurement_declaration('meas', 0, 'duration')

    base = SPT([(ramp, dict(start='min', stop='max', duration='tau/3'), dict(meas='A')),
                (ramp, dict(start='max', stop='max', duration='tau/3'), dict(meas='B')),
                (ramp, dict(start='max', stop='min', duration='tau/3'), dict(meas='C'))], {'min', 'max', 'tau'})

    repeated = RPT(base, 'n')

    return repeated


class TaborTests(unittest.TestCase):
    def test_all(self):
        from qctoolkit.hardware.awgs.tabor import TaborChannelPair, TaborAWGRepresentation
        import warnings
        tawg = TaborAWGRepresentation(r'USB0::0x168C::0x2184::0000216488::INSTR')
        tchannelpair = TaborChannelPair(tawg, (1, 2), 'TABOR_AB')
        tawg.paranoia_level = 2

        warnings.simplefilter('error', Warning)

        from qctoolkit.hardware.dacs.alazar import AlazarCard
        import atsaverage.server

        if not atsaverage.server.Server.default_instance.running:
            atsaverage.server.Server.default_instance.start(key=b'guest')

        import atsaverage.core

        alazar = AlazarCard(atsaverage.core.getLocalCard(1, 1))
        alazar.register_mask_for_channel('A', 0)
        alazar.register_mask_for_channel('B', 0)
        alazar.register_mask_for_channel('C', 0)



        from qctoolkit.hardware.setup import HardwareSetup, PlaybackChannel

        hardware_setup = HardwareSetup()
        hardware_setup.register_dac(alazar)
        hardware_setup.set_channel('TABOR_A', PlaybackChannel(tchannelpair, 1))
        hardware_setup.set_channel('TABOR_B', PlaybackChannel(tchannelpair, 2))

        repeated = get_pulse()

        from qctoolkit.pulses.sequencing import Sequencer

        sequencer = Sequencer()
        sequencer.push(repeated, dict(n=1000, min=-0.5, max=0.5, tau=192*3), channel_mapping={'default': 'TABOR_A'},
                       window_mapping=dict(A='A', B='B', C='C'))
        instruction_block = sequencer.build()

        hardware_setup.register_program('test', instruction_block)

