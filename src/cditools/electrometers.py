from ophyd import Component as Cpt, EpicsSignal, QuadEM


class F460EM(QuadEM):

    polarity = 'neg'

    # Channels
    ch0 = Cpt(EpicsSignal, 'Cur:I0-I')
    ch1 = Cpt(EpicsSignal, 'Cur:I1-I')
    ch2 = Cpt(EpicsSignal, 'Cur:I2-I')
    ch3 = Cpt(EpicsSignal, 'Cur:I3-I')

    # Channel Ranges
    ch0_range = Cpt(EpicsSignal, 'Ch0:Range-I')
    ch1_range = Cpt(EpicsSignal, 'Ch1:Range-I')
    ch2_range = Cpt(EpicsSignal, 'Ch2:Range-I')
    ch3_range = Cpt(EpicsSignal, 'Ch3:Range-I')


class I400EM(QuadEM):

    polarity = 'neg'

    # Channels
    ch0 = Cpt(EpicsSignal, 'IC1_MON')
    ch1 = Cpt(EpicsSignal, 'IC2_MON')
    ch2 = Cpt(EpicsSignal, 'IC3_MON')
    ch3 = Cpt(EpicsSignal, 'IC4_MON')

    # Channel Ranges
    ch_range = Cpt(EpicsSignal, ':RANGE_BP')
    # ch1_range = Cpt(EpicsSignal, 'Ch1:Range-I')
    # ch2_range = Cpt(EpicsSignal, 'Ch2:Range-I')
    # ch3_range = Cpt(EpicsSignal, 'Ch3:Range-I')


class I404EM(QuadEM):

    polarity = 'neg'

    # Channels
    ch0 = Cpt(EpicsSignal, 'I:R1-I')
    ch1 = Cpt(EpicsSignal, 'I:R2-I')
    ch2 = Cpt(EpicsSignal, 'I:R3-I')
    ch3 = Cpt(EpicsSignal, 'I:R4-I')

    # Channel Ranges
    ch_range = Cpt(EpicsSignal, 'Val:Rng-I')
    # ch1_range = Cpt(EpicsSignal, 'Ch1:Range-I')
    # ch2_range = Cpt(EpicsSignal, 'Ch2:Range-I')
    # ch3_range = Cpt(EpicsSignal, 'Ch3:Range-I')

# I400
ema_sys = 'XF:09IDA-BI'
ema_dev = '{i400:1}'
# I404
emb_sys = 'XF:09IDB-BI'
emb_dev = '{i404:1}'
# F460
emc_sys = 'XF:09IDC-BI'
emc_dev = '{f460:1}'

emc = F460EM(prefix=f"{emc_sys}{emc_dev}", name="emc")
ema = I400EM(prefix=f"{ema_sys}{ema_dev}", name="ema")
emb = I404EM(prefix=f"{emb_sys}{emb_dev}", name="emb")
