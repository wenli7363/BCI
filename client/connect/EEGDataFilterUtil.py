channelCount = 32
IIR_LENGTH = 10
DEN = -0.8230
NUM = 0.9115
IIRBuffer = [[0.0] * (IIR_LENGTH + 1) for _ in range(channelCount)]
offset = [0] * channelCount
offset_end = [0] * channelCount

IIR_SECTOR = 5
iir_40_params_low = [
    0.015120188, 0.015120188*2, 0.015120188, 1.864625546, -0.9251063007,
    0.014114816, 0.014114816*2, 0.014114816, 1.740642796, -0.7971020623,
    0.013359200, 0.013359200*2, 0.013359200, 1.647459981, -0.7008967811,
    0.012859054, 0.012859054*2, 0.012859054, 1.585781925, -0.6372181440,
    0.012610842, 0.012610842*2, 0.012610842, 1.555172301, -0.6056156703
]
Xn = [[0.0] * channelCount for _ in range(IIR_SECTOR + 1)]
Xn1 = [[0.0] * channelCount for _ in range(IIR_SECTOR + 1)]
Xn2 = [[0.0] * channelCount for _ in range(IIR_SECTOR + 1)]

iir_70_params_low = [
    0.044614833, 0.044614833*2, 0.044614833, 1.696646336, -0.8751056697,
    0.039878055, 0.039878055*2, 0.039878055, 1.516512594, -0.6760248163,
    0.036574835, 0.036574835*2, 0.036574835, 1.390895281, -0.5371946248,
    0.034498648, 0.034498648*2, 0.034498648, 1.311940461, -0.4499350547,
    0.033498927, 0.033498927*2, 0.033498927, 1.273922324, -0.4079180333
]
Yn = [[0.0] * 10 for _ in range(IIR_SECTOR + 1)]
Yn1 = [[0.0] * 10 for _ in range(IIR_SECTOR + 1)]
Yn2 = [[0.0] * 10 for _ in range(IIR_SECTOR + 1)]

def combFilter(mv, channel):
    global offset, offset_end, IIRBuffer
    offset_end[channel] = offset[channel] + 1
    if offset_end[channel] > IIR_LENGTH:
        offset_end[channel] = 0
    IIRBuffer[channel][offset[channel]] = mv - IIRBuffer[channel][offset_end[channel]] * DEN
    fSum = IIRBuffer[channel][offset[channel]] * NUM - IIRBuffer[channel][offset_end[channel]] * NUM
    offset[channel] += 1
    if offset[channel] > IIR_LENGTH:
        offset[channel] = 0
    return fSum

def lowPass40HzFilter(mv, ch):
    global Xn, Xn1, Xn2
    offset_l = 0
    Xn[0][ch] = mv
    for i in range(IIR_SECTOR):
        Xn[i+1][ch] = (
            iir_40_params_low[offset_l] * Xn[i][ch] +
            iir_40_params_low[offset_l+1] * Xn1[i][ch] +
            iir_40_params_low[offset_l+2] * Xn2[i][ch] +
            iir_40_params_low[offset_l+3] * Xn1[i+1][ch] +
            iir_40_params_low[offset_l+4] * Xn2[i+1][ch]
        )
        offset_l += 5
        Xn2[i][ch] = Xn1[i][ch]
        Xn1[i][ch] = Xn[i][ch]
    Xn2[IIR_SECTOR][ch] = Xn1[IIR_SECTOR][ch]
    Xn1[IIR_SECTOR][ch] = Xn[IIR_SECTOR][ch]
    return Xn[IIR_SECTOR][ch]

def lowPass70HzFilter(mv, ch):
    global Yn, Yn1, Yn2
    offset_l = 0
    Yn[0][ch] = mv
    for i in range(IIR_SECTOR):
        Yn[i+1][ch] = (
            iir_70_params_low[offset_l] * Yn[i][ch] +
            iir_70_params_low[offset_l+1] * Yn1[i][ch] +
            iir_70_params_low[offset_l+2] * Yn2[i][ch] +
            iir_70_params_low[offset_l+3] * Yn1[i+1][ch] +
            iir_70_params_low[offset_l+4] * Yn2[i+1][ch]
        )
        offset_l += 5
        Yn2[i][ch] = Yn1[i][ch]
        Yn1[i][ch] = Yn[i][ch]
    Yn2[IIR_SECTOR][ch] = Yn1[IIR_SECTOR][ch]
    Yn1[IIR_SECTOR][ch] = Yn[IIR_SECTOR][ch]
    return Yn[IIR_SECTOR][ch]
