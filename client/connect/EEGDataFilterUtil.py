class EEGDataFilterUtil:
    def __init__(self):
        self.channel_count = 32
        self.IIR_LENGTH = 10
        self.DEN = -0.8230
        self.NUM = 0.9115
        self.IIRBuffer = [[0.0] * (self.IIR_LENGTH + 1) for _ in range(self.channel_count)]
        self.offset = [0] * self.channel_count
        self.offset_end = [0] * self.channel_count
        self.IIR_SECTOR = 5
        self.iir_40_params_low = [
            0.015120188, 0.030240376, 0.015120188, 1.864625546, -0.9251063007,
            0.014114816, 0.028229632, 0.014114816, 1.740642796, -0.7971020623,
            0.013359200, 0.026718400, 0.013359200, 1.647459981, -0.7008967811,
            0.012859054, 0.025718108, 0.012859054, 1.585781925, -0.6372181440,
            0.012610842, 0.025221684, 0.012610842, 1.555172301, -0.6056156703
        ]
        self.Xn = [[0.0] * self.channel_count for _ in range(self.IIR_SECTOR + 1)]
        self.Xn1 = [[0.0] * self.channel_count for _ in range(self.IIR_SECTOR + 1)]
        self.Xn2 = [[0.0] * self.channel_count for _ in range(self.IIR_SECTOR + 1)]
        self.iir_70_params_low = [
            0.044614833, 0.089229666, 0.044614833, 1.696646336, -0.8751056697,
            0.039878055, 0.07975611, 0.039878055, 1.516512594, -0.6760248163,
            0.036574835, 0.07314967, 0.036574835, 1.390895281, -0.5371946248,
            0.034498648, 0.068997296, 0.034498648, 1.311940461, -0.4499350547,
            0.033498927, 0.066997854, 0.033498927, 1.273922324, -0.4079180333
        ]
        self.Yn = [[0.0] * 10 for _ in range(self.IIR_SECTOR + 1)]
        self.Yn1 = [[0.0] * 10 for _ in range(self.IIR_SECTOR + 1)]
        self.Yn2 = [[0.0] * 10 for _ in range(self.IIR_SECTOR + 1)]

    def comb_filter(self, mv, channel):
        self.offset_end[channel] = self.offset[channel] + 1
        if self.offset_end[channel] > self.IIR_LENGTH:
            self.offset_end[channel] = 0
        self.IIRBuffer[channel][self.offset[channel]] = mv - \
            self.IIRBuffer[channel][self.offset_end[channel]] * self.DEN
        f_sum = self.IIRBuffer[channel][self.offset[channel]] * \
            self.NUM - \
            self.IIRBuffer[channel][self.offset_end[channel]] * \
            self.NUM
        self.offset[channel] += 1
        if self.offset[channel] > self.IIR_LENGTH:
            self.offset[channel] = 0
        return f_sum

    def low_pass_40Hz_filter(self, mv, ch):
        offset_l = 0
        self.Xn[0][ch] = mv
        for i in range(self.IIR_SECTOR):
            self.Xn[i + 1][ch] = \
                self.iir_40_params_low[offset_l] * self.Xn[i][ch] + \
                self.iir_40_params_low[offset_l + 1] * self.Xn1[i][ch] + \
                self.iir_40_params_low[offset_l + 2] * self.Xn2[i][ch] + \
                self.iir_40_params_low[offset_l + 3] * self.Xn1[i + 1][ch] + \
                self.iir_40_params_low[offset_l + 4] * self.Xn2[i + 1][ch]
            offset_l += 5
            self.Xn2[i][ch] = self.Xn1[i][ch]
            self.Xn1[i][ch] = self.Xn[i][ch]
        self.Xn2[self.IIR_SECTOR][ch] = self.Xn1[self.IIR_SECTOR][ch]
        self.Xn1[self.IIR_SECTOR][ch] = self.Xn[self.IIR_SECTOR][ch]
        return self.Xn[self.IIR_SECTOR][ch]

    def low_pass_70Hz_filter(self, mv, ch):
        offset_l = 0
        self.Yn[0][ch] = mv
        for i in range(self.IIR_SECTOR):
            self.Yn[i + 1][ch] = \
                self.iir_70_params_low[offset_l] * self.Yn[i][ch] + \
                self.iir_70_params_low[offset_l + 1] * self.Yn1[i][ch] + \
                self.iir_70_params_low[offset_l + 2] * self.Yn2[i][ch] + \
                self.iir_70_params_low[offset_l + 3] * self.Yn1[i + 1][ch] + \
                self.iir_70_params_low[offset_l + 4] * self.Yn2[i + 1][ch]
            offset_l += 5
            self.Yn2[i][ch] = self.Yn1[i][ch]
            self.Yn1[i][ch] = self.Yn[i][ch]
        self.Yn2[self.IIR_SECTOR][ch] = self.Yn1[self.IIR_SECTOR][ch]
        self.Yn1[self.IIR_SECTOR][ch] = self.Yn[self.IIR_SECTOR][ch]
        return self.Yn[self.IIR_SECTOR][ch]
