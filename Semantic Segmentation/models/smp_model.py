import segmentation_models_pytorch as smp


class smp_models(object):
    def __init__(self, encoder, channels, classnums, activate):
        self.encoder = encoder
        self.channels = channels
        self.classnums = classnums
        self.activate = activate

    def unet(self):
        model = smp.UnetPlusPlus(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def unetplusplus(self):
        model = smp.UnetPlusPlus(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def deeplabv3plus(self):
        model = smp.DeepLabV3Plus(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def deeplabv3(self):
        model = smp.DeepLabV3(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def pan(self):
        model = smp.PAN(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def fpn(self):
        model = smp.FPN(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def psp(self):
        model = smp.PSPNet(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def manet(self):
        model = smp.MAnet(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model

    def linknet(self):
        model = smp.Linknet(encoder_name=self.encoder, in_channels=self.channels, classes=self.classnums, activation=self.activate)
        return model
