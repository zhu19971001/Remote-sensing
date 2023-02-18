import numpy as np
import cv2
from utils.onehot import pixel_to_index
from utils.read_write import file_name


__all__ = ['SegmentationMetric']  # 表示SegmentationMetric类可被调用


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):

        assert imgPredict.shape == imgLabel.shape, "预测图像尺寸与标签图像尺寸不等"
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

# 测试内容
if __name__ == '__main__':
    test_predi_folder = r''
    test_label_folder = r''

    pre_name_list = file_name(test_predi_folder)
    lab_name_list = file_name(test_label_folder)

    for i in range(len(pre_name_list)):
        pre_name_i = test_predi_folder + "\\" + pre_name_list[i]
        lab_name_i = test_label_folder + "\\" + lab_name_list[i]

        pre_i_array = cv2.imread(pre_name_i)
        lab_i_array = cv2.imread(lab_name_i)
        lab_i_array = pixel_to_index(lab_i_array, test_label_folder, classnum=2)

        metric = SegmentationMetric(numClass=2)
        hist = metric.addBatch(pre_i_array, lab_i_array)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        IoU = metric.IntersectionOverUnion()
        mIoU = metric.meanIntersectionOverUnion()

    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)


