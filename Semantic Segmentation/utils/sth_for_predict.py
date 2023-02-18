import numpy as np


def TifCroppingArray(img, SideLength, size=512):
    #  裁剪链表
    TifArrayReturn = []

    ColumnNum = int((img.shape[0] - SideLength * 2) / (size - SideLength * 2))

    RowNum = int((img.shape[1] - SideLength * 2) / (size - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (size - SideLength * 2) : i * (size - SideLength * 2) + size,
                          j * (size - SideLength * 2) : j * (size - SideLength * 2) + size]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    for i in range(ColumnNum):
        cropped = img[i * (size - SideLength * 2) : i * (size - SideLength * 2) + size,
                      (img.shape[1] - size) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - size) : img.shape[0],
                      j * (size-SideLength*2) : j * (size - SideLength * 2) + size]
        TifArray.append(cropped)

    cropped = img[(img.shape[0] - size) : img.shape[0],
                  (img.shape[1] - size) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)

    ColumnOver = (img.shape[0] - SideLength * 2) % (size - SideLength * 2) + SideLength

    RowOver = (img.shape[1] - SideLength * 2) % (size - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(img, shape, TifArray, RepetitiveLength, RowOver, ColumnOver, size):
    result = np.zeros(shape, np.uint8)
    j = 0
    for i, img in enumerate(img):
        img = np.squeeze(img, 0)
        if i % len(TifArray[0]) == 0:
            if j == 0:
                result[0 : size - RepetitiveLength, 0: size - RepetitiveLength] = img[0 : size - RepetitiveLength, 0: size - RepetitiveLength]
            elif j == len(TifArray) - 1:
                result[shape[0] - ColumnOver: shape[0], 0: size - RepetitiveLength] = img[size - ColumnOver: size, 0: size - RepetitiveLength]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength,
                       0:size-RepetitiveLength] = img[RepetitiveLength: size - RepetitiveLength, 0: size - RepetitiveLength]
        elif i % len(TifArray[0]) == len(TifArray[0]) - 1:
            if j == 0:
                result[0: size - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: size - RepetitiveLength, size -  RowOver: size]
            elif j == len(TifArray) - 1:
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[size - ColumnOver : size, size - RowOver : size]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : size - RepetitiveLength, size - RowOver : size]
            j = j + 1
        else:
            if j == 0:
                result[0 : size - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : size - RepetitiveLength, RepetitiveLength : size - RepetitiveLength]
            if j == len(TifArray) - 1:
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[size - ColumnOver : size, RepetitiveLength : size - RepetitiveLength]
            else:
                result[j * (size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (size - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : size - RepetitiveLength, RepetitiveLength : size - RepetitiveLength]
    return result