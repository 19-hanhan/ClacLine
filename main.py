"""
@Description :
@File        : main.py
@Project     : ClacLine
@Time        : 2022/4/3 15:30
@Author      : hanhan
@Software    : PyCharm
"""

import os

from Configs.Config import *
from Tools import *


def main():
    # 计算需要统计的文件夹的数量
    startPos = 0
    endCurvePos = len([lists for lists in os.listdir(curvePath) if os.path.isdir(os.path.join(curvePath, lists))])
    endPolylinePos = len(
        [lists for lists in os.listdir(polylinePath) if os.path.isdir(os.path.join(polylinePath, lists))])
    # endCurvePos = 11
    # endPolylinePos = -1

    # 处理曲线图
    for pos in range(startPos, endCurvePos):
        # 各个地址
        imgPath = curvePath + '/' + str(pos) + '/' + drawFileName
        drawWhitePath = curvePath + '/' + str(pos) + '/' + drawWhiteFileName
        borderPath = curvePath + '/' + str(pos) + '/' + borderFileName
        drawLinePath = curvePath + '/' + str(pos) + '/' + drawLineFileName
        dbPath = curvePath + '/' + str(pos) + '/' + dbFileName
        drawMarkPath = curvePath + '/' + str(pos) + '/' + drawMarkFileName
        ansPath = curvePath + '/' + str(pos) + '/' + ansFileName
        drawMarkPath2 = MarkImgPath2 + '/' + curvePath + '/' + str(pos) + '.png'
        drawTwoPath = curvePath + '/' + str(pos) + '/' + drawTwoFileName

        print("正在处理图片：" + imgPath)

        # 白化
        TurnWhite(imgPath, drawWhitePath)

        # 获取边框
        GetBorder(drawWhitePath, borderPath)

        # 二选一
        # 去除边框颜色
        ClearBorder(borderPath, drawWhitePath, drawLinePath)
        # 二值化
        # TurnTwo(drawWhitePath, drawTwoPath)

        # 获取五个点得结果
        GetAns(drawLinePath, borderPath, dbPath, drawMarkPath, ansPath, drawMarkPath2)
        # GetAns(drawTwoPath, borderPath, dbPath, drawMarkPath, ansPath, drawMarkPath2)

    # 处理折线图
    for pos in range(startPos, endPolylinePos):
        # 各个地址
        imgPath = polylinePath + '/' + str(pos) + '/' + drawFileName
        drawWhitePath = polylinePath + '/' + str(pos) + '/' + drawWhiteFileName
        borderPath = polylinePath + '/' + str(pos) + '/' + borderFileName
        drawLinePath = polylinePath + '/' + str(pos) + '/' + drawLineFileName
        dbPath = polylinePath + '/' + str(pos) + '/' + dbFileName
        drawMarkPath = polylinePath + '/' + str(pos) + '/' + drawMarkFileName
        ansPath = polylinePath + '/' + str(pos) + '/' + ansFileName
        drawMarkPath2 = MarkImgPath2 + '/' + polylinePath + '/' + str(pos)
        drawTwoPath = polylinePath + '/' + str(pos) + '/' + drawTwoFileName

        # 白化
        TurnWhite(imgPath, drawWhitePath)

        # 获取边框
        GetBorder(drawWhitePath, borderPath)

        # 去除边框颜色
        ClearBorder(borderPath, drawWhitePath, drawLinePath)
        # 二值化
        # TurnTwo(drawWhitePath, drawTwoPath)

        # 获取五个点得结果
        GetAns(drawLinePath, borderPath, dbPath, drawMarkPath, ansPath, drawMarkPath2)
        # GetAns(drawTwoPath, borderPath, dbPath, drawMarkPath, ansPath, drawMarkPath2)


if __name__ == '__main__':
    main()
