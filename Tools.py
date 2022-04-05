import cv2
import numpy as np


# 吴文龙

def GetBorder(imgPath, borderPath):
    """
    根据图片地址获取边框
    :param imgPath: 底色白化后的折线图
    :return: 包含四个点值的字典(upLeft, upRight, downLeft, downRight)
    """

    print("正在写入border信息...")
    img = cv2.imread(imgPath)
    row, col, type = img.shape  # 统计图片的横纵像素值
    # print("row = " + str(row) + ", col = " + str(col))

    cntRow = []  # 每一行非白像素个数
    maxRow = 0  # 行非白像素个数最大值
    for posRow in range(row):  # 循环计算行最大非白像素个数
        cnt = 0
        for posCol in range(col):
            # pass
            color = img[posRow][posCol]
            if list(color) != [255, 255, 255]:
                cnt += 1
        cntRow.append(cnt)
        maxRow = max(maxRow, cnt)

    # 与上面类似，对应为列
    cntCol = []
    maxCol = 0
    for posCol in range(col):
        cnt = 0
        for posRow in range(row):
            color = img[posRow][posCol]
            if list(color) != [255, 255, 255]:
                cnt += 1
        cntCol.append(cnt)
        maxCol = max(maxCol, cnt)

    # print(cntRow)
    # print(cntCol)
    # print("maxRow = " + str(maxRow) + ", maxCol = " + str(maxCol))

    lsPosRow = []
    for pos, item in enumerate(cntRow):
        if item > maxRow - 30:
            lsPosRow.append(pos)

    lsPosCol = []
    for pos, item in enumerate(cntCol):
        if item > maxCol - 30:
            lsPosCol.append(pos)

    # print(lsPosRow)
    # print(lsPosCol)

    dic = {}
    dic['upLeft'] = (lsPosRow[0], lsPosCol[0])
    dic['upRight'] = (lsPosRow[0], lsPosCol[len(lsPosCol) - 1])
    dic['downLeft'] = (lsPosRow[len(lsPosRow) - 1], lsPosCol[0])
    dic['downRight'] = (lsPosRow[len(lsPosRow) - 1], lsPosCol[len(lsPosCol) - 1])

    WriteDic(borderPath, dic)


def WriteDic(filePath, dic):
    """
    写入border信息
    :param filePath: border文件路径
    :param dic: border字典信息
    :return:
    """
    f = open(filePath, 'w', encoding='utf8')
    f.write('upLeft ' + str(dic['upLeft'][0]) + ' ' + str(dic['upLeft'][1]) + '\n');
    f.write('upRight ' + str(dic['upRight'][0]) + ' ' + str(dic['upRight'][1]) + '\n');
    f.write('downLeft ' + str(dic['downLeft'][0]) + ' ' + str(dic['downLeft'][1]) + '\n');
    f.write('downRight ' + str(dic['downRight'][0]) + ' ' + str(dic['downRight'][1]) + '\n');


# 康恬

def TurnWhite(imgPath, drawWhitePath):
    """
    将原图变为白底图片（折线图以外的区域都是白色）
    :param imgPath: 原图地址
    :param drawWhitePath: 保存白化图片draw_white的地址
    :return: 无
    """
    print('正在白化底色...')
    # 读入图片
    img = cv2.imread(imgPath)
    # 图片的列像素数量
    height = img.shape[0]
    # 图片的行像素数量
    weight = img.shape[1]
    # 图片原来的底色像素值（折线图以外的区域的颜色）
    (x, y, z) = img[0, 0]
    # 遍历列
    for row in range(height):
        # 遍历行
        for col in range(weight):
            # 该点像素值
            (b, g, r) = img[row, col]
            # 如果是底色就变白色
            if (b, g, r) == (x, y, z):
                img[row, col] = (255, 255, 255)
            # 否则如果是白色就变成原底色颜色
            elif (b, g, r) == (255, 255, 255):
                img[row, col] = (b, g, r)
    cv2.imwrite(drawWhitePath, img)


def ClearBorder(borderPath, drawWhitePath, drawLinePath):
    """
    去除边框颜色，将边框颜色变成白色，变成只有数字和线
    :param borderPath: 边框信息地址
    :param drawWhitePath: 白化图片的地址
    :param drawLinePath: 保存去边框图的地址
    :return: 无
    """
    print('正在清除边框颜色...')
    # 读取边框信息
    with open(borderPath, "r") as f:
        # 找到右上点的坐标
        borderTxt = f.read()
        border = borderTxt.split()
        # print(border)
        upRightX = int(border[4])
        upRightY = int(border[5])
        # 读入图片
        whiteImg = cv2.imread(drawWhitePath)
        # 图片的列像素数量
        height = whiteImg.shape[0]
        # 图片的行像素数量
        weight = whiteImg.shape[1]
        # 图片右上点的像数值
        ls = []
        for i in range(3):
            ls.append(list(whiteImg[upRightX + i, upRightY - i]))

        # print(ls)

        # 遍历列
        for row in range(height):
            # 遍历行
            for col in range(weight):
                # 该点像素值
                (b, g, r) = whiteImg[row, col]
                # 如果是右上点的颜色就变白色
                if [b, g, r] in ls:
                    whiteImg[row, col] = (255, 255, 255)
    cv2.imwrite(drawLinePath, whiteImg)


def TurnTwo(drawWhitePath, drawTwoPath):
    """
    将图片二值化处理
    :param drawWhitePath: 底色白化后的图片地址
    :param drawTwoPath: 保存二值化后的图片的地址
    :return: 无
    """
    print('正在图片二值化...')
    # 读入图片
    whiteImg = cv2.imread(drawWhitePath)
    img_gray = cv2.cvtColor(whiteImg, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    threshold_t = 0
    max_g = 0

    for t in range(255):
        front = img_gray[img_gray < t]
        back = img_gray[img_gray >= t]
        front_p = len(front) / (h * w)
        back_p = len(back) / (h * w)
        front_mean = np.mean(front) if len(front) > 0 else 0.
        back_mean = np.mean(back) if len(back) > 0 else 0.

        g = front_p * back_p * ((front_mean - back_mean) ** 2)
        if g > max_g:
            max_g = g
            threshold_t = t
    # print(f"threshold = {threshold_t}")
    img_gray[img_gray < threshold_t] = 0
    img_gray[img_gray >= threshold_t] = 255
    cv2.imwrite(drawTwoPath, img_gray)


# 刘晟

def changeColor(img, x, y, dic):
    """
    在中心点画上红色标记的函数
    :param img: cv2图片类型
    :param x: 中心点行坐标
    :param y: 中心点列坐标
    :param dic: 边界坐标
    :return:
    """
    base = 5
    red = [0, 0, 255]
    green = [0, 255, 0]
    for i in range(x - base, x + base):
        img[i][y] = red

    for j in range(y - base, y + base):
        img[x][j] = red

    for i in range(dic['upLeft'][0] - base, dic['upLeft'][0] + base):
        img[i][dic['upLeft'][1]] = green

    for j in range(dic['upLeft'][1] - base, dic['upLeft'][1] + base):
        img[dic['upLeft'][0]][j] = green

    for i in range(dic['upRight'][0] - base, dic['upRight'][0] + base):
        img[i][dic['upRight'][1]] = green

    for j in range(dic['upRight'][1] - base, dic['upRight'][1] + base):
        img[dic['upRight'][0]][j] = green

    for i in range(dic['downLeft'][0] - base, dic['downLeft'][0] + base):
        img[i][dic['downLeft'][1]] = green

    for j in range(dic['downLeft'][1] - base, dic['downLeft'][1] + base):
        img[dic['downLeft'][0]][j] = green

    for i in range(dic['downRight'][0] - base, dic['downRight'][0] + base):
        img[i][dic['downRight'][1]] = green

    for j in range(dic['downRight'][1] - base, dic['downRight'][1] + base):
        img[dic['downRight'][0]][j] = green


def fixBlack(img, y, up, down, dic):
    """
    找到图像第col列的非白色像素点
    :param img: cv2图片类型
    :param y:
    :param up:
    :param down:
    :param dic:
    :return:
    """
    white = [255, 255, 255]
    flag = False
    base = 5

    begin = 0
    end = 0

    for x in range(up + base, down - base):
        if list(img[x][y]) != white:
            if begin == 0:
                begin = x
                end = x
                flag = True
            else:
                end = x
    if flag == False:
        for x in range(up, up + base):
            if list(img[x][y]) != white:
                if begin == 0:
                    begin = x
                    end = x
                    flag = True
                else:
                    end = x
    if flag == False:
        for x in range(down - base, down):
            if list(img[x][y]) != white:
                if begin == 0:
                    begin = x
                    end = x
                    flag = True
                else:
                    end = x

    if end == 0:
        print('error')
    mid = (begin + end) >> 1
    changeColor(img, mid, y, dic)
    cv2.imshow('one', img)
    return (down - mid) / (down - up)


def GetAns(drawLinePath, borderPath, dbPath, drawMarkPath, ansPath, drawMarkPath2):
    """
    获取线上点结果的函数
    :param localPath: 图片地址
    :return:
    """
    print("正在计算结果...")
    img = cv2.imread(drawLinePath)
    dic = {}
    with open(borderPath) as f:
        for line in f:
            tmp = line.split()
            name = tmp[0]
            pos = [int(tmp[1]), int(tmp[2])]
            dic[name] = pos

    offSet = 3
    l = dic.get('downLeft')[1]
    r = dic.get('downRight')[1]
    up = dic.get('upLeft')[0]
    down = dic.get('downLeft')[0]
    length = r - l

    pos = [0.0, 0.2, 0.4, 0.6, 0.8]
    for i in range(len(pos)):
        pos[i] = round(l + length * pos[i])
    pos[0] += offSet

    maxi = 0
    with open(dbPath) as file:
        # 遍历文件中的每一行
        for line in file:
            maxi = int(line)
            break

    ans = []
    for y in pos:
        precent = fixBlack(img, y, up, down, dic)
        ans.append(round(maxi * precent, 6))

    cv2.imwrite(drawMarkPath, img)
    cv2.imwrite(drawMarkPath2, img)

    f = open(ansPath, 'w')
    print(' '.join(str(item) for item in ans), file=f)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
