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
            if color != [255, 255, 255]:
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
            if color != [255, 255, 255]:
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


def CreateBorderDic(img, lsPosRow, lsPosCol):
    """
    判断网格是否需要造框，需要则返回True
    :param img: opencv打开的图片对象
    :param lsPosRow: 框线行数组
    :param lsPosCol: 框线列数组
    :return: 是否需要造框，需要则返回True
    """
    dicColor = {}
    for col in lsPosCol:
        for row in img.shape[0]:
            if img[row][col].tolist() == img[0][0].tolist():
                continue
            if str(img[row][col]) in dicColor:
                dicColor[str(img[row][col])] += 1
            else:
                dicColor[str(img[row][col])] = 1
    lineColor = sorted(dicColor.items(), key=lambda x: x[1], reverse=True)[0][0]

    # 先同各国lsPosCol求up和down
    up = 0
    while str(img[up][lsPosCol[0]]) != lineColor:
        up += 1
    lineColor = img[up + 3][lsPosCol[-1]]
    down = img.shape[0] - 1
    while str(img[down][lsPosCol[0]]) != lineColor:
        down -= 1
    # 先同各国lsPosRow求left和right
    left = 0
    while str(img[lsPosRow[0]][left]) != lineColor:
        left += 1
    right = img.shape[1] - 1
    while str(img[lsPosRow[0]][right]) != lineColor:
        right -= 1

    return {'up': up, 'down': down, 'left': left, 'right': right}


def GetBorderDic(imgPath):
    """
    获取边框点位
    :param imgPath: 原图片地址
    :return: 边框点位字典
    """
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
            if color.tolist() != img[0][0].tolist():
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
            if color.tolist() != img[0][0].tolist():
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
    if len(lsPosRow) > 15: # 有底色的情况
        dic['up'] = lsPosRow[0]
        dic['down'] = lsPosRow[len(lsPosRow) - 1]
        dic['left'] = lsPosCol[0]
        dic['right'] = lsPosCol[len(lsPosCol) - 1]
    else: # 普通框线的情况
        # 判断是否要造框
        createBorder = CreateBorderDic(img, lsPosRow, lsPosCol)
        # print(createBorder['up'], lsPosRow[0])
        if abs(createBorder['up'] - lsPosRow[0]) > 10: # 造的框和简单找框的差距较远，说明框确实要造
            return createBorder

        # 通过lsPosRow计算上下
        i = 0
        while lsPosRow[i + 1] == lsPosRow[i] + 1:
            i += 1
        j = len(lsPosRow) - 1
        while lsPosRow[j - 1] == lsPosRow[j] - 1:
            j -= 1
        dic['up'] = lsPosRow[i]
        dic['down'] = lsPosRow[j]
        # 通过lsPosCol计算左右
        i = 0
        while lsPosCol[i + 1] == lsPosCol[i] + 1:
            i += 1
        j = len(lsPosCol) - 1
        while lsPosCol[j - 1] == lsPosCol[j] - 1:
            j -= 1
        dic['left'] = lsPosCol[i]
        dic['right'] = lsPosCol[j]

    return dic


def CutBorder(imgPath, outputPath):
    img = cv2.imread(imgPath)
    dic = GetBorderDic(imgPath)
    bgc = img[0][0].tolist()
    img = img[dic['up'] + 1: dic['down'] - 1, dic['left'] + 1: dic['right'] - 1]
    row, col, type = img.shape
    for i in range(row):
        for j in range(col):
            if img[i][j].tolist() == bgc:
                img[i][j] = img[0][0]

    cv2.imwrite(outputPath, img)


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


def TurnBinary(inputPath, outputPath):
    """
    根据cv2的二值化函数进行二值化操作
    :param inputPath: 输入图片路径
    :param outputPath: 输出图片路径
    :return: 无
    """
    img = cv2.imread(inputPath)
    # print(outputPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(outputPath, binary)
    # cv2.imshow('bin', binary)


def TurnBlackBackground(inputPath, outputPath):
    img = cv2.imread(inputPath)
    white = 0
    for i in range(img.shape[0]):  # row
        for j in range(img.shape[1]):  # col
            if img[i][j].tolist() == [255, 255, 255]:
                white += 1
    black = img.shape[0] * img.shape[1] - white
    if black > white:
        cv2.imwrite(outputPath, img)
    else:
        cv2.imwrite(outputPath, 255 - img)


def DeleteLine(imgPath, outPath):
    binary = cv2.imread(imgPath)
    row, col, type = binary.shape
    # cv2.imshow("binary", binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("binary", binary)

    lsRow = []
    for i in range(row):  # row
        cnt = 0
        for j in range(col):  # col
            if binary[i][j].tolist() == [255, 255, 255]:
                cnt += 1
        if cnt > row // 4 * 3:
            lsRow.append(i)

    lsCol = []
    for j in range(col):  # row
        cnt = 0
        for i in range(row):  # col
            if binary[i][j].tolist() == [255, 255, 255]:
                cnt += 1
        # print(cnt, col)
        if cnt > col // 2:
            lsCol.append(j)

    if len(lsRow) == 0 and len(lsCol) == 0:
        pass
    else:
        for pos in lsRow:
            for item in range(col):
                binary[pos][item] = [0, 0, 0]
        for pos in lsCol:
            for item in range(row):
                binary[item][pos] = [0, 0, 0]
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("bin", binary)
    # cv2.waitKey(0)
    cv2.imwrite(outPath, binary)


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
                for item in range(-2, 3):
                    if [b + item, g + item, r + item] in ls:
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
    # cv2.imshow('one', img)
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

    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i in range(len(pos)):
        pos[i] = round(l + length * pos[i])
    pos[0] += offSet
    pos[-1] -= offSet

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


def ChangeColor(img, x, y, up, down, l, r):
    """
    标记点
    :param img:
    :param x:
    :param y:
    :param up:
    :param down:
    :param l:
    :param r:
    :return:
    """
    if x == None or y == None:
        return
    base = 15
    green = [0, 255, 0]
    yellow = [0, 255, 255]
    for i in range(max(x - base, up), min(x + base, down)):
        img[i][y] = green

    for j in range(max(y - base, l), min(y + base, r)):
        img[x][j] = yellow


def targetPixel(img, y, up, down, l, r):
    """
    找到图像第col列的黑色像素点
    :param img:
    :param y:
    :param up:
    :param down:
    :param l:
    :param r:
    :return:
    """
    if y < l or y > r:
        return None
    white = [255, 255, 255]
    flag = False
    base = 5

    begin = 0
    end = 0

    for x in range(up + base, down - base):
        if list(img[x][y]) == white:
            if begin == 0:
                begin = x
                end = x
                flag = True
            else:
                end = x
    if flag == False:
        for x in range(up, up + base):
            if list(img[x][y]) == white:
                if begin == 0:
                    begin = x
                    end = x
                    flag = True
                else:
                    end = x
    if flag == False:
        for x in range(down - base, down):
            if list(img[x][y]) == white:
                if begin == 0:
                    begin = x
                    end = x
                    flag = True
                else:
                    end = x

    if end == 0:
        # print('error:没找到点')
        return None
    mid = (begin + end) >> 1

    # return (down - mid) / (down - up)
    return mid


def shakeTargetPixel(img, y, up, down, l, r):
    """
    抖动寻找像素点，如果当前列不存在像素点，左右寻找找mid
    :param img:
    :param y:
    :param up:
    :param down:
    :param l:
    :param r:
    :return:
    """
    pixel = targetPixel(img, y, up, down, l, r)
    if pixel != None:
        return pixel
    # print(1)
    for i in range(1, 100):
        lPixel = targetPixel(img, y - i, up, down, l, r)
        rPixel = targetPixel(img, y + i, up, down, l, r)
        if lPixel != None and rPixel != None:
            return round((lPixel + rPixel) / 2)
        elif lPixel != None:
            return lPixel
        elif rPixel != None:
            return rPixel
    return None


def GetBinaryAns(inputImgPath, outputImgPath, maxValuePath, ansPath):
    """
    计算二值化图的结果
    :param inputImgPath:
    :param outputImgPath:
    :param maxValuePath:
    :param ansPath:
    :return:
    """
    img = cv2.imread(inputImgPath)

    height = img.shape[0] - 1
    width = img.shape[1] - 1

    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i in range(len(pos)):
        pos[i] = round(width * pos[i])

    maxi = 0
    with open(maxValuePath) as file:
        # 遍历文件中的每一行
        for line in file:
            maxi = int(line)
            break

    ans = []
    for y in pos:
        x = shakeTargetPixel(img, y, 0, height, 0, width)
        if x != None:
            ChangeColor(img, x, y, 0, height, 0, width)
            precent = (height + 1 - x) / (height + 1)
            ans.append(round(maxi * precent, 6))

    cv2.imwrite(outputImgPath, img)

    f = open(ansPath, 'w')
    print(' '.join(str(item) for item in ans), file=f)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return
