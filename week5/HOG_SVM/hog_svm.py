import os
import sys
import cv2
import logging
import numpy as np

def logger_init():
    '''
    自定义python的日志信息打印配置
    :return logger: 日志信息打印模块
    '''

    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("PedestranDetect")

    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    # 文件日志
    # file_handler = logging.FileHandler("test.log")
    # file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter  # 也可以直接给formatter赋值

    # 为logger添加的日志处理器
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)

    return logger

def load_data_set(logger):
    '''
    导入数据集
    :param logger: 日志信息打印模块
    :return pos: 正样本文件名的列表
    :return neg: 负样本文件名的列表
    :return test: 测试数据集文件名的列表。
    '''
    logger.info('Checking data path!')
    pwd = os.getcwd()
    logger.info('Current path is:{}'.format(pwd))
    dataset_dir='./datasets'
    # 提取正样本
    pos_dir = os.path.join(dataset_dir, 'Positive')
    if os.path.exists(pos_dir):
        logger.info('Positive data path is:{}'.format(pos_dir))
        pos = os.listdir(pos_dir)
        logger.info('Positive samples number:{}'.format(len(pos)))

    # 提取负样本
    
    neg_dir = os.path.join(dataset_dir, 'Negative')
    if os.path.exists(neg_dir):
        logger.info('Negative data path is:{}'.format(neg_dir))
        neg = os.listdir(neg_dir)
        logger.info('Negative samples number:{}'.format(len(neg)))

    # 提取测试集
    test_dir = os.path.join(dataset_dir, 'TestData')
    if os.path.exists(test_dir):
        logger.info('Test data path is:{}'.format(test_dir))
        test = os.listdir(test_dir)
        logger.info('Test samples number:{}'.format(len(test)))

    return pos, neg, test

def load_train_samples(pos, neg):
    '''
    合并正样本pos和负样本pos，创建训练数据集和对应的标签集
    :param pos: 正样本文件名列表
    :param neg: 负样本文件名列表
    :return samples: 合并后的训练样本文件名列表
    :return labels: 对应训练样本的标签列表
    '''
    # pwd = os.getcwd()
    dataset_dir='./datasets'
    pos_dir = os.path.join(dataset_dir, 'Positive')
    neg_dir = os.path.join(dataset_dir, 'Negative')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # labels 要转换成numpy数组，类型为np.int32
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels
def extract_hog(samples, logger):
    '''
    从训练数据集中提取HOG特征，并返回
    :param samples: 训练数据集
    :param logger: 日志信息打印模块
    :return train: 从训练数据集中提取的HOG特征
    '''
    train = []
    logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    
    for f in samples:
        num += 1.
        # logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)#cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        # hog = cv2.HOGDescriptor()
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        # logger.info('hog feature descriptor size: {}'.format(descriptors.shape))    # (3780, 1)
        train.append(descriptors)
    logger.info('Extract finish')
    train = np.float32(train)
    train = np.resize(train, (total, 3780))
    #这里解释一下为什么特征向量的维度是3780
    #win_size=(64,128)
    #block_size=(16,16)
    #block_stride=(8,8)
    #cell_size=(8,8)
    #nbin=9代表在一个胞元中统计梯度的方向数目
    #那么一共有((64-16)/8+1)*((128-16)/8+1)=105个block，每个block有(16/8)*(16/8)=4个cell，每个cell统计9个梯度
    #所以特征维度是 105*4*9=3780


    return train
def get_svm_detector(svm):
    '''
    导出可以用于cv2.HOGDescriptor()的SVM检测器，实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表，可用作cv2.HOGDescriptor()的SVM检测器
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def train_svm(train, labels, logger):
    '''
    训练SVM分类器
    :param train: 训练数据集
    :param labels: 对应训练集的标签
    :param logger: 日志信息打印模块
    :return: SVM检测器（注意：opencv的hogdescriptor中的svm不能直接用opencv的svm模型，而是要导出对应格式的数组）
    '''
    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    logger.info('Training done.')

    # pwd = os.getcwd()
    # model_path = os.path.join(pwd, 'svm.xml')
    model_path='./svm.xml'
    svm.save(model_path)
    logger.info('Trained SVM classifier is saved as: {}'.format(model_path))

    return get_svm_detector(svm)


def calculate_IOU(rec1,rec2):
    '''
    计算两个矩形框的交并比
    :param rec1: 矩形框1 x y w h
    :param rec2: 矩形框2 x y w h
    :return:IOU
    '''
    #框1坐标(左上，右下)
    x1_tl=rec1[0]#左上角点的x坐标
    y1_tl=rec1[1]#左上角点的y坐标
    x1_br=rec1[0]+rec1[2]#右下角点的x坐标
    y1_br=rec1[1]+rec1[3]#右下角点的y坐标
    #框2坐标(左上，右下)
    x2_tl=rec2[0]#左上角点的x坐标
    y2_tl=rec2[1]#左上角点的y坐标
    x2_br=rec2[0]+rec2[2]#右下角点的x坐标
    y2_br=rec2[1]+rec2[3]#右下角点的y坐标
    #计算重叠区域
    x_overlap=max(0,min(x1_br,x2_br)-max(x1_tl,x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area=x_overlap*y_overlap
    area_1=rec1[2]*rec1[3]
    area_2=rec2[2]*rec2[3]
    total_area=area_1+area_2-overlap_area
    # if overlap_area==area_1 or overlap_area==area_2:#如果完全包含，也可滤除
    #     return 1
    return overlap_area/float(total_area)

def nms(detections,conf,conf_threshold=0.3,iou_threshold=0.5):
    '''
    非极大值抑制
    :param detections: 检测出来的矩形框
    :param conf: 每个框所对应的置信度值
    :param threshold:IOU阈值
    :return: 过滤后的框
    '''
    if len(detections)==0:
        return[]
    
    #首先将置信度列表归一化
    # print("befor sort")
    # print(detections)
    # print(conf)    
    #基于置信度对检测出来的框进行排序（此处打包操作对置信度列表也进行排序，方便根据置信度进行过滤）
    # detections=sorted(detections,key=lambda x:conf[detections.index(x)],reverse=True)
    zip_detections_conf=zip(detections,conf)
    sorted_zip=sorted(zip_detections_conf,key=lambda x:x[1],reverse=True)
    detections,conf=zip(*sorted_zip)
    detections=list(detections)
    conf=list(conf)
    # print("after sort")
    # print(detections)
    # print(conf)
    results=[]
    results.append(detections[0])
    for index,detection in enumerate(detections):
        # print("当前index",index)
        if conf[index]<conf_threshold:
            # print("置信度小而跳过的",index,conf[index],detections[index])
            continue
        for result in results:
            iou=calculate_IOU(detection,result)
            # print(iou)
            if iou>iou_threshold:#如果detection和result里面的框重叠过大，就跳过
                # print('删除的',index,detections[index])
                break
        else:
            results.append(detection)
            # print("保留的",index,detections[index])
    return results


def test_hog_detect(test, svm_detector, logger,result_dir,save=False):
    '''
    导入测试集，测试结果
    :param test: 测试数据集
    :param svm_detector: 用于HOGDescriptor的SVM检测器
    :param logger: 日志信息打印模块
    :return: 无
    '''
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(svm_detector)
    # opencv自带的训练好了的分类器
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    dataset_dir='./datasets'
    test_dir = os.path.join(dataset_dir, 'TestData')
    os.makedirs(result_dir,exist_ok=True)
    #单张图片展示
    img=cv2.imread('./datasets/TestData/000003.jpg')
    rects, conf = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.5)
    conf=np.array(conf)/max(conf)
    temp=img.copy()
    # print(rects)
    # print(conf)
    for index,(x,y,w,h) in enumerate(rects):
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        # cv2.putText(img,str(conf[index]),(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8)
        cv2.putText(img,"%.2f"%conf[index],(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8)
    cv2.imshow('without NMS',img)
    cv2.waitKey(0)
    results=nms(rects,conf,conf_threshold=0.3,iou_threshold=0.3)
    for (x,y,w,h) in results:
        cv2.rectangle(temp, (x,y), (x+w,y+h), (0,0,255), 2) 
    cv2.imshow('with NMS',temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #测试集检测结果保存
    if save:
        for f in test:
            file_path = os.path.join(test_dir, f)
            logger.info('Processing {}'.format(file_path))
            img = cv2.imread(file_path)
            # rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
            rects, conf = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.5)
            conf=np.array(conf)/max(conf)
            results=nms(rects,conf,conf_threshold=0.3,iou_threshold=0.3)
            for (x,y,w,h) in results:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.imwrite(os.path.join(result_dir,f),img)

if __name__=='__main__':
    # 初始化日志
    logger=logger_init()
    #加载数据集（pos 正样本，neg 负样本，test 测试集）
    pos, neg, test = load_data_set(logger=logger)
    #加载训练样例（正样本和负样本及其标签进行合并）
    samples, labels = load_train_samples(pos, neg)
    #提取训练集的HOG特征
    train = extract_hog(samples, logger=logger)
    logger.info('Size of feature vectors of samples: {}'.format(train.shape))
    logger.info('Size of labels of samples: {}'.format(labels.shape))
    svm_detector = train_svm(train, labels, logger=logger)
    result_dir='./results'
    test_hog_detect(test, svm_detector, logger,result_dir,save=True)



    # detections=[[268,   0, 216, 423],
    #             [408,  61, 144, 288],
    #             [313,  58, 144, 288],
    #             [150,  60, 144, 288],
    #             [ 50,  63, 144, 288],
    #             [580, 160,  60, 128]]
    # conf=[0.30635802, 1.15266028, 1.08438342, 0.39594683, 0.36392015, 0.18437328]
    # results=nms(detections,conf,threshold=0.5)
    # print(results)