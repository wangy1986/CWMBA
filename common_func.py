# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np
import os, sys, time
from datetime import datetime, timedelta



# 非闰年/闰年 的日数 (每月月底，从1月1日计数)
cumulative_days_year_NL = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
cumulative_days_year_L = np.array([0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
LST_OFFSET_H = 8


def is_leap_year(year):
    '''
    check if the iYear is a leap year
    :param year: the year, like 2009, 2010, etc.
    :return: true,  if iYear is leap year
             false, if iYear is not leap year
    '''
    if (year % 4) == 0 and (year % 100) != 0 or (year % 400) == 0:
        return True
    else:
        return False


def genDaysEachMonth(iYear):
    '''
    :param iYear: the year, like 2009
    :return:    a List
                if is leap year,  return [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                if not leap year, return [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    '''
    if is_leap_year(iYear):
        return [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        return [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    

def getDays_aMonth(year, month): 
    return genDaysEachMonth(year)[month]
# end of getDays_aMonth


def convert_yyyymmdd_to_yyyyddd(year, month, day):
    if is_leap_year(year):
        ddd = cumulative_days_year_L[month]
        return int(year), int(ddd+day)
    else:
        ddd = cumulative_days_year_NL[month]
        return int(year), int(ddd+day)
    

def convert_yyyyddd_to_yyyymmdd(year, days):
    assert days > 0
    if is_leap_year(year):
        day_list = days - cumulative_days_year_L
        for imon in range(12, 0, -1):
            if day_list[imon] > 0:
                return year, imon, day_list[imon]
    else:
        day_list = days - cumulative_days_year_NL
        for imon in range(12, 0, -1):
            if day_list[imon] > 0:
                return year, imon, day_list[imon]


def convert_yyyyddd_to_ddddd_since_1900(year, days):
    ddddd = 0
    for iyear in range(1900, year):
        if is_leap_year(iyear):
            ddddd = ddddd + 366
        else:
            ddddd = ddddd + 365
    return ddddd + days


def convert_ddddd_since_1900_to_yyyyddd(ddddd):
    year = 1900
    while ddddd > 366:
        if is_leap_year(year):
            ddddd = ddddd - 366
            year = year + 1
        else:
            ddddd = ddddd - 365
            year = year + 1

    # 处理剩余366天的情况，需要分 平年 和闰年
    if ddddd < 366:
        return year, ddddd
    else:
        if is_leap_year(year):
            return year, ddddd
        else:
            return year+1, ddddd-365


def convert_ddddd_since_1900_to_yyyymmdd(ddddd):
    yyyy, ddd = convert_ddddd_since_1900_to_yyyyddd(ddddd)
    return convert_yyyyddd_to_yyyymmdd(yyyy, ddd)


def convert_yyyymmdd_to_ddddd_since_1900(yyyy, mm, dd):
    yyyy, ddd = convert_yyyymmdd_to_yyyyddd(yyyy, mm, dd)
    return convert_yyyyddd_to_ddddd_since_1900(yyyy, ddd)
    

def output_log_record(fn_log, infor_str, b_screen_print=True): 
    """
    向 log 文件输出日志

    --------------------------
    Parameters
    --------------------------
    fn_log: 日志文件文件名, 如果为 None，则不输出信息

    infor_str: 日志信息，本函数会自动在信息头部添加时间信息

    b_screen_print: 是否在屏幕上打印该日志信息

    --------------------------
    Returns
    --------------------------
    None

    """
    tt = datetime.now()
    output_str = '%4d%02d%02d %02d:%02d:%02d [%s]' % (tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second, infor_str)
    
    if b_screen_print:
        print(output_str)
    
    if fn_log is None: 
        return

    with open(fn_log, 'a') as f:
        if output_str[-1] == '\n':
            f.write(output_str)
        else: 
            f.write('%s\n' % output_str)
# end of output_log_record


# 已知一组预报时间，[ year, mon, day, start_h, fst_h ]
# 将之转换为观测数据的时间
#           [ year, mon, day, hour ]
def convert_fst_time_to_obs_time(year, mon, day, start_h, fst_h, b_fst_utc, b_obs_utc):
    
    # the total-fst-hour since 1900-01-01
    total_hours = convert_yyyymmdd_to_ddddd_since_1900(year, mon, day) * 24 + start_h + fst_h
    # 1. trans total-fst-hour to UTC format
    total_hours += 0 if (b_fst_utc) else -8
    # 2. trans total-fst-hour to the observation required time format
    total_hours += 0 if (b_obs_utc) else 8
    
    obs_hour = total_hours % 24
    obs_year, obs_mon, obs_day = convert_ddddd_since_1900_to_yyyymmdd(int(total_hours/24))
    
    return obs_year, obs_mon, obs_day, obs_hour
# end of convert_fst_time_to_obs_time


def convert_fst_time_to_obs_time_tlist(tlist, start_h, fst_h, b_fst_utc, b_obs_utc):
    """
    将预报时间转换为实况时间

    --------------
    parameters
    --------------
    tlist:          [trange1, trange2, ...]
                    itrange = [start_y, start_m, start_d, end_y, end_m, end_d]

    start_h:        起报时间
    fst_h:          预报时效

    b_fst_utc:      上面给出的时报时间是否为UTC
    b_obs_utc:      待转换的实况时间是否为UTC

    --------------
    returns
    --------------
    tlist_obs, obs_h        
                    tlist_obs = [trange1, trange2, ...], 与 tlist 对应的实况时间段
                        itrange = [start_y, start_m, start_d, end_y, end_m, end_d]
                    
                    obs_h:      实况观测时间
    """

    tlist_obs = []
    for itr in tlist:
        obs_y1, obs_m1, obs_d1, obs_h = convert_fst_time_to_obs_time(itr[0], itr[1], itr[2], start_h, fst_h, b_fst_utc, b_obs_utc)
        obs_y2, obs_m2, obs_d2, obs_h = convert_fst_time_to_obs_time(itr[3], itr[4], itr[5], start_h, fst_h, b_fst_utc, b_obs_utc)
        tlist_obs.append([obs_y1, obs_m1, obs_d1, obs_y2, obs_m2, obs_d2])

    return tlist_obs, obs_h
# end of convert_fst_time_to_obs_time_tlist


# 进行 UTC -> LST 的转换
# 其中 utc_tm = datetime(year, mon, day, hour, minute, second)
def utc2lst(utc_tm):
    return utc_tm + timedelta(0, LST_OFFSET_H*3600)


# 进行 LST -> UTC 的转换
# 其中 lst_tm = datetime(year, mon, day, hour, minute, second)
def lst2utc(lst_tm):
    return lst_tm + timedelta(0, LST_OFFSET_H*(-3600))
	

# 计算地图上两点经纬度间的距离
# https://blog.csdn.net/baidu_32923815/article/details/79719813 
from math import radians, cos, sin, asin, sqrt  
#from numba import jit
#@jit(nopython=True, cache=True)
def Haversine_KM(lon1, lat1, lon2, lat2): 
    # 将十进制度数转化为弧度  
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    lon1 = lon1 * 0.01745329
    lat1 = lat1 * 0.01745329
    lon2 = lon2 * 0.01745329
    lat2 = lat2 * 0.01745329
    # Haversine公式  
    # a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2  
    # 地球平均半径: 6370.856 km
    return 2 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2)) * 6370.856
# end of Haversine_KM


def trans_tlist_to_dayslist(tlist): 
    """
    将 tlist = [[start_y, start_m, start_d, end_y, end_m, end_d], ...0 ]
    转换为 daylist = [start_ddddd, ... ], [end_dddddd, ... ]

    返回值为：
        n_total_days, [start_ddddd1, start_ddddd2, ...], [end_ddddd1, end_ddddd2, ...]
    """

    n_total_days = 0
    d1_list = []
    d2_list = []
    
    for i in range(len(tlist)): 
        d1 = convert_yyyymmdd_to_ddddd_since_1900(tlist[i][0], tlist[i][1], tlist[i][2])
        d2 = convert_yyyymmdd_to_ddddd_since_1900(tlist[i][3], tlist[i][4], tlist[i][5])
        d1_list.append(d1)
        d2_list.append(d2)
        n_total_days += (d2 - d1 + 1)

    return n_total_days, d1_list, d2_list
# end of trans_tlist_to_dayslist


def get_dict_parameters(para_dict, key_list):
    """
    用于提取 para_dict 之中的参数

    ----------------
    Parameters
    ----------------
    para_dict:  参数字典

    key_list:   待检验的关键词, = [key_name1, key_name2, ..., key_nameN]

    ----------------
    Returns
    ----------------
    value_list： 对应 key 的键值，但如果对应key不存在，则其值以 None 表示
                    = [value1, value2, ..., valueN]

    """
    
    value_list = []
    for ikey in key_list:
        if ikey in para_dict: 
            value_list.append(para_dict[ikey])
        else:
            print('para_dict中缺少必要参数 %s' % ikey)
            value_list.append(None)
    
    return value_list
# end of check_dict_parameters


def check_fn_available(fn):
    """
    检测 fn 是否可行，即查看存储文件fn时，所需的路径是否已经创建
    如果路径不存在，则创建该路径
    """
    path = fn[0:max(fn.rfind('/'), fn.rfind('\\'))]
    check_path_exist_else_mkdir(path, '', None)
# end of check_fn_available


def check_path_exist_else_mkdir(path, path_name_str, log_fn):
    """
    判断路径是否存在，如果路径不存在，则创建该路径

    -------------------
    Parameters
    -------------------
    path: 目标路径 (str/list/dict)
    path_name_str: 路径的名称

    log_fn: 日志文件名

    -------------------
    Returns
    -------------------
    None

    """

    if type(path) is dict: 
        for k, v in path.items(): 
            if os.path.exists(v) is False:
                log_str = '创建路径 %s-%s: %s' % (path_name_str, k, v)
                output_log_record(log_fn, log_str)
                os.makedirs(v)
    elif type(path) is list: 
        for ipath in path:
            if os.path.exists(ipath) is False:
                log_str = '创建路径 %s: %s' % (path_name_str, path)
                output_log_record(log_fn, log_str)
                os.makedirs(ipath)
    else:
        if os.path.exists(path) is False:
            log_str = '创建路径 %s: %s' % (path_name_str, path)
            output_log_record(log_fn, log_str)
            os.makedirs(path)

# end of check_path_exist_else_mkdir


def check_path_exist_else_exit(path, path_name_str, log_fn): 
    """
    判断路径是否存在，如果路径不存在，则程序退出！

    -------------------
    Parameters
    -------------------
    path: 目标路径 (str/list/dict)
    path_name_str: 路径的名称

    -------------------
    Returns
    -------------------
    None

    """
    if type(path) is dict: 
        for k, v in path.items(): 
            if os.path.exists(v) is False:
                err_str = 'path %s[%s]: [%s] 不存在！程序退出' % (path_name_str, k, v)
                output_log_record(log_fn, err_str)
                sys.exit(1)
    elif type(path) is list: 
        for ipath in path:
            if os.path.exists(ipath) is False:
                err_str = 'path %s: %s 不存在！程序退出' % (path_name_str, ipath)
                output_log_record(log_fn, err_str)
                sys.exit(1)
    else:
        if os.path.exists(path) is False:
            err_str = 'path %s: [%s] 不存在！程序退出' % (path_name_str, path)
            output_log_record(log_fn, err_str)
            sys.exit(1)
# end of check_path_exist_else_exit


import pickle
def save_obj(obj, name, to_path): 
    """
    存储 object, 至位置 to_path, save as file name: 'name.pkl'
    as wb format

    ---------------------
    Parameters
    ---------------------
    obj:    待存储的对象
    name:   obj的名字，同时也是存储的文件名(不带 .pkl 后缀)

    to_path:    存储路径
    
    ---------------------
    Returns
    ---------------------
    None

    """
    fn = '%s/%s.pkl' % (to_path, name)
    check_fn_available(fn)
    with open(fn, 'wb') as f: 
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# end of save_obj


def load_obj(name, from_path):
    """
    从路径 from_path 读取 object

    ---------------------
    Parameters
    ---------------------
    name:   obj的名字，同时也是存储的文件名(不带 .pkl 后缀)

    from_path:    obj file 存储路径
    
    ---------------------
    Returns
    ---------------------
    None/obj

    None: 如果读取失败，例如文件损坏或不存在, return None
    obj:  读取成功

    """

    try: 
        with open('%s/%s.pkl' % (from_path, name), 'rb') as f: 
            return pickle.load(f)
    except: 
        print('读取 obj: %s 失败，返回None' % name)
        return None

# end of load_obj


def convert_npy_2_npz(dir, b_del_npy_files, b_sub_dir):
    """
    本函数负责将文件夹内所有 npy 文件转换为 npz 文件
    npy为非压缩数组
    npz为压缩数组

    ----------------------
    Parameters
    ----------------------
    dir:                带转换的目录
    b_del_npy_files:    是否删除旧的 npy 文件
    b_loop_dir:         是否对文件夹的子文件夹进行同样操作

    ----------------------
    Returns
    ----------------------
    None
    """

    for root, dirs, files in os.walk(dir): 

        for f1 in files: 
            # 是 .npy 文件
            if f1[-4:] == '.npy': 
                f11 = '%s/%s' % (dir, f1)
                f2 = '%s/%s.npz' % (dir, f1[:-4])
                
                print('convert [%s] -> [%s]' % (f11, f2))
                np.savez_compressed(f2, np.load(f11))

                if b_del_npy_files: 
                    os.remove(f11)
        # end of loop files

        # 递归调用本函数处理子文件夹
        if b_sub_dir: 
            for idir in dirs: 
                convert_npy_2_npz(idir, b_del_npy_files, b_sub_dir)
        # end of process sub dir
    # end of loop files, dirs in given dir

# end of convert_npy_2_npz


def get_now_time_str():
    """
    获取字符串格式的当前时间
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def datatype_2_str(istruct, 
                   nblanks:    int=0): 
    """
    将字典格式转换为字符串
    每一个val, key 以 换行相隔

    ----------------
    parmaeters
    ----------------
    idict:      目标数据类型

    nblanks:    行前空格数

    ----------------
    returns
    ----------------

    """

    cur_sp = ' '*nblanks
    cur_sub_sp = ' '*(nblanks+4)

    str_mid = ''
    if type(istruct) is dict: 
        istr_list = []
        mid_str = ''
        for ikey ,ival in istruct.items():
            istr_list.append(cur_sub_sp+str(ikey) + ': ' + datatype_2_str(ival, nblanks+4))
            mid_str = ('\n').join(istr_list)
        return cur_sp + '{\n' + mid_str + cur_sp + '}\n'

    elif type(istruct) is list: 
        tmp_str = ''
        for ival in istruct: 
            tmp_str += (datatype_2_str(ival, nblanks+4) + ' ,')
        return '[\n' + cur_sub_sp + tmp_str  + '\n' + cur_sub_sp+']'
    elif type(istruct) is tuple: 
        tmp_str = ''
        for ival in istruct: 
            tmp_str += (datatype_2_str(ival, nblanks+4) + ' ,')
        return '(' + tmp_str + ')'
    elif type(istruct) is set: 
        tmp_str = ''
        for ival in istruct: 
            tmp_str += (datatype_2_str(ival, nblanks+4) + ' ,')
        return '{' + tmp_str + '}'
    else: 
        str_mid = str(istruct)
        return str_mid
# end of dict_2_str


def Gaussian_1D(x, u=0, sigma=1):
    """
    获取 1维 高斯分布
    f(x) = 1/[√(2π)*sigma] * exp(- (x-u)^2/ 2sigma^2)
    
    一个数学期望为μ、方差为σ^2 的正态分布，记为N(μ，σ^2)
    当μ = 0,σ = 1时的正态分布是标准正态分布

    ---------------------
    parameters
    ---------------------
    x:          x valuse, 1D -array

    u:          期望值μ决定了其位置
    sigma:      准差σ决定了分布的幅度
    ---------------------
    ---------------------
    """

    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-np.power((x-u), 2) / (2*sigma*sigma))

# end of  


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    box_idx_wst = 100
    box_idx_est = 170
    box_idx_sou = 100
    box_idx_nor = 200
    cx = 150
    cy = 150
    x_x = np.arange(box_idx_wst, box_idx_est+1, 1)
    x_y = np.arange(box_idx_sou, box_idx_nor+1, 1)
    sigma = max(cx-box_idx_wst, box_idx_est-cx)
    g_x = Gaussian_1D(x_x, cx, sigma).reshape(1, -1)
    g_y = Gaussian_1D(x_y, cy, sigma).reshape(-1, 1)
    mat = np.matmul(g_y, g_x)
    mat /= np.max(mat)
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
    

