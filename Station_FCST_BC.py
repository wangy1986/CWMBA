#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : Station_FCST_BC.py
@TIME      : 2024/03/28 09:57:19
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 本文件用于进行 forecast bias correction
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)



from copy import deepcopy 
from datetime import datetime, timedelta
import numpy as np 
import meteva.base as meb 
import os 
import pandas as pd 


from station_fcst_err_analysis import err_cal_mae, t2m_verification

def prepare_t2m_bc(start_date_utc, end_date_utc, fstH, fn_t2m, fn_obs, fn_bias): 
    """
    以滚动更新的形式，计算网格t2m 的 bias
    """
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = start_date_utc

    while idate_utc <= end_date_utc: 
        # load forecast data
        ifn_t2m = fn_t2m.format(t=idate_utc, fh=fstH)
        ifn_obs = fn_obs.format(t=idate_utc+timedelta(hours=fstH+8))

        t2m_g = meb.read_griddata_from_nc(ifn_t2m)
        iobs = meb.read_stadata_from_micaps3(ifn_obs, station)
        
        if (t2m_g is None) or (iobs is None): 
            idate_utc += timedelta(days=1)
            continue

        t2m_s = meb.interp_gs_linear(t2m_g, iobs)

        # 获取当前站点的预报误差
        # bias = obs - fcst
        t2m_s['bias'] = iobs['data0'] - t2m_s['data0']
        columns = list(t2m_s.columns)
        columns.remove('data0')
        t2m_s = t2m_s.loc[:, columns]

        # try to find history bc files
        for iseek_day in range(1, 7, 1): 
            jdate_utc = idate_utc - timedelta(days=iseek_day)
            ifn_bias_old = fn_bias.format(t=jdate_utc, fh=fstH)
            if not os.access(ifn_bias_old, os.R_OK): 
                continue
            else: 
                break
        
        ifn_bias_new = fn_bias.format(t=idate_utc, fh=fstH)
        if not os.access(ifn_bias_old, os.R_OK): 
            meb.write_stadata_to_micaps3(t2m_s, ifn_bias_new, True)
    
        else: 
            bias = meb.read_stadata_from_micaps3(ifn_bias_old)
            bias['today_bias'] = t2m_s['bias']
            bias['new_bias'] = 0
            def cal_bias(irec): 
                if np.isnan(irec['today_bias']): 
                    return irec['data0']
                elif np.isnan(irec['data0']): 
                    return irec['today_bias']
                else: 
                    return 0.95*irec['data0'] + 0.05*irec['today_bias']
            
            a = bias.apply(cal_bias, axis=1)
            bias['new_bias'] = a
            columns = list(bias.columns)
            columns.remove('data0')
            columns.remove('today_bias')
            bias = bias.loc[:, columns]
            meb.write_stadata_to_micaps3(bias, ifn_bias_new, True)
             

        idate_utc += timedelta(hours=24)

# end of prepare_spd_bc


def do_t2m_bc(start_date_utc, end_date_utc, fstH, fn_t2m, fn_bias, fn_t2m_s, fn_t2m_bced):
    """
    """
    idate_utc = start_date_utc
    station = meb.read_station(meb.station_国家站)

    while idate_utc <= end_date_utc: 
        # load forecast data
        ifn_t2m = fn_t2m.format(t=idate_utc, fh=fstH)
        # 获取前一日的bias
        idate_bias = idate_utc - timedelta(hours=int(np.ceil(ifstH/24))*24)
        ifn_bias = fn_bias.format(t=idate_bias, fh=fstH)

        t2m_g = meb.read_griddata_from_nc(ifn_t2m)
        bias = meb.read_stadata_from_micaps3(ifn_bias)
        
        if (t2m_g is None) or (bias is None): 
            idate_utc += timedelta(days=1)
            continue

        t2m_s = meb.interp_gs_linear(t2m_g, station)
        # forecast + bias 
        t2m_s['bced'] = t2m_s['data0'] + bias['data0']
        columns1 = list(t2m_s.columns)
        columns1.remove('data0')
        columns2 = list(t2m_s.columns)
        columns2.remove('bced')
        t2m_bced = t2m_s.loc[:, columns1]
        t2m_raw = t2m_s.loc[:, columns2]

        ifn_t2m_bced = fn_t2m_bced.format(t=idate_utc, fh=fstH)
        ifn_t2m_s = fn_t2m_s.format(t=idate_utc, fh=fstH)
        meb.write_stadata_to_micaps3(t2m_raw, ifn_t2m_s, True)
        meb.write_stadata_to_micaps3(t2m_bced, ifn_t2m_bced, True)

        idate_utc += timedelta(days=1)


def main_proc(d1_utc, d2_utc, ifstH, result_path): 
    """
    主程序流程
    """

    # 原始预报数据 & 实况数据
    fn_t2m_ecmwf = 'z:/ECMWF/T2M/{t:%Y%m%d%H/%Y%m%d%H}.{fh:03d}.nc'
    fn_t2m_ncep = 'z:/NCEP_GFS/T2M/{t:%Y%m%d%H/%Y%m%d%H}.{fh:03d}.nc'
    fn_t2m_CMA = 'z:/GRAPES_GFS/T2M/{t:%Y%m%d%H/%Y%m%d%H}.{fh:03d}.nc'
    fn_t2m_jp = 'z:/JAPAN_HR/T2M/{t:%Y%m%d%H/%Y%m%d%H}.{fh:03d}.nc'
    fn_obs = 'z:/YLRC_STATION/TEMP/rt0/{t:%Y/%Y%m%d%H}.000'
    
    # bias data 存储位置
    fn_bias_ecmwf = './raw_data/ECMWF/{t:%Y%m%d%H/SBIAS_%Y%m%d%H}.{fh:03d}.m3'
    fn_bias_ncep = './raw_data/ncep/{t:%Y%m%d%H/SBIAS_%Y%m%d%H}.{fh:03d}.m3'
    fn_bias_CMA = './raw_data/CMA/{t:%Y%m%d%H/SBIAS_%Y%m%d%H}.{fh:03d}.m3'
    fn_bias_jp = './raw_data/jp/{t:%Y%m%d%H/SBIAS_%Y%m%d%H}.{fh:03d}.m3'

    # 模式 DMO 站点预报
    fn_t2m_ecmwf_s = './raw_data/ECMWF/{t:%Y%m%d%H/SFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_ncep_s = './raw_data/ncep/{t:%Y%m%d%H/SFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_CMA_s = './raw_data/CMA/{t:%Y%m%d%H/SFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_jp_s = './raw_data/jp/{t:%Y%m%d%H/SFCST_%Y%m%d%H}.{fh:03d}.m3'

    # 经过偏差订正的 站点预报
    fn_t2m_ecmwf_bced_s = './raw_data/ECMWF/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_ncep_bced_s = './raw_data/ncep/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_CMA_bced_s = './raw_data/CMA/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_jp_bced_s = './raw_data/jp/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'

    # 站点预报的误差数据
    fn_err_t2m_ecmwf_s = result_path + '/DMO_fcst_err_ecmwf_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_t2m_ecmwf_bced_s = result_path + '/BC_fcst_err_ecmwf_{fh:03d}.csv'.format(fh=ifstH)

    fn_err_t2m_ncep_s = result_path + '/DMO_fcst_err_ncep_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_t2m_ncep_bced_s = result_path + '/BC_fcst_err_ncep_{fh:03d}.csv'.format(fh=ifstH)
    
    fn_err_t2m_CMA_s = result_path + '/DMO_fcst_err_CMA_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_t2m_CMA_bced_s = result_path + '/BC_fcst_err_CMA_{fh:03d}.csv'.format(fh=ifstH)

    fn_err_t2m_jp_s = result_path + '/DMO_fcst_err_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_t2m_jp_bced_s = result_path + '/BC_fcst_err_jp_{fh:03d}.csv'.format(fh=ifstH)

    #prepare_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf, fn_obs, fn_bias_ecmwf)
    #prepare_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_ncep, fn_obs, fn_bias_ncep)
    #prepare_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_CMA, fn_obs, fn_bias_CMA)
    prepare_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_jp, fn_obs, fn_bias_jp)

    #do_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf, fn_bias_ecmwf, fn_t2m_ecmwf_s, fn_t2m_ecmwf_bced_s)
    #do_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_ncep, fn_bias_ncep, fn_t2m_ncep_s, fn_t2m_ncep_bced_s)
    #do_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_CMA, fn_bias_CMA, fn_t2m_CMA_s, fn_t2m_CMA_bced_s)
    do_t2m_bc(d1_utc, d2_utc, ifstH, fn_t2m_CMA, fn_bias_jp, fn_t2m_jp_s, fn_t2m_jp_bced_s)

    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf_s, fn_obs, fn_err_t2m_ecmwf_s)
    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf_bced_s, fn_obs, fn_err_t2m_ecmwf_bced_s)
    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_ncep_s, fn_obs, fn_err_t2m_ncep_s)
    t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_jp_s, fn_obs, fn_err_t2m_jp_s)
    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_ncep_bced_s, fn_obs, fn_err_t2m_ncep_bced_s)
    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_CMA_s, fn_obs, fn_err_t2m_CMA_s)
    #t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_CMA_bced_s, fn_obs, fn_err_t2m_CMA_bced_s)
    t2m_verification(d1_utc, d2_utc, ifstH, fn_t2m_jp_bced_s, fn_obs, fn_err_t2m_jp_bced_s)


    '''
    fns = [fn_err_t2m_ecmwf_s, fn_err_t2m_ecmwf_bced_s, 
           fn_err_t2m_ncep_s, fn_err_t2m_ncep_bced_s, 
           fn_err_t2m_CMA_s, fn_err_t2m_CMA_bced_s ]
    descriptions = ['ECMWF Raw Fcst', 'ECMWF BCed Fcst', 
                    'NCEP Raw Fcst', 'NCEP BCed Fcst', 
                    'CMA-GFS Raw Fcst', 'CMA-GFS BCed Fcst']
    err_cal_mae(fns, descriptions) 
    '''


if __name__ == "__main__": 
    
    fstHs = list(range(3, 73, 3))
    
    d1_utc = datetime(2022, 3, 1, 0)
    d2_utc = datetime(2023, 3, 1, 0)
    for ifstH in fstHs:
        main_proc(d1_utc, d2_utc, ifstH, './result_train')
        pass
    
    d1_utc = datetime(2023, 3, 1, 0)
    d2_utc = datetime(2024, 3, 1, 0)
    
    for ifstH in fstHs:
        main_proc(d1_utc, d2_utc, ifstH, './result_test')
        pass
    
    