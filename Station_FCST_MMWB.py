#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : Station_FCST_MMWB.py
@TIME      : 2024/03/28 10:19:03
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : implement MMWB (mulit model weighted blending)
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)


from datetime import datetime, timedelta
import numpy as np 
import os 
import pandas as pd 
import meteva.base as meb 
from copy import deepcopy

from station_fcst_err_analysis import err_cal_mae, t2m_verification


def prepare_t2m_mae(start_date_utc, end_date_utc, fstH, fn_t2m, fn_obs, fn_mae): 
    """
    以滚动更新的形式，计算网格t2m 的 mae 
    """
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = start_date_utc

    while idate_utc <= end_date_utc: 
        # load forecast data
        ifn_t2m = fn_t2m.format(t=idate_utc, fh=fstH)
        ifn_obs = fn_obs.format(t=idate_utc+timedelta(hours=fstH+8))

        t2m_s = meb.read_stadata_from_micaps3(ifn_t2m, station)
        iobs = meb.read_stadata_from_micaps3(ifn_obs, station)
        
        if (t2m_s is None) or (iobs is None): 
            idate_utc += timedelta(days=1)
            continue

        #t2m_s = meb.interp_gs_linear(t2m_g, iobs)

        # 获取当前站点的预报误差
        # bias = obs - fcst
        t2m_s['mae'] = np.abs(iobs['data0'] - t2m_s['data0'])
        columns = list(t2m_s.columns)
        columns.remove('data0')
        t2m_s = t2m_s.loc[:, columns]

        # try to find history bc files
        for iseek_day in range(1, 7, 1): 
            jdate_utc = idate_utc - timedelta(days=iseek_day)
            ifn_mae_old = fn_mae.format(t=jdate_utc, fh=fstH)
            if not os.access(ifn_mae_old, os.R_OK): 
                continue
            else: 
                break
        
        ifn_mae_new = fn_mae.format(t=idate_utc, fh=fstH)
        if not os.access(ifn_mae_old, os.R_OK): 
            meb.write_stadata_to_micaps3(t2m_s, ifn_mae_new, True)
    
        else: 
            mae = meb.read_stadata_from_micaps3(ifn_mae_old)
            mae['today_mae'] = t2m_s['mae']
            mae['new_mae'] = 0
            def cal_mae(irec): 
                if np.isnan(irec['today_mae']): 
                    return irec['data0']
                elif np.isnan(irec['data0']): 
                    return irec['today_mae']
                else: 
                    return 0.95*irec['data0'] + 0.05*irec['today_mae']
                
            a = mae.apply(cal_mae, axis=1)
            mae['new_mae'] = a
            columns = list(mae.columns)
            columns.remove('data0')
            columns.remove('today_mae')
            mae = mae.loc[:, columns]
            meb.write_stadata_to_micaps3(mae, ifn_mae_new, True)
             

        idate_utc += timedelta(hours=24)

# end of prepare_t2m_mae

def prepare_corrcoef(ifstH, fn_corrcoef): 
    """
    calculate the corrcoef from history data
    """

    ### set path start ###
    # these path should be the same as [de-biased forecast's error path] 
    #   in Station_FCST_BC.py
    # for example:  fn_err_ecmwf = fn_err_t2m_ecmwf_bced_s
    #               fn_err_ncep = fn_err_t2m_ncep_bced_s
    fn_err_ecmwf = './result_train/BC_fcst_err_ecmwf_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_ncep = './result_train/BC_fcst_err_ncep_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_cma = './result_train/BC_fcst_err_cma_{fh:03d}.csv'.format(fh=ifstH)
    fn_err_jp = './result_train/BC_fcst_err_jp_{fh:03d}.csv'.format(fh=ifstH)
    ### set path end ###

    err1 = pd.read_csv(fn_err_ecmwf).iloc[:, 8:].values
    err2 = pd.read_csv(fn_err_ncep).iloc[:, 8:].values
    err3 = pd.read_csv(fn_err_cma).iloc[:, 8:].values
    err4 = pd.read_csv(fn_err_jp).iloc[:, 8:].values
    err1[err1>9990] = 0.0
    err2[err2>9990] = 0.0
    err3[err3>9990] = 0.0
    err4[err4>9990] = 0.0

    nstations = err1.shape[0]
    corrcoef = np.zeros((nstations, 9))
    # station id, lon, lat
    corrcoef[:, 0:3] = pd.read_csv(fn_err_ecmwf).iloc[:, 3:6]
    for i in range(nstations): 
        # ec-ncep
        corrcoef[i, 3] = np.corrcoef(err1[i, :], err2[i, :])[0, 1]
        # ec-cma
        corrcoef[i, 4] = np.corrcoef(err1[i, :], err3[i, :])[0, 1]
        # ec-jp
        corrcoef[i, 5] = np.corrcoef(err1[i, :], err4[i, :])[0, 1]
        # ncep-cma
        corrcoef[i, 6] = np.corrcoef(err2[i, :], err3[i, :])[0, 1]
        # ncep-jp
        corrcoef[i, 7] = np.corrcoef(err2[i, :], err4[i, :])[0, 1]
        # cma-jp
        corrcoef[i, 8] = np.corrcoef(err3[i, :], err4[i, :])[0, 1]

    header1 = 'stid, lon, lat, ec_ncep_cc, ec_cma_cc, ec_jp_cc, ncep_cma_cc, ncep_jp_cc, cma_jp_cc'
    fmt1 = '%d ' + '%8.4f'*8

    ifn_output = fn_corrcoef.format(fh=ifstH)
    np.savetxt(ifn_output, corrcoef, header=header1, fmt=fmt1)


def wfunc1(mae):
    return 1/mae
def wfunc2(mae): 
    return 1/(mae**2) 

def station_blending_2fcst_basic_MAE(d1_utc, d2_utc, fstH, 
                                     fn_fcst1, fn_fcst2, fn_bld, 
                                     fn_fcst1_mae, fn_fcst2_mae, wfunc): 
    """
    最基本的融合方法，基于 MAE 权重，对站点预报进行加权平均
    """
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = d1_utc

    while idate_utc <= d2_utc: 
        ifn_fcst1 = fn_fcst1.format(t=idate_utc, fh=fstH)
        ifn_fcst2 = fn_fcst2.format(t=idate_utc, fh=fstH)
        ifn_fcst1_mae = fn_fcst1_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst2_mae = fn_fcst2_mae.format(t=idate_utc, fh=fstH)

        ifst1_s = meb.read_stadata_from_micaps3(ifn_fcst1, station)
        ifst2_s = meb.read_stadata_from_micaps3(ifn_fcst2, station)
        ifst1_mae_s = meb.read_stadata_from_micaps3(ifn_fcst1_mae, station)
        ifst2_mae_s = meb.read_stadata_from_micaps3(ifn_fcst2_mae, station)

        if (ifst1_s is None) or (ifst2_s is None) or (ifst1_mae_s is None) or (ifst2_mae_s is None): 
            idate_utc += timedelta(hours=24)
            continue
        
        _w1 = wfunc(ifst1_mae_s['data0'].values)
        _w2 = wfunc(ifst2_mae_s['data0'].values)
        w1 = (_w1)/(_w1+_w2)
        w2 = 1 - w1 
        ifst1_s['blend'] = ifst1_s['data0'] * w1 + ifst2_s['data0'] * w2 
        columns = list(ifst1_s.columns)
        columns.remove('data0')
        ifst_bld_s = ifst1_s.loc[:, columns]

        ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
        meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
        idate_utc += timedelta(hours=24)
        continue


def station_blending_3fcst_basic_MAE(d1_utc, d2_utc, fstH, 
                                     fn_fcst1, fn_fcst2, fn_fcst3, fn_bld, 
                                     fn_fcst1_mae, fn_fcst2_mae, fn_fcst3_mae, 
                                     wfunc): 
    """
    最基本的融合方法，基于 MAE 权重，对站点预报进行加权平均
    """
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = d1_utc

    while idate_utc <= d2_utc: 
        ifn_fcst1 = fn_fcst1.format(t=idate_utc, fh=fstH)
        ifn_fcst2 = fn_fcst2.format(t=idate_utc, fh=fstH)
        ifn_fcst3 = fn_fcst3.format(t=idate_utc, fh=fstH)
        ifn_fcst1_mae = fn_fcst1_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst2_mae = fn_fcst2_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst3_mae = fn_fcst3_mae.format(t=idate_utc, fh=fstH)

        ifst1_s = meb.read_stadata_from_micaps3(ifn_fcst1, station)
        ifst2_s = meb.read_stadata_from_micaps3(ifn_fcst2, station)
        ifst3_s = meb.read_stadata_from_micaps3(ifn_fcst3, station)
        ifst1_mae_s = meb.read_stadata_from_micaps3(ifn_fcst1_mae, station)
        ifst2_mae_s = meb.read_stadata_from_micaps3(ifn_fcst2_mae, station)
        ifst3_mae_s = meb.read_stadata_from_micaps3(ifn_fcst3_mae, station)

        if (ifst1_s is None) or (ifst2_s is None) or (ifst3_s is None) or (ifst1_mae_s is None) or (ifst2_mae_s is None) or (ifst3_mae_s is None): 
            idate_utc += timedelta(hours=24)
            continue

        _w1 = wfunc(ifst1_mae_s['data0'].values)
        _w2 = wfunc(ifst2_mae_s['data0'].values)
        _w3 = wfunc(ifst3_mae_s['data0'].values)
        _sum = _w1+_w2+_w3
        w1 = _w1/_sum
        w2 = _w2/_sum
        w3 = 1 - w1 -w2
        ifst1_s['blend'] = ifst1_s['data0'] * w1 + ifst2_s['data0'] * w2 + ifst3_s['data0'] * w3
        columns = list(ifst1_s.columns)
        columns.remove('data0')
        ifst_bld_s = ifst1_s.loc[:, columns]

        ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
        meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
        idate_utc += timedelta(hours=24)
        continue


def station_blending_multi_basic_MAE(d1_utc, d2_utc, fstH, fn_bld, 
                                     fn_fcst_dict, fn_mae_dict, wfunc): 
    """
    权重融合预报
    """
    station = meb.read_station(meb.station_国家站)
    nstations = len(station)
    station['data0'] = np.nan
    idate_utc = d1_utc

    n_total_models = len(fn_fcst_dict.keys())
    fcst_template = None 

    while idate_utc <= d2_utc: 
        valid_models = []
        today_fcsts = np.zeros((nstations, n_total_models))
        today_ws = np.zeros((nstations, n_total_models))
        
        # collect today's forecasts
        i = 0
        for imodel in fn_fcst_dict.keys(): 
            ifn_fcst = fn_fcst_dict[imodel].format(t=idate_utc, fh=fstH)
            ifn_mae = fn_mae_dict[imodel].format(t=idate_utc, fh=fstH)

            ifcst_s = meb.read_stadata_from_micaps3(ifn_fcst, station)
            ifcst_mae_s = meb.read_stadata_from_micaps3(ifn_mae, station)

            if (ifcst_s is None) or (ifcst_mae_s is None): 
                continue
            
            if i == 0: 
                fcst_template = deepcopy(ifcst_s)

            valid_models.append(imodel)
            today_fcsts[:, i] = ifcst_s['data0'].values
            today_ws[:, i] = wfunc(ifcst_mae_s['data0'].values)
            i += 1

        n_valid_models = len(valid_models)

        if n_valid_models < 1: 
            # 无有效预报数据
            pass
        else:
            # start to blending them 
            today_ws /= np.sum(today_ws, axis=1).reshape(-1, 1)
            bld_results = np.sum(today_ws[:, 0:n_valid_models]*today_fcsts[:, 0:n_valid_models], axis=1)

            # output 
            fcst_template['blend'] = bld_results
            columns = list(fcst_template.columns)
            columns.remove('data0')
            ifst_bld_s = fcst_template.loc[:, columns]

            ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
            meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
        
        idate_utc += timedelta(hours=24)
        continue


def station_blending_multi_cc_MAE(d1_utc, d2_utc, fstH, fn_bld, 
                                  fn_fcst_dict, fn_mae_dict, fn_cc, cc_column_names): 
    """
    多预报权重融合
    """
    station = meb.read_station(meb.station_国家站)
    nstations = len(station)
    station['data0'] = np.nan
    idate_utc = d1_utc

    ccs = np.loadtxt(fn_cc)
    ccs = pd.DataFrame(ccs, columns=cc_column_names)
    a1 = np.sqrt(np.pi/2)

    n_total_models = len(fn_fcst_dict.keys())
    fcst_template = None 

    while idate_utc <= d2_utc: 
        valid_models = []
        today_fcsts = np.zeros((nstations, n_total_models))
        today_sigmas = np.zeros((nstations, n_total_models))

        # collect today's forecasts
        i = 0
        for imodel in fn_fcst_dict.keys(): 
            ifn_fcst = fn_fcst_dict[imodel].format(t=idate_utc, fh=fstH)
            ifn_mae = fn_mae_dict[imodel].format(t=idate_utc, fh=fstH)

            ifcst_s = meb.read_stadata_from_micaps3(ifn_fcst, station)
            ifcst_mae_s = meb.read_stadata_from_micaps3(ifn_mae, station)

            if (ifcst_s is None) or (ifcst_mae_s is None): 
                continue
            
            if i == 0: 
                fcst_template = deepcopy(ifcst_s)

            valid_models.append(imodel)
            today_fcsts[:, i] = ifcst_s['data0'].values
            today_sigmas[:, i] = a1*ifcst_mae_s['data0'].values
            i += 1

        n_valid_models = len(valid_models)

        if n_valid_models < 1: 
            print('{t: %Y%m%d%H}-{fh:03d} start blend: NO VALID models'.format(t=idate_utc, fh=fstH))
            pass
        elif n_valid_models == 1: 
            # 就一个模式
            ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
            meb.write_stadata_to_micaps3(fcst_template, ifn_bld, True)
        else: 
            # start to blending them 
            # 1. creat the SIGMA Matrix
            print('{t: %Y%m%d%H}-{fh:03d} start blend: '.format(t=idate_utc, fh=fstH) + ','.join(valid_models))
            TOTAL_SIG = np.zeros((nstations, n_valid_models, n_valid_models))
            for i in range(n_valid_models): 
                for j in range(n_valid_models): 
                    if i > j: 
                        continue
                    elif i == j:
                        TOTAL_SIG[:, i, j] = today_sigmas[:, i] * today_sigmas[:, j]
                    else: # i < j
                        cc_idx = '%s_%s_cc' % (valid_models[i], valid_models[j])
                        if cc_idx in ccs.columns:
                            TOTAL_SIG[:, i, j] = TOTAL_SIG[:, j, i] = ccs[cc_idx].values*today_sigmas[:, i]*today_sigmas[:,j]
                        else:
                            cc_idx = '%s_%s_cc' % (valid_models[j], valid_models[i])
                            TOTAL_SIG[:, i, j] = TOTAL_SIG[:, j, i] = ccs[cc_idx].values*today_sigmas[:, i]*today_sigmas[:,j]
            # 2. for each station, calculate the blending weights
            #    blend them together
            #bld_ws = np.zeros((nstations, n_models))
            bld_results = np.zeros(nstations)
            for i in range(nstations): 
                iSIG = np.squeeze(TOTAL_SIG[i, :])
                if np.isnan(np.sum(iSIG)): 
                    bld_results[i] = np.nan
                else:
                    w = np.dot(np.linalg.pinv(iSIG), np.ones((n_valid_models, 1)))
                    w = w / np.sum(w)
                    bld_results[i] = np.dot(today_fcsts[i, 0:n_valid_models], w)
            # 3. output
                    
            fcst_template['blend'] = bld_results
            columns = list(fcst_template.columns)
            columns.remove('data0')
            ifst_bld_s = fcst_template.loc[:, columns]

            ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
            meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
            
        idate_utc += timedelta(hours=24)
        continue
        

def station_blending_3fcst_cc_MAE(d1_utc, d2_utc, fstH, 
                                  fn_fcst1, fn_fcst2, fn_fcst3, fn_bld, 
                                  fn_fcst1_mae, fn_fcst2_mae, fn_fcst3_mae, fn_corrcoef):
    """
    测试 3元融合 改进算法
    """
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = d1_utc

    ccs = np.loadtxt(fn_corrcoef)
    P12 = ccs[:, 3]
    P13 = ccs[:, 4]
    P23 = ccs[:, 5]
    
    a1 = np.sqrt(np.pi/2)
    
    while idate_utc <= d2_utc: 
        ifn_fcst1 = fn_fcst1.format(t=idate_utc, fh=fstH)
        ifn_fcst2 = fn_fcst2.format(t=idate_utc, fh=fstH)
        ifn_fcst3 = fn_fcst3.format(t=idate_utc, fh=fstH)
        ifn_fcst1_mae = fn_fcst1_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst2_mae = fn_fcst2_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst3_mae = fn_fcst3_mae.format(t=idate_utc, fh=fstH)

        ifst1_s = meb.read_stadata_from_micaps3(ifn_fcst1, station)
        ifst2_s = meb.read_stadata_from_micaps3(ifn_fcst2, station)
        ifst3_s = meb.read_stadata_from_micaps3(ifn_fcst3, station)
        ifst1_mae_s = meb.read_stadata_from_micaps3(ifn_fcst1_mae, station)
        ifst2_mae_s = meb.read_stadata_from_micaps3(ifn_fcst2_mae, station)
        ifst3_mae_s = meb.read_stadata_from_micaps3(ifn_fcst3_mae, station)

        if (ifst1_s is None) or (ifst2_s is None) or (ifst3_s is None) or (ifst1_mae_s is None) or (ifst2_mae_s is None) or (ifst3_mae_s is None): 
            idate_utc += timedelta(hours=24)
            continue
        
        fst1 = ifst1_s['data0'].values
        fst2 = ifst2_s['data0'].values
        fst3 = ifst3_s['data0'].values
        mae1 = ifst1_mae_s['data0'].values
        mae2 = ifst2_mae_s['data0'].values
        mae3 = ifst3_mae_s['data0'].values
        s1 = a1*mae1 
        s2 = a1*mae2 
        s3 = a1*mae3

        SIG11 = s1*s1
        SIG12 = P12*s1*s2 
        SIG13 = P13*s1*s3
        SIG22 = s2*s2
        SIG23 = P23*s2*s3
        SIG33 = s3*s3 
        TOTAL_SIG = np.zeros((len(mae1), 3, 3))
        TOTAL_SIG[:, 0, 0] = SIG11
        TOTAL_SIG[:, 0, 1] = TOTAL_SIG[:, 1, 0] = SIG12
        TOTAL_SIG[:, 0, 2] = TOTAL_SIG[:, 2, 0] = SIG13
        TOTAL_SIG[:, 1, 1] = SIG22
        TOTAL_SIG[:, 1, 2] = TOTAL_SIG[:, 2, 1] = SIG23
        TOTAL_SIG[:, 2, 2] = SIG33

        bld_result = np.zeros((len(mae1)))
        for i in range(len(mae1)): 
            iSIG = np.squeeze(TOTAL_SIG[i, :])
            if np.isnan(np.sum(iSIG)): 
                bld_result[i] = np.nan
            else:
                try:
                    w = np.dot(np.linalg.pinv(iSIG), np.ones((3, 1)))
                except: 
                    print('error in compute pinv')
                    print(iSIG)
                    exit(0)
                w = w / np.sum(w)

                bld_result[i] = w[0]*fst1[i] + w[1]*fst2[i] + w[2]*fst3[i]

        ifst1_s['blend'] = bld_result
        columns = list(ifst1_s.columns)
        columns.remove('data0')
        ifst_bld_s = ifst1_s.loc[:, columns]

        ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
        meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
        idate_utc += timedelta(hours=24)
        continue


def station_blending_2fcst_cc_MAE(d1_utc, d2_utc, fstH, 
                                  fn_fcst1, fn_fcst2, fn_bld, 
                                  fn_fcst1_mae, fn_fcst2_mae, fn_corrcoef, cc_idx):
    """
    考虑到参与融合的站点时不独立的，因此这里对权重系数 w 的求取进行改进
    # ec + ncep:    cc_idx = 3
    # ec + cma:     cc_idx = 4
    # ncep + cma:   cc_idx = 5
    """ 
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan
    idate_utc = d1_utc

    corrcoef = np.loadtxt(fn_corrcoef)[:, cc_idx]
    #corrcoef_df = pd.DataFrame(corrcoef, columns=['stid', 'corrcoef'])

    a1 = np.sqrt(np.pi/2)

    while idate_utc <= d2_utc: 
        ifn_fcst1 = fn_fcst1.format(t=idate_utc, fh=fstH)
        ifn_fcst2 = fn_fcst2.format(t=idate_utc, fh=fstH)
        ifn_fcst1_mae = fn_fcst1_mae.format(t=idate_utc, fh=fstH)
        ifn_fcst2_mae = fn_fcst2_mae.format(t=idate_utc, fh=fstH)

        ifst1_s = meb.read_stadata_from_micaps3(ifn_fcst1, station)
        ifst2_s = meb.read_stadata_from_micaps3(ifn_fcst2, station)
        ifst1_mae_s = meb.read_stadata_from_micaps3(ifn_fcst1_mae, station)
        ifst2_mae_s = meb.read_stadata_from_micaps3(ifn_fcst2_mae, station)

        if (ifst1_s is None) or (ifst2_s is None) or (ifst1_mae_s is None) or (ifst2_mae_s is None): 
            idate_utc += timedelta(hours=24)
            continue
        
        mae1 = ifst1_mae_s['data0'].values
        mae2 = ifst2_mae_s['data0'].values
        sigma1 = a1*mae1 
        sigma2 = a1*mae2 
        t1 = sigma1*sigma2*corrcoef
        t2 = sigma2*sigma2 - t1
        t3 = sigma1*sigma1 - t1
        w1 = t2 / (t2 + t3)
        w2 = 1-w1

        ifst1_s['blend'] = ifst1_s['data0'] * w1 + ifst2_s['data0'] * w2 
        columns = list(ifst1_s.columns)
        columns.remove('data0')
        ifst_bld_s = ifst1_s.loc[:, columns]

        ifn_bld = fn_bld.format(t=idate_utc, fh=fstH)
        meb.write_stadata_to_micaps3(ifst_bld_s, ifn_bld, True)
        idate_utc += timedelta(hours=24)
        continue


def main_proc(d1_utc, d2_utc, ifstH, result_path): 
    """
    主程序
    """

    ### set path start ###
    # observation data path
    fn_obs = 'z:/YLRC_STATION/TEMP/rt0/{t:%Y/%Y%m%d%H}.000'

    # de-biased station forecat, should be the same as [de-biased forecast path] in Station_FCST_BC.py
    fn_t2m_ecmwf_bced_s = './raw_data/ECMWF/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_NCEP_bced_s = './raw_data/NCEP/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_CMA_bced_s = './raw_data/CMA/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'
    fn_t2m_jp_bced_s = './raw_data/jp/{t:%Y%m%d%H/SBCEDFCST_%Y%m%d%H}.{fh:03d}.m3'

    # pre-calculated rolling updated MAE data path 
    fn_mae_ecmwf_bced_s = './raw_data/ECMWF/{t:%Y%m%d%H/SMAE_BCED_%Y%m%d%H}.{fh:03d}.m3'
    fn_mae_NCEP_bced_s = './raw_data/NCEP/{t:%Y%m%d%H/SMAE_BCED_%Y%m%d%H}.{fh:03d}.m3'
    fn_mae_CMA_bced_s = './raw_data/CMA/{t:%Y%m%d%H/SMAE_BCED_%Y%m%d%H}.{fh:03d}.m3'
    fn_mae_jp_bced_s = './raw_data/jp/{t:%Y%m%d%H/SMAE_BCED_%Y%m%d%H}.{fh:03d}.m3'

    # corrcoef data path 
    # shoudl be the same as in "__main__"
    fn_corrcoef = './result_train/corrcoef_{fh:03d}.csv'.format(fh=ifstH)
    
    # blended forecast data path
    # basic1: w ~ 1/MAE
    fn_bld_basic1_ec_jp = './raw_data/BLD_EC_JP/{t:%Y%m%d%H/SBasic1_%Y%m%d%H}.{fh:03d}.m3'
    fn_bld_basic1_ec_ncep_cma_jp = './raw_data/BLD_EC_NCEP_CMA_JP/{t:%Y%m%d%H/SBasic1_%Y%m%d%H}.{fh:03d}.m3'
    # cc: w improved by correlation coefficients
    fn_bld_cc_ec_jp = './raw_data/BLD_EC_JP/{t:%Y%m%d%H/SCC_%Y%m%d%H}.{fh:03d}.m3'
    fn_bld_cc_ec_ncep_cma_jp = './raw_data/BLD_EC_NCEP_CMA_JP/{t:%Y%m%d%H/SCC_%Y%m%d%H}.{fh:03d}.m3'
    
    # error files data path
    # basic1: w ~ 1/MAE
    fn_bld_basic1_err_ec_jp = result_path + '/bld_basic1_err_ec_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_bld_basic1_err_ec_ncep_cma_jp = result_path + '/bld_basic1_err_ec_ncep_cma_jp_{fh:03d}.csv'.format(fh=ifstH)
    # cc: w improved by correlation coefficients
    fn_bld_cc_err_ec_jp = result_path + '/bld_cc_err_ec_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_bld_cc_err_ec_ncep_cma_jp = result_path + '/bld_cc_err_ec_ncep_cma_jp_{fh:03d}.csv'.format(fh=ifstH)
    ### set path end ###
    
    # pre-calculate rolling updated mae
    prepare_t2m_mae(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf_bced_s, fn_obs, fn_mae_ecmwf_bced_s)
    prepare_t2m_mae(d1_utc, d2_utc, ifstH, fn_t2m_NCEP_bced_s, fn_obs, fn_mae_NCEP_bced_s)
    prepare_t2m_mae(d1_utc, d2_utc, ifstH, fn_t2m_CMA_bced_s, fn_obs, fn_mae_CMA_bced_s)
    prepare_t2m_mae(d1_utc, d2_utc, ifstH, fn_t2m_jp_bced_s, fn_obs, fn_mae_jp_bced_s)
    
    # basic blending ECMWF + JAPAN-HR, w ~ (1/mae)
    station_blending_2fcst_basic_MAE(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf_bced_s, fn_t2m_jp_bced_s, fn_bld_basic1_ec_jp, fn_mae_ecmwf_bced_s, fn_mae_jp_bced_s, wfunc=wfunc1)

    # improved blending ECMWF + JAPAN-HR
    station_blending_2fcst_cc_MAE(d1_utc, d2_utc, ifstH, fn_t2m_ecmwf_bced_s, fn_t2m_jp_bced_s, fn_bld_cc_ec_jp, fn_mae_ecmwf_bced_s, fn_mae_jp_bced_s, fn_corrcoef, cc_idx=5)

    # basic blending and improved blending all four models
    fn_fcsts = {'ec': fn_t2m_ecmwf_bced_s, 
                'ncep': fn_t2m_NCEP_bced_s, 
                'cma': fn_t2m_CMA_bced_s, 
                'jp': fn_t2m_jp_bced_s}
    fn_maes = {'ec': fn_mae_ecmwf_bced_s, 
                'ncep': fn_mae_NCEP_bced_s, 
                'cma': fn_mae_CMA_bced_s, 
                'jp': fn_mae_jp_bced_s}
    station_blending_multi_basic_MAE(d1_utc, d2_utc, ifstH, fn_bld_basic1_ec_ncep_cma_jp, fn_fcsts, fn_maes, wfunc1)
    station_blending_multi_cc_MAE(d1_utc, d2_utc, ifstH, fn_bld_cc_ec_ncep_cma_jp, fn_fcsts, fn_maes, fn_corrcoef, 
                                  ['stid', 'lon', 'lat', 'ec_ncep_cc', 'ec_cma_cc', 'ec_jp_cc', 'ncep_cma_cc', 'ncep_jp_cc', 'cma_jp_cc'])

    # statistical the forecast errors for 4 blended forecasts
    t2m_verification(d1_utc, d2_utc, ifstH, fn_bld_basic1_ec_jp, fn_obs, fn_bld_basic1_err_ec_jp)
    t2m_verification(d1_utc, d2_utc, ifstH, fn_bld_basic1_ec_ncep_cma_jp, fn_obs, fn_bld_basic1_err_ec_ncep_cma_jp)
    t2m_verification(d1_utc, d2_utc, ifstH, fn_bld_cc_ec_ncep_cma_jp, fn_obs, fn_bld_cc_err_ec_ncep_cma_jp)
    t2m_verification(d1_utc, d2_utc, ifstH, fn_bld_cc_ec_jp, fn_obs, fn_bld_cc_err_ec_jp)
    
    fns = [fn_bld_basic1_err_ec_jp, fn_bld_basic1_err_ec_ncep_cma_jp, 
           fn_bld_cc_err_ec_jp, fn_bld_cc_err_ec_ncep_cma_jp]
    descriptions = ['EC+JP Basic1', 'EC+NCEP+CMA+JP Basic1',  
                    'EC+JP cc', 'EC+NCEP+CMA+JP cc', ]
    err_cal_mae(fns, descriptions) 



if __name__ == "__main__": 
    fn_corrcoef = './result_train/corrcoef_{fh:03d}.csv'

    fstHs = [24]
    d1_utc = datetime(2022, 3, 1, 0)
    d2_utc = datetime(2023, 3, 1, 0)
    for ifstH in fstHs:
        prepare_corrcoef(ifstH, fn_corrcoef)
        main_proc(d1_utc, d2_utc, ifstH, './result_train')
        pass
    
    d1_utc = datetime(2023, 3, 1, 0)
    d2_utc = datetime(2024, 3, 1, 0)
    for ifstH in fstHs:
        main_proc(d1_utc, d2_utc, ifstH, './result_test')
        pass 

