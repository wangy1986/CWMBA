#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : N03.py
@TIME      : 2024/03/31 15:48:25
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 绘制 不同订正方法 mae 
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)

import pandas as pd 
import numpy as np 


import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MultipleLocator, AutoMinorLocator
from matplotlib.font_manager import FontProperties

lgd_font = FontProperties(family='Arial', size=14)
title_font = {'fontname': 'Arial', 'fontsize': 16}
axis_font = {'fontname': 'Arial', 'fontsize': 14}
tick_font = {'fontname': 'Arial', 'fontsize': 12}

def bld_sigma(w1, w2, mae1, mae2, cc):
    a1 = np.sqrt(np.pi/2)
    return a1*np.sqrt((w1*mae1)**2 + (w2*mae2)**2 + 2*cc*w1*w2*mae1*mae2)

def bld_mae(w1, w2, mae1, mae2, cc):
    return np.sqrt((w1*mae1)**2 + (w2*mae2)**2 + 2*cc*w1*w2*mae1*mae2)

def cc_weight(mae1, mae2, cc): 
    
    a1 = np.pi/2
    t1 = a1*mae1*mae2*cc
    t2 = a1*mae2*mae2 - t1
    t3 = a1*mae1*mae1 - t1
    w1 = t2 / (t2 + t3)
    return [w1, 1-w1]
    '''
    SIG = np.array([[mae1**2, cc*mae1*mae2], 
                    [cc*mae1*mae2, mae2**2]])
    w = np.dot(np.linalg.pinv(SIG), np.ones((2, 1)))
    w = w / np.sum(w)
    return w
    '''
    


def select_station_fcst_example(): 
    ifstH = 24
    fn_ec = './result_test/BC_fcst_err_ecmwf_{fh:03d}.csv'.format(fh=ifstH)
    fn_cma = './result_test/BC_fcst_err_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_twb1 = './result_test/bld_basic1_err_ec_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_cwb1 = './result_test/bld_cc_err_ec_jp_{fh:03d}.csv'.format(fh=ifstH)
    fn_cc = './result_train/corrcoef_{fh:03d}.csv'.format(fh=ifstH)

    pd_ec = pd.read_csv(fn_ec)
    pd_cma = pd.read_csv(fn_cma)
    pd_twb1 = pd.read_csv(fn_twb1)
    pd_cwb1 = pd.read_csv(fn_cwb1)
    pd_cc = pd.DataFrame(np.loadtxt(fn_cc), columns=['stid', 'lon', 'lat', 
                                                     'ec_ncep', 'ec_cma', 'ec_jp', 'ncep_cma', 'ncep_jp', 'cma_jp'])

    ec_cma_cc = pd_cc['ec_jp'].values 
    st_ids = pd_ec.iloc[:, 3].values 
    err_mat_ec = pd_ec.iloc[:, 8:].values 
    err_mat_ec[err_mat_ec > 9990] = np.nan 
    err_mat_cma = pd_cma.iloc[:, 8:].values
    err_mat_cma[err_mat_cma > 9990] = np.nan 
    err_mat_twb1 = pd_twb1.iloc[:, 8:].values 
    err_mat_twb1[err_mat_twb1 > 9990] = np.nan 
    err_mat_cwb1 = pd_cwb1.iloc[:, 8:].values 
    err_mat_cwb1[err_mat_cwb1 > 9990] = np.nan 

    mae_ec = np.nanmean(np.abs(err_mat_ec), axis=1)
    mae_cma = np.nanmean(np.abs(err_mat_cma), axis=1)
    mae_twb1 = np.nanmean(np.abs(err_mat_twb1), axis=1)
    mae_cwb1 = np.nanmean(np.abs(err_mat_cwb1), axis=1)

    mae_improve = mae_twb1 - mae_cwb1
    sorted_idx = np.argsort(mae_improve)
    count_twb1_bad = 0
    count_cwb1_bad = 0

    for i, st_id in enumerate(st_ids): 
        if (mae_twb1[i] > mae_ec[i]) or (mae_twb1[i] > mae_cma[i]):
            count_twb1_bad += 1 
            #print('st id: %d, mae ec: %.3f mae cma: %.3f mae twb1: %.3f mae cwb1: %.3f' %
            #      (st_id, mae_ec[i], mae_cma[i], mae_twb1[i], mae_cwb1[i]))
            
        if (mae_cwb1[i] > mae_ec[i]) or (mae_cwb1[i] > mae_cma[i]):
            count_cwb1_bad += 1
        '''
        i_improve = mae_twb1[i] - mae_cwb1[i]
            icc = ec_cma_cc[i]
            mae1, mae2 = mae_ec[i], mae_cma[i]
            w1_twb = (1/mae1) / ((1/mae1)+(1/mae2))
            w2_twb = 1-w1_twb 
            est_mae_twb1 = bld_mae(w1_twb, w2_twb, mae1, mae2, icc)

            t1 = mae1*mae2*icc
            t2 = mae2*mae2 - t1
            t3 = mae1*mae1 - t1
            w1_cwb = t2 / (t2 + t3)
            w2_cwb = 1-w1_cwb
            est_mae_cwb = bld_mae(w1_cwb, w2_cwb, mae1, mae2, icc)
    
            if est_mae_cwb > mae1:
                print('st id: %d, mae ec: %.3f, mae cma: %.3f, mae twb1: %.3f, est mae twb1: %.3f, mae cwb1: %.3f, est mae cwb1: %.3f, cc: %.3f, ' %
                    (st_id, mae_ec[i], mae_cma[i], mae_twb1[i], est_mae_twb1, mae_cwb1[i], est_mae_cwb, icc))
            
            if (mae1 - est_mae_cwb) < 0.001: 
                print('st id: %d, mae ec: %.3f, mae cma: %.3f, mae twb1: %.3f, est mae twb1: %.3f, mae cwb1: %.3f, est mae cwb1: %.3f, cc: %.3f, ' %
                    (st_id, mae_ec[i], mae_cma[i], mae_twb1[i], est_mae_twb1, mae_cwb1[i], est_mae_cwb, icc))
                print('w1_twb: %.3f, w1_cwb: %.3f' % (w1_twb, w1_cwb))
        '''

    for i in sorted_idx: 
        if (mae_twb1[i] > mae_ec[i]) or (mae_twb1[i] > mae_cma[i]):
            print('st id: %d, mae ec: %.3f mae jp: %.3f mae twb1: %.3f mae cwb1: %.3f' %
                    (st_ids[i], mae_ec[i], mae_cma[i], mae_twb1[i], mae_cwb1[i]))
            
    print('twb1 bad: %d, cwb1 bad %d' % (count_twb1_bad, count_cwb1_bad))

def fig_N03(st_id, fstH): 
    """"""
    # twb = traditional weighted blending
    # cwb = corrcoef weighted blending
    fn_twb1 = './result_test/bld_basic1_err_ec_jp_{fh:03d}.csv'
    fn_twb2 = './result_test/bld_basic1_err_ec_ncep_cma_jp_{fh:03d}.csv'
    fn_cwb1 = './result_test/bld_cc_err_ec_jp_{fh:03d}.csv'
    fn_cwb2 = './result_test/bld_cc_err_ec_ncep_cma_jp_{fh:03d}.csv'

    maes = np.zeros((24, 4))
    # collect data for fig 03a
    
    for i, ifstH in enumerate(range(3, 73, 3)): 
        raw_mat0 = pd.read_csv(fn_twb1.format(fh=ifstH))
        err_mat0 = raw_mat0.iloc[:, 8:].values 
        err_mat0[err_mat0 > 9990] = np.nan 
        maes[i, 0] = np.nanmean(np.abs(err_mat0))

        raw_mat1 = pd.read_csv(fn_twb2.format(fh=ifstH))
        err_mat1 = raw_mat1.iloc[:, 8:].values 
        err_mat1[err_mat1 > 9990] = np.nan 
        maes[i, 1] = np.nanmean(np.abs(err_mat1))

        raw_mat2 = pd.read_csv(fn_cwb1.format(fh=ifstH))
        err_mat2 = raw_mat2.iloc[:, 8:].values 
        err_mat2[err_mat2 > 9990] = np.nan 
        maes[i, 2] = np.nanmean(np.abs(err_mat2))

        raw_mat3 = pd.read_csv(fn_cwb2.format(fh=ifstH))
        err_mat3 = raw_mat3.iloc[:, 8:].values 
        err_mat3[err_mat3 > 9990] = np.nan 
        maes[i, 3] = np.nanmean(np.abs(err_mat3))
    

    # collect for fig 03b
    fn_ec = './result_test/BC_fcst_err_ecmwf_{fh:03d}.csv'.format(fh=fstH)
    fn_jp = './result_test/BC_fcst_err_jp_{fh:03d}.csv'.format(fh=fstH)
    fn_cc = './result_train/corrcoef_{fh:03d}.csv'.format(fh=fstH)
    pd_cc = pd.DataFrame(np.loadtxt(fn_cc), columns=['stid', 'lon', 'lat', 
                                                     'ec_ncep', 'ec_cma', 'ec_jp', 'ncep_cma', 'ncep_jp', 'cma_jp'])
    
    st_cc = pd_cc[pd_cc['stid']==st_id]['ec_jp'].values[0]
    print('%d: cc: %.4f' % (st_id, st_cc))
    raw_mat_ec = pd.read_csv(fn_ec)
    raw_mat_jp = pd.read_csv(fn_jp)
    st_rec_ec = raw_mat_ec[raw_mat_ec['id']==st_id]
    st_err_ec = st_rec_ec.iloc[:, 8:].values
    st_err_ec[st_err_ec>9999] = np.nan 
    st_mae_ec = np.nanmean(np.abs(st_err_ec))
    st_rec_jp = raw_mat_jp[raw_mat_jp['id']==st_id]
    st_err_jp = st_rec_jp.iloc[:, 8:].values
    st_err_jp[st_err_jp>9999] = np.nan
    st_mae_jp = np.nanmean(np.abs(st_err_jp))
    
    w1 = np.arange(-0.25, 2.01, 0.01)
    y2 = bld_mae(w1, 1-w1, st_mae_ec, st_mae_jp, st_cc)
    w1_twb = (1/st_mae_ec)/(1/st_mae_ec+1/st_mae_jp)
    w = cc_weight(st_mae_ec, st_mae_jp, st_cc)
    mae_twb = bld_mae(w1_twb, 1-w1_twb, st_mae_ec, st_mae_jp, st_cc)
    mae_cwb = bld_mae(w[0], w[1], st_mae_ec, st_mae_jp, st_cc)
    
    print('/// 3a ///')    
    print('mean mae:', np.mean(maes, axis=0))
    print('/// 3b ///')
    print('mae jp: twb: %.3f, cwb: %.3f' % ((1-w1_twb), w[1]))

    # plot 
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    x = np.arange(3, 73, 3)
    ax1.plot(x, maes[:, 0], 'g--', label='ECMWF+JAPAN | w ~ 1/MAE')
    ax1.plot(x, maes[:, 1], 'm--', label='ECMWF+NCEP+CMA+JAPAN | w ~ 1/MAE')
    ax1.plot(x, maes[:, 2], 'g-',  label='ECMWF+JAPAN | w see Eq. (16)')
    ax1.plot(x, maes[:, 3], 'm-',  label='ECMWF+NCEP+CMA+JAPAN | w see Eq. (16)')

    ax1.legend(prop=lgd_font, framealpha=0)
    ax1.set_title('forecast MAE for different blending algorithm', fontdict=title_font)
    ax1.set_xlabel('forecast lead time (H)', fontdict=axis_font)
    ax1.set_ylabel('MAE (K)', fontdict=axis_font)
    
    ax1.xaxis.set_ticks(np.arange(0, 73, 12))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.set_xticklabels(ax1.get_xticks(), fontdict=tick_font)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.set_yticklabels(ax1.get_yticks(), fontdict=tick_font)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.text(72, 1.05, '(a)', fontdict=title_font)

    # fig 03b
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(w1, y2, 'g-', label='Estimated blending MAE curve')
    ax2.plot([0,0], [1.5, bld_mae(0, 1, st_mae_ec, st_mae_jp, st_cc)], 'k--')
    ax2.plot([1,1], [1.5, bld_mae(1, 0, st_mae_ec, st_mae_jp, st_cc)], 'k--')
    ax2.plot(w1_twb, mae_twb, 'k+', label='the traditional method', markersize=14)
    ax2.plot(w[0], mae_cwb, 'kx', label='our improved method', markersize=14)
    
    ax2.legend(prop=lgd_font, framealpha=0)
    ax2.set_title('Station %d blending MAE' % (st_id), fontdict=title_font)
    ax2.set_xlabel('ECMWF weight', fontdict=axis_font)
    ax2.set_ylabel('Estimated MAE (K)', fontdict=axis_font)
    
    ax2.xaxis.set_ticks(np.arange(-0.25, 2.01, 0.25))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.set_xticklabels(ax2.get_xticks(), fontdict=tick_font)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_yticklabels(ax2.get_yticks(), fontdict=tick_font)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.text(2.0, 1.5, '(b)', fontdict=title_font)

    plt.savefig('./N03.png')
    #plt.show()


if __name__ == "__main__": 
    #select_station_fcst_example()
    fig_N03(53594, 24)

