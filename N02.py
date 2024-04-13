#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : N02.py
@TIME      : 2024/03/29 17:23:26
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : ...
'''


### to import parent dir files ###
# import os, sys
### this is for jupyter notebook ###
#current_folder = globals()['_dh'][0]
#parentdir = os.path.dirname(current_folder)
### this is for normal python file ###
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)


import numpy as np 
import pandas as pd
from scipy import stats 
import fitter 

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties


lgd_font = FontProperties(family='Arial', size=14)
title_font = {'fontname': 'Arial', 'fontsize': 16}
axis_font = {'fontname': 'Arial', 'fontsize': 14}
tick_font = {'fontname': 'Arial', 'fontsize': 12}



def figN02(fn_data): 
    dmat = np.loadtxt(fn_data, )
    data_df = pd.DataFrame(dmat, columns=['stid', 'lon', 'lat', 
                                          'ec_loc', 'ec_scale', 'ec_mae', 
                                          'ncep_loc', 'ncep_scale', 'ncep_mae', 
                                          'cma_loc', 'cma_scale', 'cma_mae', 
                                          'jp_loc', 'jp_scale', 'jp_mae', 
                                          'blend_loc', 'blend_scale', 'blend_mae', 
                                          'cc_ec_ncep', 'cc_ec_cma', 'cc_ec_jp',
                                          'cc_ncep_cma', 'cc_ncep_jp', 'cc_cma_jp'])

    data_df.dropna(axis=0, inplace=True)


    ec_loc = data_df['ec_loc'].values
    ec_scale = data_df['ec_scale'].values
    ec_mae = data_df['ec_mae'].values 
    ncep_loc = data_df['ncep_loc'].values
    ncep_scale = data_df['ncep_scale'].values
    ncep_mae = data_df['ncep_mae'].values 
    cma_loc = data_df['cma_loc'].values
    cma_scale = data_df['cma_scale'].values
    cma_mae = data_df['cma_mae'].values 
    jp_loc = data_df['jp_loc'].values 
    jp_scale = data_df['jp_scale'].values 
    jp_mae = data_df['jp_mae'].values 

    def mae_cal(miu, sig):
        a1 = np.sqrt(2/np.pi)
        a2 = 1/np.sqrt(2*np.pi)
        return a1*sig + a2*(miu**2)/sig
    
    # 通过公式 计算 mae
    ec_mae_est = mae_cal(ec_loc, ec_scale)
    ncep_mae_est = mae_cal(ncep_loc, ncep_scale)
    cma_mae_est = mae_cal(cma_loc, cma_scale)
    jp_mae_est = mae_cal(jp_loc, jp_scale)
    # 回归
    reg_ec = stats.linregress(ec_mae_est, ec_mae)
    reg_ncep = stats.linregress(ncep_mae_est, ncep_mae)
    reg_cma = stats.linregress(cma_mae_est, cma_mae)
    reg_jp = stats.linregress(jp_mae_est, jp_mae)

    # 绘制 计算MAE 与 真实MAE 的散点图以及回归方程
    fig = plt.figure(figsize=(26, 6))

    ax1 = fig.add_subplot(1, 4, 1)
    xmin = max(0, min(np.min(ec_mae_est), np.min(ec_mae)) - 0.1)
    xmax = max(np.max(ec_mae_est), np.max(ec_mae)) + 0.1 
    x = np.linspace(xmin, xmax, 100)    
    y = reg_ec.slope*x + reg_ec.intercept
    ax1.plot(ec_mae_est, ec_mae, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(ec_mae_est, ec_mae)[0, 1])
    ax1.plot(x, y, 'k--', label='curve fit: y = %.3fx %.3f' % (reg_ec.slope, reg_ec.intercept))
    ax1.axis('equal')
    ax1.legend(prop=lgd_font, framealpha=0)
    ax1.set_title('ECMWF', fontdict=title_font)
    ax1.set_xlabel('Estimated MAE (K)', fontdict=axis_font)
    ax1.set_ylabel('Real MAE (K)', fontdict=axis_font)
    ax1.set_xticklabels(ax1.get_xticks(), fontdict=tick_font)
    ax1.set_yticklabels(ax1.get_yticks(), fontdict=tick_font)

    ax2 = fig.add_subplot(1, 4, 2)
    xmin = max(0, min(np.min(ncep_mae_est), np.min(ncep_mae)) - 0.1)
    xmax = max(np.max(ncep_mae_est), np.max(ncep_mae)) + 0.1 
    x = np.linspace(xmin, xmax, 100)    
    y = reg_ncep.slope*x + reg_ncep.intercept
    ax2.plot(ncep_mae_est, ncep_mae, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(ncep_mae_est, ncep_mae)[0, 1])
    ax2.plot(x, y, 'k--', label='curve fit: y = %.3fx %.3f' % (reg_ncep.slope, reg_ncep.intercept))
    ax2.axis('equal')
    ax2.legend(prop=lgd_font, framealpha=0)
    ax2.set_title('NCEP-GFS', fontdict=title_font)
    ax2.set_xlabel('Estimated MAE (K)', fontdict=axis_font)
    ax2.set_ylabel('Real MAE (K)', fontdict=axis_font)
    ax2.set_xticklabels(ax2.get_xticks(), fontdict=tick_font)
    ax2.set_yticklabels(ax2.get_yticks(), fontdict=tick_font)
    
    ax3 = fig.add_subplot(1, 4, 3)
    xmin = max(0, min(np.min(cma_mae_est), np.min(cma_mae)) - 0.1)
    xmax = max(np.max(cma_mae_est), np.max(cma_mae)) + 0.1 
    x = np.linspace(xmin, xmax, 100)    
    y = reg_cma.slope*x + reg_cma.intercept
    ax3.plot(cma_mae_est, cma_mae, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(cma_mae_est, cma_mae)[0, 1])
    ax3.plot(x, y, 'k--', label='curve fit: y = %.3fx %.3f' % (reg_cma.slope, reg_cma.intercept))
    ax3.axis('equal')
    ax3.legend(prop=lgd_font, framealpha=0)
    ax3.set_title('CMA-GFS', fontdict=title_font)
    ax3.set_xlabel('Estimated MAE (K)', fontdict=axis_font)
    ax3.set_ylabel('Real MAE (K)', fontdict=axis_font)
    ax3.set_xticklabels(ax3.get_xticks(), fontdict=tick_font)
    ax3.set_yticklabels(ax3.get_yticks(), fontdict=tick_font)

    ax4 = fig.add_subplot(1, 4, 4)
    xmin = max(0, min(np.min(jp_mae_est), np.min(jp_mae)) - 0.1)
    xmax = max(np.max(jp_mae_est), np.max(jp_mae)) + 0.1 
    x = np.linspace(xmin, xmax, 100)    
    y = reg_jp.slope*x + reg_jp.intercept
    ax4.plot(jp_mae_est, jp_mae, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(jp_mae_est, jp_mae)[0, 1])
    ax4.plot(x, y, 'k--', label='curve fit: y = %.3fx +%.3f' % (reg_jp.slope, reg_jp.intercept))
    ax4.axis('equal')
    ax4.legend(prop=lgd_font, framealpha=0)
    ax4.set_title('JAPAN-HR', fontdict=title_font)
    ax4.set_xlabel('Estimated MAE (K)', fontdict=axis_font)
    ax4.set_ylabel('Real MAE (K)', fontdict=axis_font)
    ax4.set_xticklabels(ax4.get_xticks(), fontdict=tick_font)
    ax4.set_yticklabels(ax4.get_yticks(), fontdict=tick_font)

    plt.savefig('./N02.png')
    #plt.show()
    
    
if __name__ == "__main__": 
    fn_output = './result_train/bld_basic1_pdf_ec_ncep_cma_jp_024.txt'
    figN02(fn_output)
