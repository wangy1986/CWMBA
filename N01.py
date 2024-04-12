#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : N01.py
@TIME      : 2024/03/29 16:08:07
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 对 融合预报 误差模型 进行评估
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


lgd_font = FontProperties(family='Times New Roman', size=14)
title_font = {'fontname': 'Times New Roman', 'fontsize': 16}
axis_font = {'fontname': 'Times New Roman', 'fontsize': 14}
tick_font = {'fontname': 'Times New Roman', 'fontsize': 12}


def prepare(fn_err_ec, fn_err_ncep, fn_err_cma, fn_err_jp, fn_err_bld, fn_output): 
    """
    """
    raw_mat1 = pd.read_csv(fn_err_ec)
    err_mat1 = raw_mat1.iloc[:, 8:].values 
    err_mat1[err_mat1 > 9990] = np.nan 
    
    raw_mat2 = pd.read_csv(fn_err_ncep)
    err_mat2 = raw_mat2.iloc[:, 8:].values 
    err_mat2[err_mat2 > 9990] = np.nan 

    raw_mat3 = pd.read_csv(fn_err_cma)
    err_mat3 = raw_mat3.iloc[:, 8:].values 
    err_mat3[err_mat3 > 9990] = np.nan 

    raw_mat4 = pd.read_csv(fn_err_jp)
    err_mat4 = raw_mat4.iloc[:, 8:].values 
    err_mat4[err_mat4 > 9990] = np.nan 

    raw_mat_bld = pd.read_csv(fn_err_bld)
    err_mat_bld = raw_mat_bld.iloc[:, 8:].values 
    err_mat_bld[err_mat_bld > 9990] = np.nan 

    # station id 
    st_ids = raw_mat1.iloc[:, 3].values
    st_err_dif = np.zeros((len(st_ids), 24))
    st_err_dif[:, 0] = st_ids
    st_err_dif[:, 1] = raw_mat1.iloc[:, 4].values
    st_err_dif[:, 2] = raw_mat1.iloc[:, 5].values

    for i in range(len(st_ids)): 
        stid = st_ids[i]
        mae1 = np.nanmean(np.abs(err_mat1[i, :]))
        mae2 = np.nanmean(np.abs(err_mat2[i, :]))
        mae3 = np.nanmean(np.abs(err_mat3[i, :]))
        mae4 = np.nanmean(np.abs(err_mat4[i, :]))
        mae_bld = np.nanmean(np.abs(err_mat_bld[i, :]))

        # 拟合 正态分布 曲线
        fit1 = fitter.Fitter(err_mat1[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit2 = fitter.Fitter(err_mat2[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit3 = fitter.Fitter(err_mat3[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit4 = fitter.Fitter(err_mat4[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit_bld = fitter.Fitter(err_mat_bld[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit1.fit()
        fit2.fit()
        fit3.fit()
        fit4.fit()
        fit_bld.fit()
        par1 = fit1.get_best()['norm']
        par2 = fit2.get_best()['norm']
        par3 = fit3.get_best()['norm']
        par4 = fit4.get_best()['norm']
        par_bld = fit_bld.get_best()['norm']

        # ec, ncep, cma-gfs 拟合正态分布曲线的参数： loc, scale, 
        # ec
        st_err_dif[i, 3] = par1['loc']
        st_err_dif[i, 4] = par1['scale']
        st_err_dif[i, 5] = mae1
        # ncep
        st_err_dif[i, 6] = par2['loc']
        st_err_dif[i, 7] = par2['scale']
        st_err_dif[i, 8] = mae2 
        # cma-gfs
        st_err_dif[i, 9] = par3['loc']
        st_err_dif[i, 10] = par3['scale']
        st_err_dif[i, 11] = mae3
        # jp 
        st_err_dif[i, 12] = par4['loc']
        st_err_dif[i, 13] = par4['scale']
        st_err_dif[i, 14] = mae4
        # bld 
        st_err_dif[i, 15] = par_bld['loc']
        st_err_dif[i, 16] = par_bld['scale']
        st_err_dif[i, 17] = mae_bld

        # corrcoef
        ec_fst = err_mat1[i, :]
        ncep_fst = err_mat2[i, :]
        cma_fst = err_mat3[i, :]
        jp_fst = err_mat4[i, :]
        ec_fst[np.isnan(ec_fst)] = 0
        ncep_fst[np.isnan(ncep_fst)] = 0
        cma_fst[np.isnan(cma_fst)] = 0
        jp_fst[np.isnan(jp_fst)] = 0

        st_err_dif[i, 18] = np.corrcoef(ec_fst, ncep_fst)[0, 1]
        st_err_dif[i, 19] = np.corrcoef(ec_fst, cma_fst)[0, 1]
        st_err_dif[i, 20] = np.corrcoef(ec_fst, jp_fst)[0, 1]
        st_err_dif[i, 21] = np.corrcoef(ncep_fst, cma_fst)[0, 1]
        st_err_dif[i, 22] = np.corrcoef(ncep_fst, jp_fst)[0, 1]
        st_err_dif[i, 23] = np.corrcoef(cma_fst, jp_fst)[0, 1]

    # 对 st_err_dif, 按 pdf 误差 从大到小 排序
    header1 = 'stid, lon, lat, ec_norm_loc, ec_norm_scale, ec_mae, \
                ncep_norm_loc, ncep_norm_scale, ncep_mae, cma_norm_loc, cma_norm_scale, cma_mae, \
                jp_norm_loc, jp_norm_scale, jp_mae, blend_norm_loc, blend_norm_scale, blend_mae, \
                cc_ec_ncep, cc_ec_cma, cc_ec_jp, cc_ncep_cma, cc_ncep_jp, cc_cma_jp'
    fmt1 = '%d %.2f %.2f'+'%8.4f'*21
    np.savetxt(fn_output, st_err_dif, fmt=fmt1, header=header1)



def figN01(fn_data): 
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
    bld_loc_real = data_df['blend_loc'].values
    bld_sig_real = data_df['blend_scale'].values
    cc_ec_ncep = data_df['cc_ec_ncep'].values 
    cc_ec_cma = data_df['cc_ec_cma'].values 
    cc_ec_jp = data_df['cc_ec_jp'].values
    cc_ncep_cma = data_df['cc_ncep_cma'].values 
    cc_ncep_jp = data_df['cc_ncep_jp'].values
    cc_cma_jp = data_df['cc_cma_jp'].values

    _w1 = 1/(ec_mae)
    _w2 = 1/(ncep_mae)
    _w3 = 1/(cma_mae)
    _w4 = 1/(jp_mae)
    w1 = _w1/(_w1 + _w2 + _w3 + _w4)
    w2 = _w2/(_w1 + _w2 + _w3 + _w4)
    w3 = _w3/(_w1 + _w2 + _w3 + _w4)
    w4 = 1 - w1 - w2 - w3

    def sig_cal(w1, w2, w3, w4, sig1, sig2, sig3, sig4, cc12, cc13, cc14, cc23, cc24, cc34): 
        ws1 = w1*sig1 
        ws2 = w2*sig2 
        ws3 = w3*sig3 
        ws4 = w4*sig4
        term1 = ws1**2 + ws2**2 + ws3**2 + ws4**2
        term2 = cc12*ws1*ws2 + cc13*ws1*ws3 + cc14*ws1*ws4+ cc23*ws2*ws3 + cc24*ws2*ws4 + cc34*ws3*ws4
        return np.sqrt(term1 + 2*term2)

    # estimate the blending error distribution 
    bld_loc_est = w1 * ec_loc + w2 * ncep_loc + w3*cma_loc + w4*jp_loc
    bld_sig_est = sig_cal(w1, w2, w3, w4, ec_scale, ncep_scale, cma_scale, jp_scale, 
                          cc_ec_ncep, cc_ec_cma, cc_ec_jp, cc_ncep_cma, cc_ncep_jp, cc_cma_jp)

    #### 回归测试 ####
    # 回归 - μ
    reg_miu = stats.linregress(bld_loc_est, bld_loc_real)
    xmin1 = min(np.min(bld_loc_est), np.min(bld_loc_real)) - 0.05
    xmax1 = max(np.max(bld_loc_est), np.max(bld_loc_real)) + 0.05 
    x1 = np.linspace(xmin1, xmax1, 100)    
    y1 = reg_miu.slope*x1 + reg_miu.intercept
    # 回归 - σ
    reg_sig = stats.linregress(bld_sig_est, bld_sig_real)
    xmin2 = max(0, min(np.min(bld_sig_est), np.min(bld_sig_real)) - 0.1)
    xmax2 = max(np.max(bld_sig_est), np.max(bld_sig_real)) + 0.1 
    x2 = np.linspace(xmin2, xmax2, 100)    
    y2 = reg_sig.slope*x2 + reg_sig.intercept

    #### 绘图 ####
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(bld_loc_est, bld_loc_real, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(bld_loc_est, bld_loc_real)[0, 1])
    ax1.plot(x1, y1, 'k--', label='curve fit: y = %.3fx %.3f' % (reg_miu.slope, reg_miu.intercept))
    ax1.axis('equal')
    ax1.legend(prop=lgd_font, framealpha=0)
    ax1.set_title('Estimation of error distribution parameters: μ', fontdict=title_font)
    ax1.set_xlabel('Estimated μ (℃)', fontdict=axis_font)
    ax1.set_ylabel('Real μ (℃)', fontdict=axis_font)
    ax1.set_xticklabels(ax1.get_xticks(), fontdict=tick_font)
    ax1.set_yticklabels(ax1.get_yticks(), fontdict=tick_font)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.text(0.7, -0.85, '(a)', fontdict=title_font)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(bld_sig_est, bld_sig_real, '.', color='dimgray', label='raw data, correlation coefficients: %.3f' % np.corrcoef(bld_sig_est, bld_sig_real)[0, 1])
    ax2.plot(x2, y2, 'k--', label='curve fit: y = %.3fx %.3f' % (reg_sig.slope, reg_sig.intercept))
    ax2.axis('equal')
    ax2.legend(prop=lgd_font, framealpha=0)
    ax2.set_title('Estimation of error distribution parameters: σ', fontdict=title_font)
    ax2.set_xlabel('Estimated σ (℃)', fontdict=axis_font)
    ax2.set_ylabel('Real σ (℃)', fontdict=axis_font)
    ax2.set_xticklabels(ax2.get_xticks(), fontdict=tick_font)
    ax2.set_yticklabels(ax2.get_yticks(), fontdict=tick_font)
    ax2.text(4.0, 0.75, '(b)', fontdict=title_font)

    plt.savefig('N01.png')
    #plt.show()



if __name__ == "__main__": 
    fn_err_ec = './result_train/BC_fcst_err_ecmwf_024.csv'
    fn_err_ncep = './result_train/BC_fcst_err_ncep_024.csv'
    fn_err_cma = './result_train/BC_fcst_err_cma_024.csv'
    fn_err_jp = './result_train/BC_fcst_err_jp_024.csv'
    fn_err_bld = './result_train/bld_basic1_err_ec_ncep_cma_jp_024.csv'
    fn_output = './result_train/bld_basic1_pdf_ec_ncep_cma_jp_024.txt'

    #prepare(fn_err_ec, fn_err_ncep, fn_err_cma, fn_err_jp, fn_err_bld, fn_output)
    figN01(fn_output)