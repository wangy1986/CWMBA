#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@FILE      : station_fcst_err_analysis.py
@TIME      : 2024/03/21 15:17:33
@AUTHOR    : wangyu / NMC
@VERSION   : 1.0
@DESC      : 进行站点预报误差分析
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
import meteva.base as meb  
import pandas as pd 
from copy import deepcopy
from datetime import datetime, timedelta

#import matplotlib 
#matplotlib.use('Agg') 
from matplotlib import pyplot as plt

from scipy import stats 
from scipy import signal 

import fitter 

def t2m_verification(start_date_utc, end_date_utc, fstH, fn_fcst, fn_obs, fn_error): 
    """
    站点预报的检验
    """
    idate_utc = start_date_utc
    station = meb.read_station(meb.station_国家站)
    station['data0'] = np.nan 

    ndays = (end_date_utc - start_date_utc).days + 1
    
    errors = deepcopy(station)
    errors['data0'] = 0

    while idate_utc <= end_date_utc: 
        ifn_fcst = fn_fcst.format(t=idate_utc, fh=fstH)
        ifn_obs = fn_obs.format(t=idate_utc+timedelta(hours=fstH+8))

        fst_s = meb.read_stadata_from_micaps3(ifn_fcst, station=station)
        obs_s = meb.read_stadata_from_micaps3(ifn_obs, station=station)
        
        istr = '{t:%Y%m%d%H}_{fh:03d}'.format(t=idate_utc, fh=fstH)
        if (fst_s is None) or (obs_s is None): 
            errors[istr] = np.nan 
        else: 
            errors[istr] = fst_s['data0'].values - obs_s['data0'].values
        
        idate_utc += timedelta(hours=24)

    # 输出 为 csv 格式
    errors.to_csv(fn_error, sep=',', na_rep='999999', float_format='%.2f', index=False)


def err_cal_mae(fn_errs: list, descriptions: list):
    """
    误差分析
    """
    for idis, ifn in zip(descriptions, fn_errs): 
        raw_mat = pd.read_csv(ifn, )
        err_mat = raw_mat.iloc[:, 8:].values 
        err_mat[err_mat > 9990] = np.nan 
        print('%s MAE: %.2f' % (idis, np.nanmean(np.abs(err_mat))))


def err_analysis_err_dis_dif_sum(fn_err_src1, fn_err_src2, fn_err_bld, fn_output):  
    """
    计算 不同模式 误差分布函数 的差距，并将该差距 输出
    """
    raw_mat1 = pd.read_csv(fn_err_src1)
    err_mat1 = raw_mat1.iloc[:, 8:].values 
    err_mat1[err_mat1 > 9990] = np.nan 
    
    raw_mat2 = pd.read_csv(fn_err_src2)
    err_mat2 = raw_mat2.iloc[:, 8:].values 
    err_mat2[err_mat2 > 9990] = np.nan 

    raw_mat_bld = pd.read_csv(fn_err_bld)
    err_mat_bld = raw_mat_bld.iloc[:, 8:].values 
    err_mat_bld[err_mat_bld > 9990] = np.nan 

    # station id 
    st_ids = raw_mat1.iloc[:, 3].values
    st_err_dif = np.zeros((len(st_ids), 12))
    st_err_dif[:, 0] = st_ids

    for i in range(len(st_ids)): 
        stid = st_ids[i]
        mae1 = np.nanmean(np.abs(err_mat1[i, :]))
        mae2 = np.nanmean(np.abs(err_mat2[i, :]))
        mae_bld = np.nanmean(np.abs(err_mat_bld[i, :]))

        if (mae_bld <= mae1 ) and (mae_bld <= mae2): 
            ifn_pic = './pics/bld_good/ec_ncep_station_%d_pdf.png' % stid
        else: 
            if mae1 <= mae2:
                ifn_pic = './pics/ec_good/ec_ncep_station_%d_pdf.png' % stid
            else: 
                ifn_pic = './pics/ncep_good/ec_ncep_station_%d_pdf.png' % stid

        f1 = stats.relfreq(err_mat1[i, :], numbins=60, defaultreallimits=(-15, 15))
        f2 = stats.relfreq(err_mat2[i, :], numbins=60, defaultreallimits=(-15, 15))
        pdf1 = f1.frequency/f1.binsize
        pdf2 = f2.frequency/f1.binsize 
        x = f1.lowerlimit + np.linspace(0, f1.binsize*f1.frequency.size, f1.frequency.size)
        
        # 两个 pdf 曲线的差值，可以视同为 两个 pdf 曲线之间的面积
        st_err_dif[i, 1] = np.sum(np.abs(pdf1-pdf2))*f1.binsize
        
        # 拟合 正态分布 曲线
        fit1 = fitter.Fitter(err_mat1[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit2 = fitter.Fitter(err_mat2[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit3 = fitter.Fitter(err_mat_bld[i, :], xmin=-15, xmax=15, distributions=['norm'])
        fit1.fit()
        fit2.fit()
        fit3.fit()
        par1 = fit1.get_best()['norm']
        par2 = fit2.get_best()['norm']
        par3 = fit3.get_best()['norm']
        
        # 根据参数 绘图
        '''
        title1 = 'MAE ec: %.2f, ncep: %.2f, blend: %.2f' % (mae1, mae2, mae_bld)
        norm1 = stats.norm(loc=par1['loc'], scale=par1['scale'])
        norm2 = stats.norm(loc=par2['loc'], scale=par2['scale'])
        norm3 = stats.norm(loc=par3['loc'], scale=par3['scale'])
        plt.plot(x, pdf1, 'rx-', label='ecmwf raw')
        plt.plot(x, norm1.pdf(x), 'rx--', label='ecmwf norm fit: \nloc: %.2f, scale: %.2f'%(par1['loc'], par1['scale']))
        plt.plot(x, pdf2, 'b+-', label=' ncep raw')
        plt.plot(x, norm2.pdf(x), 'b+--', label=' ncep norm fit: \nloc: %.2f, scale: %.2f'%(par2['loc'], par2['scale']))
        plt.plot(x, norm3.pdf(x), 'k--', label ='blend norm fitL \nloc: %.2f, scale: %.2f'%(par3['loc'], par3['scale']))
        plt.legend()
        plt.title(title1)
        plt.savefig(ifn_pic)
        plt.close()
        '''
        # ec, ncep, blend 拟合正态分布曲线的参数： loc, scale, 
        # ec
        st_err_dif[i, 2] = par1['loc']
        st_err_dif[i, 3] = par1['scale']
        st_err_dif[i, 4] = mae1
        # ncep
        st_err_dif[i, 5] = par2['loc']
        st_err_dif[i, 6] = par2['scale']
        st_err_dif[i, 7] = mae2 
        # blend 
        st_err_dif[i, 8] = par3['loc']
        st_err_dif[i, 9] = par3['scale']
        st_err_dif[i, 10] = mae_bld
        # corrcoef
        ec_fst = err_mat1[i, :]
        ncep_fst = err_mat2[i, :]
        ec_fst[np.isnan(ec_fst)] = 0
        ncep_fst[np.isnan(ncep_fst)] = 0
        st_err_dif[i, 11] = np.corrcoef(ec_fst, ncep_fst)[0, 1]

    # 对 st_err_dif, 按 pdf 误差 从大到小 排序
    idx = np.argsort(st_err_dif[:, 1])[::-1]
    st_err_dif2 = st_err_dif[idx]
    header1 = 'stid, ec_ncep_pdf_dif, ec_norm_loc, ec_norm_scale, ec_mae, ncep_norm_loc, ncep_norm_scale, ncep_mae, blend_norm_loc, blend_norm_scale, blend_mae, ec_ncep_corrcoef'
    np.savetxt(fn_output, st_err_dif2, fmt='%d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f', header=header1)


def err_analysis_mae_estimate(fn_input):
    """
    对 融合后的 mae 进行 估计
    """
    dmat = np.loadtxt(fn_input, )
    data_df = pd.DataFrame(dmat, columns=['stid', 'pdf_dif', 
                                          'ec_loc', 'ec_scale', 'ec_mae', 
                                          'ncep_loc', 'ncep_scale', 'ncep_mae', 
                                          'blend_loc', 'blend_scale', 'blend_mae', 'corrcoef'])

    data_df.dropna(axis=0, inplace=True)
    a1 = np.sqrt(2/np.pi)
    a2 = 1/np.sqrt(2*np.pi)
    
    # estimate mae and real mae
    mae_ec_theoretically = a1*data_df['ec_scale'].values + a2*(data_df['ec_loc'].values**2)/data_df['ec_scale'].values
    mae_ec_real = data_df['ec_mae'].values
    mae_ncep_theoretically = a1*data_df['blend_scale'].values + a2*(data_df['blend_loc'].values**2)/data_df['blend_scale'].values
    mae_ncep_real = data_df['blend_mae'].values

    xmin = max(0, min(np.min(mae_ec_theoretically), np.min(mae_ncep_theoretically)) - 0.1)
    xmax = max(np.max(mae_ec_theoretically), np.max(mae_ncep_theoretically)) + 0.1 
    x = np.linspace(xmin, xmax, 100)

    # 回归
    reg_ec = stats.linregress(mae_ec_theoretically, mae_ec_real)
    a1 = reg_ec.slope
    b1 = reg_ec.intercept
    y1 = a1*x +b1

    reg_ncep = stats.linregress(mae_ncep_theoretically, mae_ncep_real)
    a2 = reg_ncep.slope
    b2 = reg_ncep.intercept
    y2 = a2*x + b2

    # estimate blending sigma 
    aa1 = np.power(1/mae_ec_real, 2)
    aa2 = np.power(1/mae_ncep_real, 2)
    w1 = aa1 / (aa1+aa2)
    w2 = 1 - w1 

    sig_ec = data_df['ec_scale'].values
    sig_ncep = data_df['ncep_scale'].values
    sig_bld_real = data_df['blend_scale'].values
    corrcoef = data_df['corrcoef'].values
    sig_bld_estimate = np.sqrt(w1*w1*sig_ec*sig_ec + w2*w2*sig_ncep*sig_ncep + 2*w1*w2*sig_ec*sig_ncep*corrcoef)
    
    reg_bld = stats.linregress(sig_bld_estimate, sig_bld_real)
    a3 = reg_bld.slope
    b3 = reg_bld.intercept
    xx = np.linspace(0, 4, 100)
    yy = a3*xx +b3

    # 绘图 
    fig = plt.subplot(131)
    plt.plot(mae_ec_theoretically, mae_ec_real, 'rx', label='ec raw data')
    plt.plot(x, y1, 'k--', label='curve fit: y = %.3fx + %.3f' % (a1, b1))
    plt.axis('equal')
    plt.legend()
    plt.subplot(132)
    plt.plot(mae_ncep_theoretically, mae_ncep_real, 'g+', label='ncep raw data')
    plt.plot(x, y2, 'k--', label='curve fit: y = %.3fx + %.3f' % (a2, b2))
    plt.axis('equal')
    plt.legend()
    plt.subplot(133)
    plt.plot(sig_bld_estimate, sig_bld_real, 'r+', label='blend raw data')
    plt.plot(xx, yy, 'k--', label='curve fit: y = %.3fx + %.3f' % (a3, b3))
    plt.axis('equal')
    plt.legend()
    plt.show()




def model_fcst_dependent_check(fn_err_src1, fn_err_src2, fn_cov): 
    """
    不同模式预报的 独立性 检验
    """
    raw_mat1 = pd.read_csv(fn_err_src1)
    err_mat1 = raw_mat1.iloc[:, 8:].values 
    err_mat1[err_mat1 > 9990] = np.nan 
    err_mat1[np.isnan(err_mat1)] = 0.0

    raw_mat2 = pd.read_csv(fn_err_src2)
    err_mat2 = raw_mat2.iloc[:, 8:].values 
    err_mat2[err_mat2 > 9990] = np.nan 
    err_mat2[np.isnan(err_mat2)] = 0.0

    # station id 
    st_ids = raw_mat1.iloc[:, 3].values
    st_err_corrcoef = np.zeros((len(st_ids), 2))
    st_err_corrcoef[:, 0] = st_ids

    for i in range(len(st_ids)): 
        stid = st_ids[i]
        corrcoef = np.corrcoef(err_mat1[i, :], err_mat2[i, :])
        st_err_corrcoef[i, 1] = corrcoef[0, 1]

    np.savetxt(fn_cov, st_err_corrcoef, '%d %8.4f', header='stid, ec-ncep cov')

    # 统计 cov 分布
    f1 = stats.relfreq(st_err_corrcoef[:, 1], numbins=50, defaultreallimits=[-1, 1])
    pdf1 = f1.frequency/f1.binsize
    x = f1.lowerlimit + np.linspace(0, f1.binsize*f1.frequency.size, f1.frequency.size)

    plt.plot(x, pdf1, 'k+-')
    plt.show()



def fftconvolve_test(): 
    """
    测试 卷积
    """
    uniform_dist = stats.uniform(loc=2, scale=3)
    std = 0.25
    normal_dist = stats.norm(loc=0, scale=std)

    delta = 1e-4
    big_grid = np.arange(-10,10,delta)

    pmf1 = uniform_dist.pdf(big_grid)*delta
    print("Sum of uniform pmf: "+str(sum(pmf1)))

    pmf2 = normal_dist.pdf(big_grid)*delta
    print("Sum of normal pmf: "+str(sum(pmf2)))


    conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
    print("Sum of convoluted pmf: "+str(sum(conv_pmf)))

    pdf1 = pmf1/delta
    pdf2 = pmf2/delta
    conv_pdf = conv_pmf/delta
    print("Integration of convoluted pdf: " + str(np.trapz(conv_pdf, big_grid)))


    plt.plot(big_grid,pdf1, label='Uniform')
    plt.plot(big_grid,pdf2, label='Gaussian')
    plt.plot(big_grid,conv_pdf, label='Sum')
    plt.legend(loc='best'), plt.suptitle('PDFs')
    plt.show()


def bld_tst(): 
    mae1 = 1.68
    mae2 = 1.43
    m1 = -0.05
    m2 = -0.04
    sig1 = 2.33
    sig2 = 1.61
    sig_bld = 1.72

    a1 = np.power(1/sig1, 2)
    a2 = np.power(1/sig2, 2)
    w1 = a1 / (a1+a2)
    w2 = 1 - w1 
    
    sig_exp = np.sqrt(w1*w1*sig1*sig1 + w2*w2*sig2*sig2)
    print('sigma expect: ', sig_exp)
    print('factor: %.2f' % (sig_bld/sig_exp))


if __name__ == "__main__": 
    # 站点预报的误差数据
    fn_err_t2m_ecmwf_bced_s = './results/ecmwf_BC_fcst_err.csv'
    fn_err_t2m_ncep_bced_s = './results/ncep_BC_fcst_err.csv'
    fn_err_t2m_cmagfs_bced_s = './results/cmagfs_BC_fcst_err.csv'
    fn_bld_ec_ncep_basic_err = './results/bld_ec_ncep_basic_fcst_err.csv'

    # 
    fn_ec_ncep_err_dis_pdf_dif_sum = './results/ec_ncep_pdf_dif_sort.txt'
    fn_pic_pdf = './pics/ec_ncep_station_{id:d}_pdf.png'
    #err_analysis_err_dis_dif_sum(fn_err_t2m_ecmwf_bced_s, fn_err_t2m_ncep_bced_s, fn_bld_ec_ncep_basic_err,
    #                             fn_ec_ncep_err_dis_pdf_dif_sum)
    #bld_tst()
    
    fn_err_ecmwf_ncep_corrcoef = './results/ec_ncep_bced_corrcoef.txt'
    fn_err_ecmwf_cma_corrcoef = './results/ec_cma_bced_corrcoef.txt'
    fn_err_ncep_cma_corrcoef = './results/ncep_cma_bced_corrcoef.txt'
    model_fcst_dependent_check(fn_err_t2m_ecmwf_bced_s, fn_err_t2m_ncep_bced_s, fn_err_ecmwf_ncep_corrcoef)
    model_fcst_dependent_check(fn_err_t2m_ecmwf_bced_s, fn_err_t2m_cmagfs_bced_s, fn_err_ecmwf_cma_corrcoef)
    model_fcst_dependent_check(fn_err_t2m_ncep_bced_s, fn_err_t2m_cmagfs_bced_s, fn_err_ncep_cma_corrcoef)
    
    # 验证 mae, sigma 的公式是否正确
    #err_analysis_mae_estimate(fn_ec_ncep_err_dis_pdf_dif_sum)

    #fftconvolve_test()