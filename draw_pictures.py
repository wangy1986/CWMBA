# -*- coding: utf-8 -*-
# !/usr/bin/env python


# 绘图部分
import cmath
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mpl
import matplotlib.pyplot as plt

# 字体管理
from matplotlib import font_manager
# 设置tick标签格式
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import custom color bar
from palettable.cmocean import sequential as pcmseq
from palettable.cartocolors import sequential as pccmseq
import numpy as np
import pandas as pd
from common_func import check_fn_available

shp_prov = Reader('f:/Data/shape_file/Province.shp')
shp_cn = Reader('f:/Data/shape_file/NationalBorder.shp')

def show_2D_uvfield(u_mat, v_mat, lons, lats, archive_fn, contours_lonlat=None, 
    b_show=True, titles=None, local_marks_dict=None, b_show_nine_lines=False):
    """
    绘制二维 uv wind field

    ----------
    Parameters
    ----------
    u_mat:              u风向数据矩阵 = [nrows, ncols]
    v_mat:              v风向数据矩阵 = [nrows, ncols]

    lons:               longitude list = [ncols]
    lats:               latitude list = [nrows]
    
    archive_fn:         图片存储路径

    clvls:              counterf 绘图时使用的等值线
                        如果不指定，则使用 [0.1, 2,  4,  8,  12, 16, 20, 30, 40, 50, 999]

    contours_lonlat:    轮廓数据，list
                        = [contour1, contour2, ...]
                        contour_i 是 np.ndarray, shape = [npoints, 2]
                        contour_i = [ [lon1, lat1], 
                                      [lon2, lat2], 
                                      ...
                                      [lonn, latn] ]

    titles:             图片的title 
                        = { 'left':     ltitle, 
                            'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }

    local_marks_dict    用于在地图上绘制标记点
                        格式为：
                            '标记点的形状、颜色': [标记点的经纬度列表]]
                        = {
                            'bo':    [[lon1, lon2, ...], [lat1, lat2, ...]],  
                            'rx':    [[lonn, lonn+1, ...], [latn, latn+1, ...]
                        }

    b_show_nine_lines   是否绘制中国南海九段线, bool

    -------
    Returns
    -------
    None 
    """ 
    check_fn_available(archive_fn)

    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)

    u_mat = np.squeeze(u_mat)
    v_mat = np.squeeze(v_mat)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor = max(nor-sou, est-wst) / 10
    w = (est-wst) / scale_factor
    h = (nor-sou) / scale_factor
    fig = plt.figure(figsize=(w, h))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()
    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax = plt.axes([0.08, 0.1, 0.85, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep
    # 画线
    
    gl = ax.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                      xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                      draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    gl.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])

    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    #shp = Reader('E:/Data_set/shape_file/Province.shp')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.2, 0.2, 0.2)], linewidths=1, facecolor='none')
    shp_China = Reader('E:/Data_set/shape_file/NationalBorder.shp')
    ax.add_geometries(shp_China.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

    # 加入底图，效果见：https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
    # ax.stock_img() 
    # 至此，地图的底图部分设定完毕

    ### 2.a 绘制风场
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons, lats)
    handle = None
    handle = ax.barbs(llons, llats, u_mat, v_mat, barbcolor='b', flagcolor='b',
        barb_increments={'half':2,'full':4,'flag':20}, 
        sizes={'spacing': 0.2, 'height': 0.4, 'width': 0.25, 'emptybarb': 0.0}, 
        length=6, zorder=99)

    ### 2.b 在地图上添加标记点 - 目前是用于显示区域极值 
    if type(local_marks_dict) is dict:
        for ikey, lonlat_list in local_marks_dict.items(): 
            ax.plot(lonlat_list[0], lonlat_list[1], ikey, markersize=10+scale_factor, transform=proj)

    ### 2.c 添加轮廓线
    linestyle = ['-', '-']
    if type(contours_lonlat) is list: 
        for ii, icc in enumerate(contours_lonlat): 
            ilstyle = linestyle[ii%len(linestyle)]
            if icc.shape[0] > 1:
                ax.plot(icc[:, 0], icc[:, 1], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj)
                ax.plot([icc[-1, 0], icc[0, 0]], [icc[-1, 1], icc[0, 1]], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj)
            else: 
                pass
                # ax.plot(icc[0, 0], icc[0, 1], 'kx', markersize=10+scale_factor, transform=proj)  

    ### 3. set titles 
    if type(titles) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor}
        if 'center' in titles:
            plt.title(titles['center'], fontdict=ifont, loc='center')
        if 'left' in titles: 
            plt.title(titles['left'], fontdict=ifont, loc='left')
        if 'right' in titles: 
            plt.title(titles['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    ### 5 添加南海九段线
    if b_show_nine_lines: 
        ax_9line = plt.axes([0.678, 0.1, 0.28, 0.3], projection=proj)
        # 设置绘图的地理范围
        ax_9line.set_extent([107, 123, 2, 27], crs=proj)
        shp_9 = Reader('E:/Data_set/shape_file/NationalBorder.shp')
        ax_9line.add_geometries(shp_9.geometries(), crs=proj, edgecolor=[(0.1, 0.1, 0.1)], linewidths=0.5, facecolor='none')

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)

# end of show_2D_uvfield


def show_2D_uvfield_add_mat(u_mat, v_mat, c_mat, lons, lats, archive_fn, 
    c_map=pccmseq.agSunset_7.mpl_colormap, 
    c_lvls=None,
    contours_lonlat=None, 
    b_show=True, titles=None, local_marks_dict=None, b_show_nine_lines=False):
    """
    绘制二维 uv wind field， 并叠加一个shading 底图

    ----------
    Parameters
    ----------
    u_mat:              u风向数据矩阵 = [nrows, ncols]
    v_mat:              v风向数据矩阵 = [nrows, ncols]
    c_mat:              作为底图叠加到风场之下 = [nrows, ncols]

    lons:               longitude list = [ncols]
    lats:               latitude list = [nrows]
    
    archive_fn:         图片存储路径

    clvls:              counterf 绘图时使用的等值线
                        如果不指定，则使用 [0.1, 2,  4,  8,  12, 16, 20, 30, 40, 50, 999]

    cmap:               c_mat 绘图使用的色标
    contours_lonlat:    轮廓数据，list
                        = [contour1, contour2, ...]
                        contour_i 是 np.ndarray, shape = [npoints, 2]
                        contour_i = [ [lon1, lat1], 
                                      [lon2, lat2], 
                                      ...
                                      [lonn, latn] ]

    titles:             图片的title 
                        = { 'left':     ltitle, 
                            'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }

    local_marks_dict    用于在地图上绘制标记点
                        格式为：
                            '标记点的形状、颜色': [标记点的经纬度列表]]
                        = {
                            'bo':    [[lon1, lon2, ...], [lat1, lat2, ...]],  
                            'rx':    [[lonn, lonn+1, ...], [latn, latn+1, ...]
                        }

    b_show_nine_lines   是否绘制中国南海九段线, bool

    -------
    Returns
    -------
    None 
    """ 
    check_fn_available(archive_fn)

    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor = max(nor-sou, est-wst) / 10
    w = (est-wst) / scale_factor
    h = (nor-sou) / scale_factor
    fig = plt.figure(figsize=(w, h))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()
    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax = plt.axes([0.08, 0.1, 0.8, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep
    # 画线
    
    gl = ax.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                      xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                      draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    gl.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])

    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    #shp = Reader('E:/Data_set/shape_file/Province.shp')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.2, 0.2, 0.2)], linewidths=1, facecolor='none')
    shp_China = Reader('E:/Data_set/shape_file/NationalBorder.shp')
    ax.add_geometries(shp_China.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

    # 加入底图，效果见：https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
    # ax.stock_img() 
    # 至此，地图的底图部分设定完毕

    ### 2.a 绘制风场
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons, lats)
    handle = None
    handle = ax.barbs(llons, llats, u_mat, v_mat, barbcolor='b', flagcolor='b',
        barb_increments={'half':2,'full':4,'flag':20}, 
        sizes={'spacing': 0.2, 'height': 0.4, 'width': 0.25, 'emptybarb': 0.0}, 
        length=6, zorder=99)

    ### 2.aa 绘制 shading
    handle_c = None
    if c_mat is not None: 
        if c_lvls is None: 
            max_val = np.nanmax(c_mat)
            min_val = np.nanmin(c_mat)
            dstep = (max_val - min_val) / 20
            c_lvls = np.arange(min_val, max_val+0.1*dstep, dstep)

        c_map.set_over('k')
        handle_c = ax.contourf(llons, llats, c_mat, levels=c_lvls, 
                         norm = mpl.colors.BoundaryNorm(c_lvls, ncolors=c_map.N), 
                         cmap=c_map, zorder=-1, extend='max',
                         transform=proj)

    ### 2.b 在地图上添加标记点 - 目前是用于显示区域极值 
    if type(local_marks_dict) is dict:
        for ikey, lonlat_list in local_marks_dict.items(): 
            if (len(lonlat_list[0]) == 0) or (len(lonlat_list[1]) == 0): 
                continue
            ax.plot(lonlat_list[0], lonlat_list[1], ikey, markersize=10+scale_factor, 
                    transform=proj, zorder=999)

    ### 2.c 添加轮廓线
    linestyle = ['-', '-']
    if type(contours_lonlat) is list: 
        for ii, icc in enumerate(contours_lonlat): 
            ilstyle = linestyle[ii%len(linestyle)]
            if icc.shape[0] > 1:
                ax.plot(icc[:, 0], icc[:, 1], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj, zorder=1000)
                ax.plot([icc[-1, 0], icc[0, 0]], [icc[-1, 1], icc[0, 1]], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj, zorder=1000)
            else: 
                pass
                # ax.plot(icc[0, 0], icc[0, 1], 'kx', markersize=10+scale_factor, transform=proj)  

    ### 3. set titles 
    if type(titles) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor}
        if 'center' in titles:
            plt.title(titles['center'], fontdict=ifont, loc='center')
        if 'left' in titles: 
            plt.title(titles['left'], fontdict=ifont, loc='left')
        if 'right' in titles: 
            plt.title(titles['right'], fontdict=ifont, loc='right')
    else: 
        pass

    ### 4. 绘制色标
    if handle_c is not None: 
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14+scale_factor
        cbp1 = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar1 = plt.colorbar(handle_c, ticks=c_lvls, format='%.2f', cax=cbp1, orientation='vertical')
        # cbar1.ax.set_xlabel('QPE 01H (mm)')
    
    ### 5 添加南海九段线
    if b_show_nine_lines: 
        ax_9line = plt.axes([0.678, 0.1, 0.28, 0.3], projection=proj)
        # 设置绘图的地理范围
        ax_9line.set_extent([107, 123, 2, 27], crs=proj)
        shp_9 = Reader('E:/Data_set/shape_file/NationalBorder.shp')
        ax_9line.add_geometries(shp_9.geometries(), crs=proj, edgecolor=[(0.1, 0.1, 0.1)], linewidths=0.5, facecolor='none')

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)

# end of show_2D_uvfield


def show_2D_mat(dmat, lons, lats, 
                archive_fn, 
                b_show=True, 
                clvls=[0.1, 2,  4,  8,  12, 16, 20, 30, 40, 50, 999],
                contours_lonlat=None,  
                cmap=pccmseq.agSunset_7.mpl_colormap, 
                titles=None, 
                local_marks_dict=None, 
                b_show_nine_lines=False): 
    """
    简单的绘图函数

    ----------
    Parameters
    ----------
    dmat:               数据矩阵 = [nrows, ncols]
    lons:               longitude list = [ncols]
    lats:               latitude list = [nrows]
    archive_fn:         图片存储路径

    clvls:              counterf 绘图时使用的等值线
                        如果不指定，则使用 [0.1, 2,  4,  8,  12, 16, 20, 30, 40, 50, 999]

    contours_lonlat:    轮廓数据，list
                        = [contour1, contour2, ...]
                        contour_i 是 np.ndarray, shape = [npoints, 2]
                        contour_i = [ [lon1, lat1], 
                                      [lon2, lat2], 
                                      ...
                                      [lonn, latn] ]

    cmap:               使用的色标
                        如果不指定，则使用 pccmseq.agSunset_7.mpl_colormap

    titles:             图片的title 
                        = { 'left':     ltitle, 
                            'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }

    local_marks_dict    用于在地图上绘制标记点
                        格式为：
                            '标记点的形状、颜色': [标记点的经纬度列表]]
                        = {
                            'bo':    [[lon1, lon2, ...], [lat1, lat2, ...]],  
                            'rx':    [[lonn, lonn+1, ...], [latn, latn+1, ...]
                        }

    b_show_nine_lines   是否绘制中国南海九段线, bool

    -------
    Returns
    -------
    None
    """
    check_fn_available(archive_fn)
    dmat = np.squeeze(dmat)
    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor = max(nor-sou, est-wst) / 10
    w = (est-wst) / scale_factor
    h = (nor-sou) / scale_factor
    fig = plt.figure(figsize=(w, h))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()
    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax = plt.axes([0.08, 0.1, 0.8, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 绘制填色图
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons, lats)

    # cmap.colorbar_extend=True
    cmap.set_over('k')
    # 绘制离散色标填色图
    handle = None
    
    handle = ax.contourf(llons, llats, dmat, levels=clvls, 
                         norm = mpl.colors.BoundaryNorm(clvls, ncolors=cmap.N), 
                         cmap=cmap, zorder=1, extend='max',
                         transform=proj)
    
    ### 2.b 在地图上添加标记点 - 目前是用于显示区域极值 
    if type(local_marks_dict) is dict:
        for ikey, lonlat_list in local_marks_dict.items(): 
            ax.plot(lonlat_list[0], lonlat_list[1], ikey, markersize=10+scale_factor, transform=proj)

    ### 2.c 添加轮廓线
    linestyle = ['-', '-']
    if type(contours_lonlat) is list: 
        for ii, icc in enumerate(contours_lonlat): 
            ilstyle = linestyle[ii%len(linestyle)]
            if icc.shape[0] > 1:
                ax.plot(icc[:, 0], icc[:, 1], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj)
                ax.plot([icc[-1, 0], icc[0, 0]], [icc[-1, 1], icc[0, 1]], 'k', linestyle=ilstyle, linewidth=2.0, transform=proj)
            else: 
                pass
                # ax.plot(icc[0, 0], icc[0, 1], 'kx', markersize=10+scale_factor, transform=proj)  

    ### 3. set titles 
    if type(titles) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor}
        if 'center' in titles:
            plt.title(titles['center'], fontdict=ifont, loc='center')
        if 'left' in titles: 
            plt.title(titles['left'], fontdict=ifont, loc='left')
        if 'right' in titles: 
            plt.title(titles['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    ### 4. 绘制色标
    if handle is not None: 
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14+scale_factor
        cbp1 = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar1 = plt.colorbar(handle, ticks=clvls, format='%.2f', cax=cbp1, orientation='vertical')
        # cbar1.ax.set_xlabel('QPE 01H (mm)')

    # 画线
    
    gl = ax.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                      xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                      draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    gl.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])
    #ax.add_feature(cfeature.RIVERS, linewidth=2)
    # ax.add_feature(cfeature.LAKES)

    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    shp = Reader('E:/Data_set/shape_file/Province.shp')
    ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.2, 0.2, 0.2)], linewidths=1, facecolor='none')

    shp_China = Reader('E:/Data_set/shape_file/NationalBorder.shp')
    ax.add_geometries(shp_China.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')
    
    # 加入底图，效果见：https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
    # ax.stock_img() 
    # 至此，地图的底图部分设定完毕
    
    ### 5 添加南海九段线
    if b_show_nine_lines: 
        ax_9line = plt.axes([0.678, 0.1, 0.28, 0.3], projection=proj)
        # 设置绘图的地理范围
        ax_9line.set_extent([107, 123, 2, 27], crs=proj)
        shp_9 = Reader('E:/Data_set/shape_file/NationalBorder.shp')
        ax_9line.add_geometries(shp_9.geometries(), crs=proj, edgecolor=[(0.1, 0.1, 0.1)], linewidths=0.5, facecolor='none')

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)


def _draw_grid_contour(ax, dmat, lons, lats, cmap, clvls, titles, proj, scale_factor): 
    """
    绘图基础组件1
    绘制 网格数据的 contourf
    """
    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)
    # 设置绘图的地理范围
    ax.set_extent([wst, est, sou, nor], crs=proj)
    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 绘制填色图
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons, lats)

    # cmap.colorbar_extend=True
    cmap.set_over('k')
    # 绘制离散色标填色图
    handle = None
    dmat[dmat>clvls[-1]] = clvls[-1]
    handle = ax.contourf(llons, llats, dmat, levels=clvls, 
                         norm = mpl.colors.BoundaryNorm(clvls, ncolors=cmap.N), 
                         cmap=cmap, zorder=1, extend='max',
                         transform=proj)
    
    ### 3. set titles 
    if type(titles) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor}
        if 'center' in titles:
            plt.title(titles['center'], fontdict=ifont, loc='center')
        if 'left' in titles: 
            plt.title(titles['left'], fontdict=ifont, loc='left')
        if 'right' in titles: 
            plt.title(titles['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    # 绘制经纬度虚线
    gl = ax.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                      xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                      draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    gl.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])
    #ax.add_feature(cfeature.RIVERS, linewidth=2)
    # ax.add_feature(cfeature.LAKES)

    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    ax.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')
    return handle

def _draw_stat_scatter(ax, stat, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles, proj, scale_factor):
    """
    绘图基础组件2
    绘制 站点数据的散点分布图
    """
    # 设置绘图的地理范围
    ax.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 
    # 绘制离散色标填色图
    X = stat['lon'].values
    Y = stat['lat'].values
    r = stat[val_name].values
    #r[r>vmax] = vmax
    handle1 = ax.scatter(X, Y, c=r, vmin=vmin, vmax=vmax, cmap=cmap)

    ### 3. set titles 
    if type(titles) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor}
        if 'center' in titles:
            plt.title(titles['center'], fontdict=ifont, loc='center')
        if 'left' in titles: 
            plt.title(titles['left'], fontdict=ifont, loc='left')
        if 'right' in titles: 
            plt.title(titles['right'], fontdict=ifont, loc='right')
    else: 
        pass
    ### 5. 添加地图标志
    # 画线
    gl1 = ax.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    gl1.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor}
    
    ax.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')
    return handle1


def show_grid_stat_fcst_4pic(data1, data2, data3, data4, lons, lats, val_name, 
                             cmap, vmin, vmax, dval, 
                             titles1, titles2, titles3, titles4, 
                             colorbar_title, fn_archive, b_show=False): 
    """
    绘制 2*2 图像
    """
    if fn_archive is not None:
        check_fn_available(fn_archive)

    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor1 =int(max(nor-sou, est-wst) / 6.5)
    w1 = (est-wst) / scale_factor1
    h1 = (nor-sou) / scale_factor1

    fig = plt.figure(figsize=(2*w1, 2.1*h1))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()
    ifont = {'fontname': 'Times New Roman', 'fontsize': 20+scale_factor1}

    ############
    # subplop 1
    ############
    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    #ax1 = plt.axes([0.10, 0.12, 0.85, 0.85], projection=proj)
    ax1 = plt.axes([0.04, 0.55, 0.45, 0.4], projection=proj)
    if type(data1) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle1 = _draw_grid_contour(ax1, data1, lons, lats, cmap, clvls, titles1, proj, scale_factor1)
    elif type(data1) == pd.DataFrame:
        handle1 = _draw_stat_scatter(ax1, data1, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles1, proj, scale_factor1)
    ax1.text(118.5, 38.5, '(a)', fontdict=ifont, transform=proj)
    ############
    # subplop 2 - stat data 
    ############
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax2 = plt.axes([0.54, 0.55, 0.45, 0.4], projection=proj)
    if type(data2) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle2 = _draw_grid_contour(ax2, data2, lons, lats, cmap, clvls, titles2, proj, scale_factor1)
    elif type(data2) == pd.DataFrame:
        handle2 = _draw_stat_scatter(ax2, data2, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles2, proj, scale_factor1)
    ax2.text(118.5, 38.5, '(b)', fontdict=ifont, transform=proj)

    ############
    # subplop 3 - stat data 
    ############
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax3 = plt.axes([0.04, 0.1, 0.45, 0.4], projection=proj)
    if type(data3) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle3 = _draw_grid_contour(ax3, data3, lons, lats, cmap, clvls, titles3, proj, scale_factor1)
    elif type(data3) == pd.DataFrame:
        handle3 = _draw_stat_scatter(ax3, data3, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles3, proj, scale_factor1)
    ax3.text(118.5, 38.5, '(c)', fontdict=ifont, transform=proj)
    ############
    # subplop 4 - stat data 
    ############
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax4 = plt.axes([0.54, 0.1, 0.45, 0.4], projection=proj)
    if type(data4) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle4 = _draw_grid_contour(ax4, data4, lons, lats, cmap, clvls, titles4, proj, scale_factor1)
    elif type(data4) == pd.DataFrame:
        handle4 = _draw_stat_scatter(ax4, data4, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles4, proj, scale_factor1)
    ax4.text(118.5, 38.5, '(d)', fontdict=ifont, transform=proj)
    ############
    # subplot 5 - colorbar 
    ############
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14+scale_factor1
    cbp1 = fig.add_axes([0.1, 0.05, 0.8, 0.015])
    cbar1 = plt.colorbar(handle1, ticks=clvls, format='%d', cax=cbp1, orientation='horizontal')
    cbar1.ax.set_xlabel(colorbar_title)

    ### 6. save and close picture
    if fn_archive is not None: 
        plt.savefig(fn_archive)
        print('>> %s' % fn_archive)

    if b_show: 
        plt.show()
    
    plt.close()


def show_grid_stat_fcst_2pic(data1, data2, lons, lats, val_name, 
                             cmap, vmin, vmax, dval, 
                             titles1, titles2, 
                             colorbar_title, fn_archive, b_show=False): 
    """
    绘制1张图，左边为网格，右边为站点

    ------------------
    parameters
    ------------------
    dmat:               网格数据
    lons:               经度数据
    lats:               纬度数据
    stat:               pandas.dataframe 格式的站点数据
                        至少需要包括 'lon', 'lat', val_name 三列
    val_name:           指定 stat 的 哪一列作为 绘图数据
    
    cmap:               色标表
    vmin:               色标表的参数   
    vmax:               
    dval:

    titles1:            左图title
                        = { 'left':     ltitle, 
                                'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }
    titles2:            右图title，格式同 titles1
    colorbar_title:     色标表的参数

    fn_archive:         图片保存路径
    b_show:             是否显示在屏幕上
    ------------------
    returns
    ------------------
    """
    if fn_archive is not None:
        check_fn_available(fn_archive)

    nor = np.max(lats)
    sou = np.min(lats)
    wst = np.min(lons)
    est = np.max(lons)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor1 =int(max(nor-sou, est-wst) / 6.5)
    w1 = (est-wst) / scale_factor1
    h1 = (nor-sou) / scale_factor1

    fig = plt.figure(figsize=(2*w1, 1.1*h1))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()
    ifont = {'fontname': 'Times New Roman', 'fontsize': 20+scale_factor1}
    ############
    # subplop 1 - grid data 
    ############
    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    #ax1 = plt.axes([0.10, 0.12, 0.85, 0.85], projection=proj)
    ax1 = plt.axes([0.04, 0.15, 0.45, 0.8], projection=proj)
    if type(data1) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle1 = _draw_grid_contour(ax1, data1, lons, lats, cmap, clvls, titles1, proj, scale_factor1)
    elif type(data1) == pd.DataFrame:
        handle1 = _draw_stat_scatter(ax1, data1, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles1, proj, scale_factor1)
    ax1.text(118.5, 38.5, '(a)', fontdict=ifont, transform=proj)

    ############
    # subplop 2 - stat data 
    ############
    ax2 = plt.axes([0.54, 0.15, 0.45, 0.8], projection=proj)
    if type(data2) == np.ndarray:
        clvls = np.arange(vmin, vmax+0.5*dval, dval)
        handle2 = _draw_grid_contour(ax2, data2, lons, lats, cmap, clvls, titles2, proj, scale_factor1)
    elif type(data2) == pd.DataFrame:
        handle2 = _draw_stat_scatter(ax2, data2, val_name, vmin, vmax, cmap, nor, sou, wst, est, titles2, proj, scale_factor1)
    ax2.text(118.5, 38.5, '(b)', fontdict=ifont, transform=proj)
    
    ############
    # subplot 3 - colorbar 
    ############
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14+scale_factor1
    cbp1 = fig.add_axes([0.1, 0.08, 0.8, 0.03])
    clvls = np.arange(vmin, vmax+0.5*dval, dval)
    cbar1 = plt.colorbar(handle1, ticks=clvls, format='%.2f', cax=cbp1, orientation='horizontal')
    cbar1.ax.set_xlabel(colorbar_title)

    ### 6. save and close picture
    if fn_archive is not None: 
        plt.savefig(fn_archive)
        print('>> %s' % fn_archive)

    if b_show: 
        plt.show()
    
    plt.close()
    

def show_2D_scatter_1pic(
        stat1, stat1_cmap, titles1, val_name, 
        vmin, vmax, dval, colorbar_title, 
        nor, sou, wst, est, 
        archive_fn, 
        b_show=False, b_show_nine_lines=False): 
    """
    绘制1张 散点分布图
    -----------------
    Parameters
    -----------------
    stat1:              pandas.dataframe 格式的站点数据，至少需要包括 'lon', 'lat', 'val' 三列
    stat1_cmap:         绘制 stat1 时使用的色标
    titles1:            左侧图片的 titles
                        = { 'left':     ltitle, 
                                'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }
    val_name:           绘制 stat1 中的哪个要素

    vmin:               绘制散点图时，色标对应的 vmin, vmax, dval
    vmax:
    dval:
    colorbar_title:     色标表的名字

    nor:                绘图的地理范围，单位是经纬度
    sou:
    wst:
    est:

    archive_fn:         图片存储路径
    b_show:             是否在屏幕上显示图片
    -----------------
    Return
    -----------------
    None
    """
    check_fn_available(archive_fn)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor1 =int(max(nor-sou, est-wst) / 6.5)/2
    w1 = (est-wst) / scale_factor1
    h1 = (nor-sou) / scale_factor1

    fig = plt.figure(figsize=(w1, 1.1*h1))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()

    #################
    ### subplot 1 ###
    #################

    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax1 = plt.axes([0.10, 0.07, 0.85, 0.85], projection=proj)
    # 设置绘图的地理范围
    ax1.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 10
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 
    # 绘制离散色标填色图
    handle2 = None
    X = stat1['lon'].values
    Y = stat1['lat'].values
    r = stat1[val_name].values
    handle12 = ax1.scatter(X, Y, c=r, vmin=vmin, vmax=vmax, 
                           cmap=stat1_cmap)

    ### 3. set titles 
    if type(titles1) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor1}
        if 'center' in titles1:
            plt.title(titles1['center'], fontdict=ifont, loc='center')
        if 'left' in titles1: 
            plt.title(titles1['left'], fontdict=ifont, loc='left')
        if 'right' in titles1: 
            plt.title(titles1['right'], fontdict=ifont, loc='right')
    else: 
        pass

    ### 4. 绘制色标 # 右侧图片负责在整张图最下方绘制 colorbar
    if (handle12 is not None) and (colorbar_title is not None): 
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14+scale_factor1
        # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
        cbp1 = fig.add_axes([0.12, 0.08, 0.81, 0.03])
        cbar1 = plt.colorbar(handle12, format='%d', cax=cbp1, orientation='horizontal')
        cbar1.ax.set_xlabel(colorbar_title)

    ### 5. 添加地图标志
    # 画线
    gl1 = ax1.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    gl1.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])
    ax1.add_feature(cfeature.RIVERS, linewidth=2)
    # ax.add_feature(cfeature.LAKES)

    ax1.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax1.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

        ### 5 添加南海九段线
    if b_show_nine_lines: 
        ax_9line = plt.axes([0.73, 0.12, 0.28, 0.3], projection=proj)
        # 设置绘图的地理范围
        ax_9line.set_extent([107, 123, 2, 27], crs=proj)
        ax_9line.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.1, 0.1, 0.1)], linewidths=0.5, facecolor='none')

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)


def show_2D_scatter_2pic(
        stat1, stat1_cmap, titles1, 
        stat2, stat2_cmap, titles2, 
        vmin, vmax, dval, colorbar_title, 
        nor, sou, wst, est, 
        archive_fn, 
        b_show=False, ):
    """
    绘制散点图，左侧一副，右侧一副

    -----------------
    Parameters
    -----------------
    stat1:              pandas.dataframe 格式的站点数据，至少需要包括 'lon', 'lat', 'val' 三列
    stat1_cmap:         绘制 stat1 时使用的色标
    titles1:            左侧图片的 titles
                        = { 'left':     ltitle, 
                                'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }

    stat2:              pandas.dataframe 格式的站点数据，至少需要包括 'lon', 'lat', 'val' 三列
    stat2_cmap:         绘制 stat2 时使用的色标
    titles2:            右侧图片的 titles
                        = { 'left':     ltitle, 
                                'right':    rtitle, }
                        or 
                        = { 'center':   ctitle  }

    vmin:               绘制散点图时，色标对应的 vmin, vmax, dval
    vmax:
    dval:
    colorbar_title:     色标表的名字

    nor:                绘图的地理范围，单位是经纬度
    sou:
    wst:
    est:

    archive_fn:         图片存储路径
    b_show:             是否在屏幕上显示图片
    -----------------
    Return
    -----------------
    None
    """
    check_fn_available(archive_fn)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor1 =int(max(nor-sou, est-wst) / 6.5)
    w1 = (est-wst) / scale_factor1
    h1 = (nor-sou) / scale_factor1

    fig = plt.figure(figsize=(2*w1, 1.1*h1))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()

    #################
    ### subplot 1 ###
    #################

    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax1 = plt.axes([0.04, 0.15, 0.45, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax1.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 
    # 绘制离散色标填色图
    handle2 = None
    X = stat1['lon'].values
    Y = stat1['lat'].values
    r = stat1['val'].values
    handle12 = ax1.scatter(X, Y, c=r, vmin=vmin, vmax=vmax, 
                           cmap=stat1_cmap)

    ### 3. set titles 
    if type(titles1) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor1}
        if 'center' in titles1:
            plt.title(titles1['center'], fontdict=ifont, loc='center')
        if 'left' in titles1: 
            plt.title(titles1['left'], fontdict=ifont, loc='left')
        if 'right' in titles1: 
            plt.title(titles1['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    ### 4. 绘制色标 # 左侧图片不绘制 colorbar

    ### 5. 添加地图标志
    # 画线
    gl1 = ax1.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    gl1.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    
    ax1.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax1.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

    #################
    ### subplot 2 ###
    #################

    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax2 = plt.axes([0.54, 0.15, 0.45, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax1.set_extent([wst, est, sou, nor], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou-dstep)/dstep) * dstep
    lon_start = np.floor((wst-dstep)/dstep) * dstep

    ### 2.a 
    # 绘制离散色标填色图
    handle4 = None
    X = stat2['lon'].values
    Y = stat2['lat'].values
    r = stat2['val'].values
    handle4 = ax2.scatter(X, Y, c=r, vmin=vmin, vmax=vmax, 
                           cmap=stat2_cmap)

    ### 3. set titles 
    if type(titles2) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor1}
        if 'center' in titles2:
            plt.title(titles2['center'], fontdict=ifont, loc='center')
        if 'left' in titles2: 
            plt.title(titles2['left'], fontdict=ifont, loc='left')
        if 'right' in titles2: 
            plt.title(titles2['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    ### 4. 绘制色标 # 右侧图片负责在整张图最下方绘制 colorbar
    if handle4 is not None: 
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14+scale_factor1
        # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
        cbp1 = fig.add_axes([0.1, 0.08, 0.8, 0.03])
        cbar1 = plt.colorbar(handle4, format='%.2f', cax=cbp1, orientation='horizontal')
        cbar1.ax.set_xlabel(colorbar_title)

    ### 5. 添加地图标志
    # 画线
    gl2 = ax2.gridlines(ylocs=np.arange(lat_start, nor+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    gl2.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    
    ax2.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax2.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

    ###################
    ### subplot end ###
    ###################

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)


def show_2D_mat_2pic_with_obs(
                dmat1, lons1, lats1, clvls1, cmap1, title1, obs1, obs1_cmap, 
                dmat2, lons2, lats2, clvls2, cmap2, title2, obs2, obs2_cmap,
                archive_fn, 
                b_show=False): 
    """
    绘制两幅图片，左侧一副，右侧一副

    ----------
    Parameters
    ----------
    dmat1:              数据矩阵 = [nrows, ncols]
    lons1:              longitude list = [ncols]
    lats1:              latitude list = [nrows]
    clvls1:             counterf 绘图时使用的等值线
                            like: [0.1, 2,  4,  8,  12, 16, 20, 30, 40, 50, 999]
    cmap1:              使用的色标
                            like: pccmseq.agSunset_7.mpl_colormap
    title1:             图片的title 
                            = { 'left':     ltitle, 
                                'right':    rtitle, }
                            or 
                            = { 'center':   ctitle  }
    obs1:               pandas.dataframe 格式的实况数据
    obs1_cmap:          绘制 obs1 时使用的色标
    

    dmat2:              第二幅图的各种参数，同 xxx1
    lons2:
    lats2:
    clvls2:
    cmap2:
    title2:
    obs2:
    obs2_cmap:          


    archive_fn:         图片存储路径
    b_show:             是否在屏幕上显示图片

    -------
    Returns
    -------
    None
    """
    check_fn_available(archive_fn)
    dmat1 = np.squeeze(dmat1)
    dmat2 = np.squeeze(dmat2)
    nor1 = np.max(lats1)
    sou1 = np.min(lats1)
    wst1 = np.min(lons1)
    est1 = np.max(lons1)
    nor2 = np.max(lats2)
    sou2 = np.min(lats2)
    wst2 = np.min(lons2)
    est2 = np.max(lons2)

    ### 1. 设置底图
    # 基于经纬度计算合适的图片大小
    # set scale factor
    scale_factor1 = max(nor1-sou1, est1-wst1) / 10
    scale_factor2 = max(nor1-sou2, est1-wst2) / 10
    w1 = (est1-wst1) / scale_factor1
    h1 = (nor1-sou1) / scale_factor1

    w2 = (est2-wst2) / scale_factor2
    h2 = (nor2-sou2) / scale_factor2
    fig = plt.figure(figsize=(1*(w1+w2), 1.05*h1))

    # 设置地图投影格式
    proj = ccrs.PlateCarree()

    #################
    ### subplot 1 ###
    #################

    # 申请画布
    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax1 = plt.axes([0.04, 0.15, 0.4, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax1.set_extent([wst1, est1, sou1, nor1], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou1-dstep)/dstep) * dstep
    lon_start = np.floor((wst1-dstep)/dstep) * dstep

    ### 2.a 绘制填色图
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons1, lats1)

    # cmap.colorbar_extend=True
    cmap1.set_over('k')
    # 绘制离散色标填色图
    handle1 = None
    
    handle1 = ax1.contourf(llons, llats, dmat1, levels=clvls1, 
                         norm = mpl.colors.BoundaryNorm(clvls1, ncolors=cmap1.N), 
                         cmap=cmap1, zorder=1, extend='max',
                         transform=proj)
    X = obs1['lon'].values
    Y = obs1['lat'].values
    r = obs1['obs'].values
    handle12 = ax1.scatter(X, Y, c=r, vmin = 25.0, vmax=50.0, 
                           cmap=obs1_cmap)

    ### 3. set titles 
    if type(title1) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor1}
        if 'center' in title1:
            plt.title(title1['center'], fontdict=ifont, loc='center')
        if 'left' in title1: 
            plt.title(title1['left'], fontdict=ifont, loc='left')
        if 'right' in title1: 
            plt.title(title1['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    ### 4. 绘制色标
    if handle1 is not None: 
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14+scale_factor1
        # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
        cbp1 = fig.add_axes([0.04, 0.10, 0.4, 0.03])
        cbar1 = plt.colorbar(handle1, ticks=clvls1, format='%d', cax=cbp1, orientation='horizontal')
        cbar1.ax.set_xlabel('PQPF06 ( % )')

    if handle12 is not None: 
        cbp2 = fig.add_axes([0.54, 0.10, 0.4, 0.03])
        cbar2 = plt.colorbar(handle12, format = '%d', cax=cbp2, orientation='horizontal')
        cbar2.ax.set_xlabel('6H precipitation ( mm )')


    # 画线
    gl1 = ax1.gridlines(ylocs=np.arange(lat_start, nor1+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est1+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    gl1.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor1}
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    #ax1.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor=[(0, 0.0, 0.0)])
    #ax.add_feature(cfeature.RIVERS, linewidth=2)
    # ax.add_feature(cfeature.LAKES)
    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    ax1.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    #ax.add_geometries(shp.geometries(), crs=proj, edgecolor=[(0.2, 0.2, 0.2)], linewidths=1, facecolor='none')
    ax1.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')

    #################
    ### subplot 2 ###
    #################

    # [x1, y1, xx, yy] = [col_start, row_start, col_length, row_length]
    ax2 = plt.axes([0.54, 0.15, 0.4, 0.8], projection=proj)
    # 设置绘图的地理范围
    ax2.set_extent([wst2, est2, sou2, nor2], crs=proj)

    # 设置经纬度网格线的位置和标识
    # 这里的 10 标识经纬线的跨距
    dstep = 5
    lat_start = np.floor((sou2-dstep)/dstep) * dstep
    lon_start = np.floor((wst2-dstep)/dstep) * dstep

    ### 2.a 绘制填色图
    # 制作 contourf 所需的 经纬度矩阵
    llons, llats = np.meshgrid(lons2, lats2)

    # cmap.colorbar_extend=True
    cmap2.set_over('k')
    # 绘制离散色标填色图
    handle2 = None
    handle2 = ax2.contourf(llons, llats, dmat2, levels=clvls2, 
                         norm = mpl.colors.BoundaryNorm(clvls2, ncolors=cmap2.N), 
                         cmap=cmap2, zorder=1, extend='max',
                         transform=proj)

    X = obs2['lon'].values
    Y = obs2['lat'].values
    r = obs2['obs'].values
    handle22 = ax2.scatter(X, Y, c=r, vmin = 25.0, vmax=50.0, 
                           cmap=obs2_cmap)
    ### 3. set titles 
    if type(title2) is dict:
        ifont = {'fontname': 'Times New Roman', 'fontsize': 16+scale_factor2}
        if 'center' in title2:
            plt.title(title2['center'], fontdict=ifont, loc='center')
        if 'left' in title2: 
            plt.title(title2['left'], fontdict=ifont, loc='left')
        if 'right' in title2: 
            plt.title(title2['right'], fontdict=ifont, loc='right')
    else: 
        pass
    
    # 画线
    gl2 = ax2.gridlines(ylocs=np.arange(lat_start, nor2+0.5*dstep, dstep), 
                       xlocs=np.arange(lon_start, est2+0.5*dstep, dstep), 
                       draw_labels=True, linestyle='--', alpha=0.7)
    # 在经纬度线上加标识
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.xlabel_style = {'family': 'Times new roman', 'size': 16+scale_factor2}
    gl2.ylabel_style = {'family': 'Times new roman', 'size': 16+scale_factor2}
    
    # add land, ocean, coastline, rivers, lakes on the base-map
    # 这一部分文件是 cartopy 自带的地形文件，第一次使用时需要下载
    # 载入省界文件 - 可以使用 Micaps4 自带的 shp 文件
    ax2.add_geometries(shp_prov.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidths=0.5, facecolor='none')
    ax2.add_geometries(shp_cn.geometries(), crs=proj, edgecolor=[(0.4, 0.4, 0.4)], linewidth=1, facecolor='none')
    
    ###################
    ### subplot end ###
    ###################

    ### 6. save and close picture
    plt.savefig(archive_fn)
    if b_show: 
        plt.show()
    
    plt.close()
    print('>> %s' % archive_fn)