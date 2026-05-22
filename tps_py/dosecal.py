#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
版权说明：
    版权所有（c）2025，国科离子医疗科技有限公司，保留所有权利

版本号：1.0.0
生成日期：
作者：

修改日志：
    2024/12/18 - 李晶 - other: Scale不用fluence_map计算剂量
    2024/12/30 - 李晶 - other: Scale不用fluence_map计算剂量 - 剂量计算bug修改
    2025/03/06 - 李晶 - feat: 用fluence_map对voxel聚类 - 修改优化fluence_map计算
    2025/03/17 - 李晶 - feat: 用HU和frac对voxel聚类，优化过程降维
    2025/03/21 - 李鸿飞 - feat: 【算】软件加密, 注释问题函数
    2025/04/30 - 李晶 李鸿飞 - feat: 优化过程降维 - 将计算矩阵二范数集成进剂量引擎
"""

import gc
import csv
import os
import sys
import time
from concurrent.futures import (
    ALL_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from itertools import groupby
from tkinter import NS

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import hstack as csc_stack

from content.bm_operator.beamModel import BEAM_MODEL

sys.path.append("..")
sys.path.append("../..")

from content.tools.cudaPKG.cudaCalDose1 import (
    cuCalDose3,
    cuCalDoseNorm,
    cuFinalDoseAndRBEMap,
    cuRotate3DArray,
)

from content.tools.cudaPKG.cudaCalDoseRTD import (
    cuFinalDose,
)

from content.tools.cudaPKG.cudaCalWEQ import (
    cuCalWEQ,
    cuGetWeqForRegion,
    cuWeqToPhysPos,
)
from content.tools.cudaPKG.cudaGeometry import cuCalMinDisToRay
from content.tools.cudaPKG.cudaMemUtils import cuMemTestAlloc

# from content.tools.cudaPKG.cudaCalDose import cuCalDose
from content.tools.newcudaPKG.cudaCalDose import cuCalDose, cuPrepareWEQ
from scipy.interpolate import interpn

from content.base_operator.base_operator import BASE
from content.bm_operator.commissionTypes import PtclType
from content.geo_operator.geometry import (
    addMargin_GPU,
    getRotateMat,
    ifAInB,
    resample,
)
from content.geo_operator.grid import GRID
from content.tools.math_tools import calGaussiantwoRs
from content.tools.network.cashim_net import NetWork

# def setOrder(tmpidx, jj, indices, indptr):
#     j0 = indptr[jj]
#     j1 = indptr[jj+1]
#     # print("j0 ", j0)
#     tmpidx[j0:j1] = np.argsort(indices[j0:j1])+j0


class DOSECAL(BASE):
    """
    本py文件解决所有
        给定CT(density and stopping power weighted)
            grid类, 距离约化因子的值和此grid的空间朝向
        给定dose grid设置
            grid corner, grid resolution, grid number
            均为vector,坐标系 HFS dicom
        给定治疗床平移和旋转角度
            couch IEC coordinates
            x平移 +x R-->L
            y平移 +y I-->S
            z平移 +z P-->A
            旋转 绕AP轴 + HFS往右手边
            翻滚 绕RL轴 + HFS往左手边
            俯仰 绕IS轴 + HFS头抬起来
        给定束流设置
            beam angle <-- gantry angle
            isocener: HFS dicom 坐标系 (x,y,z)
    的dose计算
    输出包括：
        物理剂量: dosemap (nVoxels行, nBeamSpot列)稀疏矩阵
            nVoxels是target region voxel数目
            nBeamSpot是所有射野集的布点数目
        生物剂量: RBE map numpy array, dimension same as grid (TODO, now in rbecal.py)
        alpha, beta map numpy array, dimension same as grid
    """

    def __init__(self, serviceMode=False, infoFirst=False) -> None:
        super(DOSECAL, self).__init__(name="DOSECAL")
        if infoFirst:
            self.logger.info(
                "\n\
                ***********************\n\
                ALL units of angle are radiant\n\
                All units of distance, resolution are millimeter!!!\n\
                ***********************\n"
            )
        self.serviceMode = serviceMode
        self.task_id = "123"
        self.doseGridPosition = "HFS"

    def setDoseGrid(self, dims, corner, resolution, doseGridPosition, initdata=True):
        self.doseGrid = GRID(dims, initdata)
        self.doseGrid.setCorner(corner)
        self.doseGrid.setResolution(resolution)
        self.doseGridPosition = doseGridPosition

    def getDoseGrid(self, ct: GRID):
        self.logger.info("Resampling dose grid")
        # self.doseGrid.data = ct.resample(self.doseGrid.corner, \
        #                                  self.doseGrid.resolution, \
        #                                  self.doseGrid.dims,\
        #                                 self.doseGrid.orientation)
        self.doseGrid.data = resample(self.doseGrid, ct)

    def setGeometry(
        self,
        gantryAngle,
        couchAngle,
        translation,
        isocenter=[0, 0, 0],
        sad=6632,
    ):
        # self.infoFirst("geometry", "Angle in degree and distance in mm")
        self.gantryAngle = gantryAngle * 0.0174533
        self.couchAngle = np.array(couchAngle) * 0.0174533
        self.translation = np.array(translation)  # translation, couch IEC [x, y, z]
        self.translation = np.reshape(self.translation, (3, 1))
        self.isocenter = np.reshape(isocenter, (3, 1))
        # find source position, beam direction, beam x direction
        # in dicom coordinates

        # following steps are based on [x, y, z]
        # 1 rotate gantry
        bmdir = np.array([-np.sin(self.gantryAngle), np.cos(self.gantryAngle), 0])  # use dicom as reference coordinate
        bmdir = np.reshape(bmdir, (3, 1))
        bmdir = np.dot(self.doseGrid.orientation.T, bmdir)

        bmxdir = np.array([np.cos(self.gantryAngle), np.sin(self.gantryAngle), 0])
        bmxdir = np.reshape(bmxdir, (3, 1))
        bmxdir = np.dot(self.doseGrid.orientation.T, bmxdir)

        # 2 rotate couch
        # couchAngle in dicom y x z
        R1 = getRotateMat(-self.couchAngle[0], "y")  # Yaw
        # dicom y
        R2 = getRotateMat(self.couchAngle[1], "x")  # Pitch
        # dicom x
        R3 = getRotateMat(self.couchAngle[2], "z")  # Roll
        # dicom z

        # Note, rotation is applied on couch. Effect on beam was reversed --> R.T
        # Note Note, first rotation, then roll, then pitch angle R = R2.T*R3.T*R1.T (dicom coordinate)
        # Note Note Note, R may be written as R = (R2*R3*R1).T (coordinate frame not changed)
        # Note Note Note Note, we adopted dicom coordinate!!!

        R = np.matmul(R3.T, R1.T)
        R = np.matmul(R2.T, R)
        bmdir = np.dot(R, bmdir)
        bmxdir = np.dot(R, bmxdir)

        # difference between doseGrid orientation and gridPosition
        rotateMat = getRotateMat(0, "z")
        if self.doseGridPosition == "HFS":
            rotateMat = getRotateMat(0, "z")
        elif self.doseGridPosition == "HFP":
            rotateMat = getRotateMat(np.pi, "z")
        elif self.doseGridPosition == "FFS":
            rotateMat = getRotateMat(np.pi, "y")
        elif self.doseGridPosition == "FFP":
            rotateMat = getRotateMat(np.pi, "x")
        elif self.doseGridPosition == "HFDR":
            rotateMat = getRotateMat(-np.pi / 2, "z").T
        elif self.doseGridPosition == "HFDL":
            rotateMat = getRotateMat(np.pi / 2, "z").T
        elif self.doseGridPosition == "FFDR":
            rotateMat = (getRotateMat(np.pi, "y") @ getRotateMat(-np.pi / 2, "z")).T
        elif self.doseGridPosition == "FFDL":
            rotateMat = (getRotateMat(np.pi, "y") @ getRotateMat(np.pi / 2, "z")).T
        else:
            raise RuntimeError(
                f"Import failed due to an unsupported patient \
                               position({self.doseGridPosition}). Only HFP, HFS, \
                               FFP, FFS, HFDR, HFDL, FFDR or FFDL is supported."
            )
        self.bmdir = np.dot(rotateMat, bmdir)
        self.bmxdir = np.dot(rotateMat, bmxdir)

        self.bmydir = np.cross(self.bmdir, self.bmxdir, axis=0)

        # 3 translation
        source = np.reshape(isocenter, (3, 1)) - sad * self.bmdir
        source -= self.translation  # translation is applied on couch
        # effect on beam was reversed
        self.source = source

    def copyGeometry(self, tmpdc):
        self.source = tmpdc.source
        self.bmdir = tmpdc.bmdir
        self.bmxdir = tmpdc.bmxdir
        self.bmydir = tmpdc.bmydir
        self.isocenter = tmpdc.isocenter

    def getThetaMax(self, roiIndex, margin=5):
        # should be done with GPU later
        if np.size(margin) == 1:
            margin = np.ones((3, 1)) * margin
        newRegion = addMargin_GPU(margin, roiIndex, self.doseGrid)

        index = np.unravel_index(newRegion, self.doseGrid.data.shape, order="C")
        nPos = np.size(index) // 3
        index = np.reshape(np.array(index), (3, nPos))
        pos = self.doseGrid.getPos(index)

        relativePos = pos - self.source
        length = np.sqrt(np.sum(relativePos**2, 0, keepdims=True)).T

        halfVoxel = np.sqrt(np.sum(np.square(self.doseGrid.resolution))) * 0.5
        crossProduct = np.sqrt(
            np.sum(
                np.square(np.cross(relativePos, self.bmdir, axisa=0, axisb=0)),
                axis=1,
                keepdims=True,
            )
        )
        dum = (crossProduct + halfVoxel) / length
        dum[np.where(dum > 1)] = 1
        theta = np.arcsin(dum)

        return np.max(theta)

    def getRangeInBeamCoor(self, roiIndex, margin=5):
        if np.size(margin) == 1:
            margin = np.ones((3, 1)) * margin
        newRegion = addMargin_GPU(margin, roiIndex, self.doseGrid)

        index = np.unravel_index(newRegion, self.doseGrid.data.shape, order="C")
        nPos = np.size(index) // 3
        index = np.reshape(np.array(index), (3, nPos))
        pos = self.doseGrid.getPos(index)

        relativePos = pos - self.source

        xrange = np.dot(relativePos.T, np.reshape(self.bmxdir, (3,)))
        yrange = np.dot(relativePos.T, np.reshape(self.bmydir, (3,)))
        return np.array(
            [
                [np.min(xrange), np.min(yrange)],
                [np.max(xrange), np.max(yrange)],
            ]
        )

    def getWEQBetweenTwoPos(self, source, target, mode="value", minDis=6000, step=0.1):
        # mode: value or 0 means return weq value between source and target
        # mode: array or 1 means return weq value between source and each sampled point along source to target
        source = np.reshape(source, (3, 1))
        target = np.reshape(target, (3, 1))
        length = np.sqrt(np.sum((target - source) ** 2))
        if length == 0:
            if mode == "value" or mode == 0:
                return 0
            elif mode == "array" or mode == 1:
                return np.array([0])
            else:
                self.logger.error("getWEQBetweenTwoPos error mode")
                return 0

        if length < minDis:
            # self.infoFirst("minDis", "consider reduce minDis")
            minDis = 0
        tmpDir = (target - source) / length

        nSample = (length - minDis) // step
        iSample = np.arange(nSample)
        iPos = source + tmpDir * (iSample * step + minDis)

        density = self.doseGrid.getData(iPos)

        weq = 0
        if mode == "value" or mode == 0:
            weq = np.sum(density) * step
            if nSample != 0:
                weq += density[-1] * (length - minDis - step * nSample)
        elif mode == "array" or mode == 1:
            weq = np.cumsum(density) * step
            if nSample == 0:
                weq = np.array([0])
        else:
            self.logger.error("getWEQBetweenTwoPos error mode")

        return weq

    def getWEQForROI_GPU(self, roiIndex, margin=5):
        # get water equivalent length for all voxels in ROI to source position

        # 找到 inf 的位置
        inf_positions = np.argwhere(np.isinf(self.doseGrid.data))

        # 获取数组的形状
        shape = self.doseGrid.data.shape

        # 替换每个 inf 值为周围 4 个格子的平均值
        for pos in inf_positions:
            x, y, z = pos
            neighbors = []

            # 获取相邻的 4 个格子，确保不越界
            if x > 0:
                neighbors.append(self.doseGrid.data[x - 1, y, z])  # 前一个
            if x < shape[0] - 1:
                neighbors.append(self.doseGrid.data[x + 1, y, z])  # 后一个
            if y > 0:
                neighbors.append(self.doseGrid.data[x, y - 1, z])  # 上一个
            if y < shape[1] - 1:
                neighbors.append(self.doseGrid.data[x, y + 1, z])  # 下一个
            if z > 0:
                neighbors.append(self.doseGrid.data[x, y, z - 1])  # 左一个
            if z < shape[2] - 1:
                neighbors.append(self.doseGrid.data[x, y, z + 1])  # 右一个

            # 过滤掉 inf 值的邻居
            neighbors = [n for n in neighbors if not np.isinf(n)]

            # 计算邻居的平均值
            if neighbors:
                self.doseGrid.data[x, y, z] = np.mean(neighbors)
            else:
                self.doseGrid.data[x, y, z] = 0  # 如果没有有效邻居，就设为 0

        roiIndex = addMargin_GPU(margin, roiIndex, self.doseGrid)
        index = np.unravel_index(roiIndex, self.doseGrid.data.shape, order="C")
        nPos = np.size(index) // 3
        index = np.reshape(np.array(index), (3, nPos))
        pos = self.doseGrid.getPos(index)  # pos in x y z
        # nBeam = np.size(roiIndex)
        #
        # tmpDir = pos - self.source
        # # tmpDir = np.transpose(tmpDir, (-1, 0))
        # # tmpDir = tmpDir / np.linalg.norm(tmpDir, axis=1,keepdims=True)
        # tmpSource =  self.source * np.ones((1, nBeam))
        # ctCubeSize = np.squeeze(self.doseGrid.resolution) * self.doseGrid.dims
        #
        # maxTraceSepth = int(
        #     (ctCubeSize[0] ** 2 + ctCubeSize[1] ** 2 + ctCubeSize[2] ** 2) ** 0.5 / np.min(self.doseGrid.resolution))
        #
        # weq = np.zeros((nBeam, maxTraceSepth), dtype=np.float32)
        # # 这个记录的是每个Ray打中CT的点的距离
        # hitDis = np.zeros((nBeam,), dtype=np.float32)
        # # 这个记录的是每个Ray打中CT的点的坐标
        # hitPos = np.zeros((nBeam, 3), dtype=np.float32)
        # calWEQ(self.doseGrid.data, self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, tmpSource,
        #        tmpDir, weq, hitPos, hitDis, nBeam)
        # tmpDis = np.sqrt(np.sum(np.power(pos - self.source, 2), axis=0))
        #
        # result = []
        # cols = np.arange(maxTraceSepth)
        # ind = (tmpDis - hitDis - 0.5 * np.min(self.doseGrid.resolution)) / np.min(self.doseGrid.resolution)
        # for i in range(len(ind)):
        #     result.append(np.interp(ind[i], cols, weq[i,:]))
        # return np.array(result)

        pos = pos.flatten(order="F")
        #
        weq = np.zeros((np.size(roiIndex),), dtype=np.float32)
        cuGetWeqForRegion(
            weq,
            pos,
            self.source,
            self.doseGrid.data,
            self.doseGrid.corner,
            self.doseGrid.resolution,
            self.doseGrid.dims,
            0,
        )

        # weq[np.isinf(weq)] = 0
        # weq[np.isnan(weq)] = 0
        #
        return weq

    # def getWEQForROI_GPU(self, roiIndex, margin=5):
    #     # get water equivalent length for all voxels in ROI to source position
    #
    #     # 找到 inf 的位置
    #     inf_positions = np.argwhere(np.isinf(self.doseGrid.data))
    #
    #     # 获取数组的形状
    #     shape = self.doseGrid.data.shape
    #
    #     # 替换每个 inf 值为周围 4 个格子的平均值
    #     for pos in inf_positions:
    #         x, y, z = pos
    #         neighbors = []
    #
    #         # 获取相邻的 4 个格子，确保不越界
    #         if x > 0: neighbors.append(self.doseGrid.data[x - 1, y, z])  # 前一个
    #         if x < shape[0] - 1: neighbors.append(self.doseGrid.data[x + 1, y, z])  # 后一个
    #         if y > 0: neighbors.append(self.doseGrid.data[x, y - 1, z])  # 上一个
    #         if y < shape[1] - 1: neighbors.append(self.doseGrid.data[x, y + 1, z])  # 下一个
    #         if z > 0: neighbors.append(self.doseGrid.data[x, y, z - 1])  # 左一个
    #         if z < shape[2] - 1: neighbors.append(self.doseGrid.data[x, y, z + 1])  # 右一个
    #
    #         # 过滤掉 inf 值的邻居
    #         neighbors = [n for n in neighbors if not np.isinf(n)]
    #
    #         # 计算邻居的平均值
    #         if neighbors:
    #             self.doseGrid.data[x, y, z] = np.mean(neighbors)
    #         else:
    #             self.doseGrid.data[x, y, z] = 0  # 如果没有有效邻居，就设为 0
    #
    #
    #
    #     roiIndex = addMargin_GPU(margin, roiIndex, self.doseGrid)
    #
    #     index = np.unravel_index(roiIndex, self.doseGrid.data.shape, order='C')
    #     nPos  = np.size(index) // 3
    #     index = np.reshape(np.array(index), (3, nPos))
    #     pos   = self.doseGrid.getPos(index)  # pos in x y z
    #     nBeam = np.size(roiIndex)
    #
    #     tmpDir = pos - self.source
    #     # tmpDir = np.transpose(tmpDir, (-1, 0))
    #     # tmpDir = tmpDir / np.linalg.norm(tmpDir, axis=1,keepdims=True)
    #     tmpSource =  self.source * np.ones((1, nBeam))
    #     ctCubeSize = np.squeeze(self.doseGrid.resolution) * self.doseGrid.dims
    #
    #     maxTraceSepth = int(
    #         (ctCubeSize[0] ** 2 + ctCubeSize[1] ** 2 + ctCubeSize[2] ** 2) ** 0.5 / np.min(self.doseGrid.resolution))
    #
    #     weq = np.zeros((nBeam, maxTraceSepth), dtype=np.float32)
    #     # 这个记录的是每个Ray打中CT的点的距离
    #     hitDis = np.zeros((nBeam,), dtype=np.float32)
    #     # 这个记录的是每个Ray打中CT的点的坐标
    #     hitPos = np.zeros((nBeam, 3), dtype=np.float32)
    #     calWEQ(self.doseGrid.data, self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, tmpSource,
    #            tmpDir, weq, hitPos, hitDis, nBeam)
    #     tmpDis = np.sqrt(np.sum(np.power(pos - self.source, 2), axis=0))
    #
    #     result = []
    #     cols = np.arange(maxTraceSepth)
    #     ind = (tmpDis - hitDis - 0.5 * np.min(self.doseGrid.resolution)) / np.min(self.doseGrid.resolution)
    #     for i in range(len(ind)):
    #         result.append(np.interp(ind[i], cols, weq[i,:]))
    #     return np.array(result)
    #
    #     pos = pos.flatten(order='F')
    #     #
    #     weq = np.zeros((np.size(roiIndex),), dtype=np.float32)
    #     cuGetWeqForRegion(weq, pos, self.source, self.doseGrid.data, \
    #                       self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, 0)
    #
    #     # weq[np.isinf(weq)] = 0
    #     # weq[np.isnan(weq)] = 0
    #     #
    #     return weq

    def getBeamDir(self, theta, phi, mode="z"):
        # self.infoFirst("bmdir", "beam direction relative to gantry angle; unit in radiant")
        # Beam direction，即 Ray 相对于给定坐标系的向量
        R = np.array([self.bmxdir, self.bmydir, self.bmdir])
        R = np.squeeze(R)
        R = R.T  # Note this transpose is used only because of list -> numpy array
        # NOTE, in this setting, phi start from dicom x,
        # positive angle means x->-z, namely HFS head to left

        localDir = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
        localDir = np.reshape(np.array(localDir), (3, np.size(theta)))
        localDir = np.matmul(R, np.array(localDir))
        #  x y z
        if mode == "z":
            return localDir
        else:
            localXDir = [
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                -np.sin(theta),
            ]
            localXDir = np.reshape(np.array(localXDir), (3, np.size(theta)))
            localXDir = np.matmul(R, np.array(localXDir))

            return localDir, localXDir

    def weqToPhysPos(self, weq, theta, phi, step=0.1):
        localDir = self.getBeamDir(theta, phi)

        physPos = np.zeros((3, np.size(theta)))
        for i in range(np.size(theta)):
            tmpDir = np.reshape(localDir[:, i], (3, 1))
            target = self.source + tmpDir * 7000  # consider maximum distance 7000 mm
            tmpweq = self.getWEQBetweenTwoPos(self.source, target, "array", 6000, step)
            idx = np.argmin(np.abs(tmpweq - weq[i]))
            deltaD = weq[i] - tmpweq[idx]
            physPos[:, i] = np.squeeze(self.source + tmpDir * (6000 + idx * step + deltaD))

        return physPos

    def weqToPhysPos_GPU(self, weq, theta, phi, step=0.1):
        localDir = self.getBeamDir(theta, phi)
        localDir = localDir.T.flatten()
        weq[weq < 0] = 0
        physPos = -100 * np.ones((np.size(theta), 3), dtype=np.float32)
        cuWeqToPhysPos(
            physPos,
            weq,
            localDir,
            self.source,
            self.doseGrid.data,
            self.doseGrid.corner,
            self.doseGrid.resolution,
            self.doseGrid.dims,
            step,
            0,
        )
        physPos = physPos.T
        return physPos

    def hitTargetFlag(
        self,
        weq,
        theta,
        phi,
        roiIndex,
        disThres=5,
        backEneThres=0,
        forwardEneThres=0,
    ):
        self.logger.info("hit:Theta phi should be in unit of radiant")
        if np.size(disThres) == 3:
            disThres = np.reshape(disThres, (3, 1))
        weq = np.array(weq)
        theta = np.array(theta)
        phi = np.array(phi)

        physPos0 = self.weqToPhysPos_GPU(weq + backEneThres, theta, phi)
        physPos1 = self.weqToPhysPos_GPU(weq + forwardEneThres, theta, phi)
        nStep = (np.max(np.sqrt(np.sum(np.square(physPos1 - physPos0), axis=0))) // 2).astype(int)

        index = np.unravel_index(roiIndex, self.doseGrid.data.shape, order="C")
        index = np.squeeze(np.array(index))
        pos = self.doseGrid.getPos(index)  # pos in x y z

        hitFlag = np.zeros((np.size(theta),), dtype=np.bool)
        for i in range(nStep + 2):
            physPos = physPos0 + i * (physPos1 - physPos0) / (nStep + 1)
            tmpflag = ifAInB(disThres, self.doseGrid.resolution, physPos, pos, 0)
            hitFlag = np.logical_or(hitFlag, tmpflag)
        return hitFlag

    def getRayAccumulatedWEQ(
        self,
        theta,
        phi,
        extcontour_linear=None,
        transverseCutoff=10,
        crossStep=0.2,
        parallelStep=0.5,
    ):
        # theta, phi, ray direction relative to beam direction
        # extcontour_linear. linear index in grid
        nBeam = np.size(theta)
        if (extcontour_linear == None).any():
            extcontour_linear = np.arange(np.size(self.doseGrid.data))
        nPos = np.size(extcontour_linear)

        index = np.unravel_index(extcontour_linear, self.doseGrid.data.shape, order="C")
        index = np.squeeze(np.array(index))
        pos = self.doseGrid.getPos(index)  # pos in x y z

        localDir = self.getBeamDir(theta, phi)

        alongBeamWEQ = csc_matrix((np.size(extcontour_linear), 0))
        crossBeamWEQ = csc_matrix((np.size(extcontour_linear), 0))
        for ibeam in range(nBeam):
            self.logger.info("Calculating beam {:d}".format(ibeam))
            projectedLength = np.dot(localDir[:, ibeam], pos - self.source)
            target = self.source + np.reshape(localDir[:, ibeam], (3, 1)) * projectedLength
            tmpAlongBeamWEQ = np.zeros((nPos, 1))
            tmpCrossBeamWEQ = np.zeros((nPos, 1))
            for iPos in range(nPos):
                crossDis = self.getWEQBetweenTwoPos(target[:, iPos], pos[:, iPos], "value", 0, crossStep)
                # crossDis = np.sqrt(np.sum(np.square(target[:,iPos]-pos[:,iPos])))
                if crossDis > transverseCutoff:
                    continue
                tmpCrossBeamWEQ[iPos, 0] = crossDis
                # tmpCrossBeamWEQ[iPos,0] = self.getWEQBetweenTwoPos(target[:,iPos], pos[:,iPos], "value", 0, crossStep)
                tmpAlongBeamWEQ[iPos, 0] = self.getWEQBetweenTwoPos(self.source, target[:, iPos], "value", 6000, parallelStep)
            alongBeamWEQ = csc_stack((alongBeamWEQ, csc_matrix(tmpAlongBeamWEQ, (nPos, 1))))
            crossBeamWEQ = csc_stack((crossBeamWEQ, csc_matrix(tmpCrossBeamWEQ, (nPos, 1))))

        return alongBeamWEQ, crossBeamWEQ

    def getRegionToRayMinDis_GPU(self, rayDir, pos, source, minLongitudalDis=0, maxLongitudalDis=100000):
        nBeam = np.size(rayDir) // 3
        minDis = np.zeros(nBeam, dtype=np.float32)

        if np.size(minLongitudalDis) == 1:
            minLongitudalDis = (
                np.ones(
                    nBeam,
                )
                * minLongitudalDis
            )
        elif np.size(minLongitudalDis) != nBeam:
            raise RuntimeError("Dimension mismatch.")
        else:
            minLongitudalDis = np.array(minLongitudalDis)

        if np.size(maxLongitudalDis) == 1:
            maxLongitudalDis = (
                np.ones(
                    nBeam,
                )
                * maxLongitudalDis
            )
        elif np.size(maxLongitudalDis) != nBeam:
            raise RuntimeError("Dimension mismatch.")
        else:
            maxLongitudalDis = np.array(maxLongitudalDis)

        rayDir = rayDir.flatten(order="F")
        pos = pos.flatten(order="F")

        cuCalMinDisToRay(minDis, source, rayDir, pos, minLongitudalDis, maxLongitudalDis, 0)
        return minDis

    def getRayAccumulatedWEQ_GPU(
        self,
        theta,
        phi,
        extcontour_linear=None,
        transverseCutoff=10,
        crossStep=0.2,
        parallelStep=0.5,
        ene=None,
        longitudalCutoff=1000,
        startProgress=0.0,
        endProgress=1.0,
    ):
        # GPU version of calculating weq
        localDir = self.getBeamDir(theta, phi)
        if (extcontour_linear == None).any():
            extcontour_linear = np.arange(np.size(self.doseGrid.data))
        alongBeamWEQ, crossBeamWEQ = self.getAccumulatedWEQFromDir_GPU(
            localDir,
            extcontour_linear,
            transverseCutoff,
            crossStep,
            parallelStep,
            np.array(ene),
            longitudalCutoff,
            startProgress,
            endProgress,
        )

        return alongBeamWEQ, crossBeamWEQ

    def getAccumulatedWEQFromDir_GPU(
        self,
        localDir,
        extcontour_linear,
        transverseCutoff=20,
        crossStep=0.2,
        parallelStep=0.5,
        ene=None,
        longitudalCutoff=1000,
        startProgress=-1.0,
        endProgress=1.0,
        orderFlag=True,
    ):
        index = np.unravel_index(extcontour_linear, self.doseGrid.data.shape, order="C")
        index = np.squeeze(np.array(index))
        index = index.T.flatten()

        nBeam = localDir.shape[1]
        localDir = localDir.flatten(order="F")

        preCutoff = np.array([20, 30, 35, 40, 50])  # 200, 250, 300, 350, 400
        if transverseCutoff < 0:
            if ene is None or np.size(ene) != nBeam:
                raise RuntimeError("Provided parameters for cal WEQ WRONG!!!")
            else:
                # idx = np.ceil((ene-200)/50.0).astype(np.int)
                # idx[idx>np.size(preCutoff)-1] = np.size(preCutoff)-1
                # transverseCutoff = preCutoff[idx]
                transverseCutoff = np.interp(ene, np.arange(200, 450, 50), preCutoff)
        else:
            transverseCutoff = transverseCutoff * np.ones(nBeam)

        if np.size(longitudalCutoff) == 1:
            longitudalCutoff = longitudalCutoff * np.ones(nBeam)
        if np.size(longitudalCutoff) != nBeam:
            raise RuntimeError("WRONG parameter for longitudal cutoff!")
        # divide into batches
        nCalVoxels = np.size(extcontour_linear)
        nVoxels = self.doseGrid.size
        num_per_group = 501
        self.logger.info("voxels {} calvoxes {}".format(nVoxels, nCalVoxels))
        requireMem = 18 * 1024 * 1024 * 1024  # 18 G GPU memory
        status = cuMemTestAlloc(requireMem, 0)  # only one card is visible now
        while status == -1:
            requireMem /= 1.2
            status = cuMemTestAlloc(np.uint64(requireMem), 0)
            if np.uint64(requireMem) < 256 * 1024 * 1024:  # 256 MB
                self.logger.error("NOT enough memory")
                raise RuntimeError("Not enough GPU resourses! please WAIT!")
        excessRatio = 0.1
        if np.max(transverseCutoff) > 40:
            excessRatio = 0.2
        num_per_group = np.int((requireMem / 4 / (1 + excessRatio) - nVoxels) / nCalVoxels / 2)
        if nCalVoxels * num_per_group > 2**31:
            num_per_group = 2**31 // nCalVoxels
        if num_per_group < 1:
            raise RuntimeError(
                "ROI voxels is too many!!! \
                Consider use smaller ROI or larger dose grid resolution!!!"
            )
        requireMem = (2 * num_per_group * nCalVoxels + nVoxels) * 4 * (1 + excessRatio)
        self.logger.info(
            "num spots per group {}, memory pressure {} MB, {} groups".format(
                num_per_group,
                requireMem // (1024**2),
                int(nBeam / num_per_group) + 1,
            )
        )

        totalEstNNZ = 2.5 * np.sum(4 * (transverseCutoff + np.max(self.doseGrid.resolution) * 3) ** 2 * (longitudalCutoff + 3 * np.max(self.doseGrid.resolution))) / self.doseGrid.voxSize
        if totalEstNNZ < 1e7:
            totalEstNNZ = 1e7
        maxCPUMem = 100  # GB
        maxEstNNZ = np.int(maxCPUMem * 1024**3 / 8 / 2 / 1.5 / 4)
        if maxEstNNZ > 2**31:
            maxEstNNZ = 2**31
        if totalEstNNZ > maxEstNNZ:
            transverseCutoff = transverseCutoff * np.sqrt(maxEstNNZ / totalEstNNZ)
            self.logger.info("squeeze ratio {}".format(np.sqrt(maxEstNNZ / totalEstNNZ)))
            totalEstNNZ = maxEstNNZ

        totalEstNNZ = np.int(totalEstNNZ)
        currentNNZ = 0
        alongBeamWEQData = np.zeros(totalEstNNZ, dtype=np.float32)
        alongBeamWEQIndices = np.zeros(totalEstNNZ, dtype=np.int32)
        alongBeamWEQIndptr = np.zeros(nBeam + 1, dtype=np.int32)
        crossBeamWEQData = np.zeros(totalEstNNZ, dtype=np.float32)
        crossBeamWEQIndices = np.zeros(totalEstNNZ, dtype=np.int32)
        crossBeamWEQIndptr = np.zeros(nBeam + 1, dtype=np.int32)
        _total_batch = int(nBeam / num_per_group) + 1
        for i in range(_total_batch):
            self.logger.info("calculating the {}th batch".format(i))
            if self.serviceMode and startProgress > 0:
                NetWork.sendStatus(
                    None,
                    self.task_id,
                    "Calculating the {}th / {} batch".format(i, str(_total_batch)),
                    startProgress + i * (endProgress - startProgress) * 0.78 / (int(nBeam / num_per_group) + 1),
                )
            remainder = nBeam - i * num_per_group
            if remainder <= 0:
                break
            if remainder > num_per_group:
                remainder = num_per_group
            dosemap1, sparse_ind1, dosemap2, sparse_ind2 = self.getRayAccumulatedWEQ_GPU_Batch(
                nCalVoxels,
                remainder,
                localDir[3 * i * num_per_group : 3 * (i * num_per_group + remainder)],
                index,
                self.doseGrid,
                transverseCutoff[i * num_per_group : (i * num_per_group + remainder)],
                longitudalCutoff[i * num_per_group : (i * num_per_group + remainder)],
                crossStep,
                parallelStep,
                orderFlag,
                0,
            )
            nnz = sparse_ind1[0]
            alongBeamWEQData[currentNNZ : currentNNZ + nnz] = dosemap1[0:nnz]
            alongBeamWEQIndices[currentNNZ : currentNNZ + nnz] = sparse_ind1[1 : 1 + nnz]
            alongBeamWEQIndptr[i * num_per_group + 1 : 1 + i * num_per_group + remainder] = currentNNZ + sparse_ind1[2 + nnz : 1 + nnz + remainder + 1]
            del dosemap1, sparse_ind1

            crossBeamWEQData[currentNNZ : currentNNZ + nnz] = dosemap2[0:nnz]
            crossBeamWEQIndices[currentNNZ : currentNNZ + nnz] = sparse_ind2[1 : 1 + nnz]
            crossBeamWEQIndptr[i * num_per_group + 1 : 1 + i * num_per_group + remainder] = currentNNZ + sparse_ind2[2 + nnz : 1 + nnz + remainder + 1]
            del dosemap2, sparse_ind2
            currentNNZ += nnz
        alongBeamWEQIndptr[nBeam] = currentNNZ
        crossBeamWEQIndptr[nBeam] = currentNNZ

        alongBeamWEQ = csc_matrix(
            (alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr),
            (nCalVoxels, nBeam),
        )
        del alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr
        crossBeamWEQ = csc_matrix(
            (crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr),
            (nCalVoxels, nBeam),
        )
        del crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr
        return alongBeamWEQ, crossBeamWEQ

        # if(not orderFlag):
        #     alongBeamWEQ = csc_matrix((alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr),(nCalVoxels, nBeam))
        #     del alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr
        #     crossBeamWEQ = csc_matrix((crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr),(nCalVoxels, nBeam))
        #     del crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr
        #     return alongBeamWEQ, crossBeamWEQ

        # self.logger.info("===== Order matrix ======= {}".format(currentNNZ))

        # if(self.serviceMode and startProgress>=0):
        #         NetWork.sendStatus(None, self.task_id, "Ordering WEQ", startProgress+(endProgress-startProgress)*0.8)
        # tmpidx = np.zeros(currentNNZ, dtype=np.int)
        # with ThreadPoolExecutor() as t:
        #     allTask = [t.submit(setOrder, tmpidx, jj, alongBeamWEQIndices, alongBeamWEQIndptr) for jj in range(nBeam) \
        #         if alongBeamWEQIndptr[jj+1]-alongBeamWEQIndptr[jj]>1]
        #     wait(allTask, return_when=ALL_COMPLETED)
        #     # for task in allTask:
        #     #     print(task.exception())
        # alongBeamWEQIndices = alongBeamWEQIndices[tmpidx]
        # alongBeamWEQData = alongBeamWEQData[tmpidx]

        # alongBeamWEQ = csc_matrix((alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr),(nCalVoxels, nBeam))
        # del alongBeamWEQData, alongBeamWEQIndices, alongBeamWEQIndptr
        # # alongBeamWEQ.eliminate_zeros()

        # if(self.serviceMode and startProgress>=0):
        #         NetWork.sendStatus(None, self.task_id, "Ordering WEQ", startProgress+(endProgress-startProgress)*0.9)
        # with ThreadPoolExecutor() as t:
        #     allTask = [t.submit(setOrder, tmpidx, jj, crossBeamWEQIndices, crossBeamWEQIndptr) for jj in range(nBeam) \
        #         if crossBeamWEQIndptr[jj+1]-crossBeamWEQIndptr[jj]>1]
        #     wait(allTask, return_when=ALL_COMPLETED)
        # crossBeamWEQIndices = crossBeamWEQIndices[tmpidx]
        # crossBeamWEQData = crossBeamWEQData[tmpidx]
        # crossBeamWEQ = csc_matrix((crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr),(nCalVoxels, nBeam))
        # del crossBeamWEQData, crossBeamWEQIndices, crossBeamWEQIndptr, tmpidx
        # self.logger.info("===== Order matrix done ======= {}".format(currentNNZ))
        # return alongBeamWEQ, crossBeamWEQ

    def getRayAccumulatedWEQ_GPU_Batch(
        self,
        nPos,
        nBeam,
        localDir,
        index,
        doseGrid: GRID,
        transverseCutoff,
        longitudalCutoff,
        crossStep=0.2,
        parallelStep=0.5,
        orderFlag=True,
        gpuid=0,
    ):
        sparsity = 0.1
        if np.max(transverseCutoff) >= 40:
            sparsity = 0.2
        while True:
            guessNum = int(nBeam * nPos * sparsity)
            dosemap1 = np.zeros((guessNum), dtype=np.float32)
            sparse_ind1 = np.zeros((1 + guessNum + nBeam + 1), dtype=np.int32)
            dosemap2 = np.zeros((guessNum), dtype=np.float32)
            sparse_ind2 = np.zeros((1 + guessNum + nBeam + 1), dtype=np.int32)

            status = 1
            status = cuCalWEQ(
                dosemap1,
                sparse_ind1,
                dosemap2,
                sparse_ind2,
                self.source,
                localDir,
                doseGrid.data,
                doseGrid.corner,
                doseGrid.resolution,
                doseGrid.dims,
                np.hstack((doseGrid.orientation, doseGrid.translation)),
                index,
                transverseCutoff,
                longitudalCutoff,
                crossStep,
                parallelStep,
                0.5,
                orderFlag,
                gpuid,
            )

            if status is not None:
                raise RuntimeError("cal weq error")

            nnz1 = sparse_ind1[0]
            nnz2 = sparse_ind2[0]
            self.logger.info("nnz is {:d} and {:d}".format(nnz1, nnz2))

            if nnz1 < 0 or nnz2 < 0:
                del dosemap1, dosemap2, sparse_ind1, sparse_ind2
                sparsity += 0.1
                self.logger.info("sparsity is too low, increased by 0.1. Now {:.2f}".format(sparsity))
                continue
            else:
                self.logger.info("Actual sparsity {:.4f}".format(nnz1 / guessNum * sparsity))
                return dosemap1, sparse_ind1, dosemap2, sparse_ind2

    def selectEnergies(self, wed_bound, bm, energy_option):
        return_message = ""
        # 第一步，根据布点区域最大最小水等效深度选择能量。
        (wed_min, wed_max) = (wed_bound[0], wed_bound[1])
        data = bm.triGaussian
        iddDepth = data["IDDDepth"]
        available_energy = np.array(data["meaEneList"]).astype("float32")  # list of available energies in beam model.
        available_range = np.array(data["R80"]).astype("float32")  # already in water
        # # 减去机头水等效。
        # nozzle_rs =  bm.getRS(available_energy)
        # available_range = available_range - nozzle_rs

        if (available_range < wed_min).all():
            self.logger.error("[Func selectEnergies] Available ranges in beam model are all shorter than wed_min.")
            self.logger.warning("Did you use correct machine and ROI?")
            _msg = """Target position is too deep. Minimum target WED is {} mm. Maximum energy {} MeV/u has range {} mm.
            Please choose other beam angles or extend energy range.""".format(
                str(np.round(wed_min, 1)),
                str(np.max(available_energy)),
                str(np.max(available_range)),
            )
            return_message = _msg
            raise RuntimeError(_msg)

        if (available_range > wed_max).all():
            self.logger.error("[Func selectEnergies] Available ranges in beam model are all longer than wed_max.")
            self.logger.warning("Did you forget adding range shifter?")
            _msg = """Target position is too shallow. Maximum target WED is {} mm. Minimum energy {} MeV/u has range {} mm.
            Please choose other beam angles or extend energy range.""".format(
                str(np.round(wed_max, 1)),
                str(np.min(available_energy)),
                str(np.min(available_range)),
            )
            return_message = _msg
            raise RuntimeError(_msg)

        # add warning for request
        if np.max(available_range) < wed_max or np.min(available_range) > wed_min:
            return_message = "Some parts of the target are not covered. "
            if np.max(available_range) < wed_max:
                return_message = return_message + "Target is too deep: maximum target WED is {} mm, maximum energy has range {} mm.".format(
                    str(np.round(wed_max, 1)),
                    str(np.round(np.max(available_range), 1)),
                )
            if np.min(available_range) > wed_min:
                return_message = return_message + "Target is too shallow: minimum target WED is {} mm, minimum energy has range {} mm.".format(
                    str(np.round(wed_min, 1)),
                    str(np.round(np.min(available_range), 1)),
                )
            return_message = return_message + " Please confirm before dose optimization."
            self.logger.warning("Part of the ROI will be missed!!! Confirm it before opt!!")
            self.logger.info("available max {:.2f} wanted max {:.2f}".format(np.max(available_range), wed_max))
            self.logger.info("available min {:.2f} wanted min {:.2f}".format(np.min(available_range), wed_min))

        energy_max_idx = np.argmin(abs(available_range - wed_max))
        # 最高能量的range要比 wed_max 长，为了包住布点区域。
        # correct possible boundary issues
        if energy_max_idx < np.size(available_energy) - 1 and available_range[energy_max_idx] < wed_max:
            self.logger.info(
                "Max energy selected {} MeV/u has shorter range {} mm than wed_max {} mm. Add 1 to index.".format(
                    str(available_energy[energy_max_idx]),
                    str(available_range[energy_max_idx]),
                    str(wed_max),
                )
            )
            energy_max_idx += 1
        self.logger.info(
            "Max energy selected is {} MeV/u, with range in water {} mm. wed_max {} mm.".format(
                str(available_energy[energy_max_idx]),
                str(available_range[energy_max_idx]),
                str(wed_max),
            )
        )

        selected_energy = [available_energy[energy_max_idx]]
        selected_idx = [energy_max_idx]
        if energy_option[0] == "fixed":
            energy_spacing = energy_option[1]  # energy spacing in water equivalent mm.
            assert wed_max > energy_spacing > 0
            energy_idx = energy_max_idx
            for i in range(energy_max_idx):
                range1 = available_range[energy_idx]  # range of the previous higher energy.
                energy_idx = np.argmin(abs(available_range - (range1 - energy_spacing)))
                # 只选择 range + 0.2mm and -1.8 mm 范围内的能量。范围可调整，但需要等于beam model里range的间隔。TODO: range 间隔暂时写死。
                if available_range[energy_idx] > range1 - energy_spacing + 0.2:
                    energy_idx -= 1
                elif available_range[energy_idx] < range1 - energy_spacing - 1.8:
                    energy_idx += 1
                # # 如果出于 index 原因选能比已选的能量高，结束选能。
                if (available_energy[energy_idx] > selected_energy).any() or energy_idx < 0:
                    break
                # 如果选能等于上一个选能，选beam model 里下一个能量低的能量。（还没发生过，万一呢）。
                if (available_energy[energy_idx] == selected_energy).any():
                    energy_idx -= 1
                selected_energy.append(available_energy[energy_idx])
                selected_idx.append(energy_idx)
                # 如果当前选能的 range 比 wed_min 短，结束选能。
                if available_range[energy_idx] < wed_min:
                    break
        elif energy_option[0] == "auto":
            energy_spacing = energy_option[1]  # energy spacing in % Bragg peak of the higher energy.
            if energy_spacing != 1:
                assert 1 > energy_spacing > 0
                energy_idx = energy_max_idx
                for i in range(energy_max_idx):
                    tmpidd = data["dose"][energy_idx, :]
                    peak_idx = np.argmax(tmpidd)  # range of the previous higher energy.
                    peak80_mag = tmpidd[peak_idx] * energy_spacing
                    __idx = np.argmin(abs(tmpidd[0 : peak_idx + 1] - peak80_mag))
                    assert __idx <= peak_idx
                    energy_idx = np.argmin(abs(available_range - iddDepth[__idx]))

                    # 只选择 range + 0.2mm and -1.8 mm 范围内的能量。范围可调整，但需要等于beam model里range的间隔。TODO: range 间隔暂时写死。
                    if available_range[energy_idx] > __idx * 0.1 + 0.2:
                        energy_idx -= 1
                    elif available_range[energy_idx] < __idx * 0.1 - 1.8:
                        energy_idx += 1
                    # # 如果出于 index 原因选能比已选的能量高，结束选能。
                    if (available_energy[energy_idx] > selected_energy).any() or energy_idx < 0:
                        break
                    # 如果选能等于上一个选能，选beam model 里下一个能量低的能量。（还没发生过，万一呢）。
                    if (available_energy[energy_idx] == selected_energy).any():
                        energy_idx -= 1
                    selected_energy.append(available_energy[energy_idx])
                    selected_idx.append(energy_idx)
                    # 如果当前选能的 range 比 wed_min 短，结束选能。
                    if available_range[energy_idx] < wed_min:
                        break
            else:
                energy_min_idx = np.where((available_range < wed_min) == True)[0][-1]
                selected_energy = (available_energy[energy_min_idx : energy_max_idx + 1])[::-1]
                selected_idx = np.arange(energy_max_idx, energy_min_idx, -1).astype(int)

        self.logger.info("Selected energies {} MeV".format(str(selected_energy)))
        self.logger.info("Selected energy indices {}".format(str(selected_idx)))
        self.logger.info("{} energies are selected.".format(str(len(selected_energy))))

        return available_energy, selected_idx, return_message

    def rayTracingSetPoint(
        self,
        wed_bound,
        thetamax,
        bm: BEAM_MODEL,
        SAD,
        energy_option,
        spot_option,
    ):  # by yunzhou
        # bm： beam model。python 字典。
        # wed_bound: 布点区域最大最小水等效 （water equivalent mm）。[wed_min, wed_max] array or list.
        # thetamax：布点区域横向 theta 最大值（radian）。float.
        # SAD：source-axis distance （mm）。float.
        # energy_option: ('fixed', 2 (water equivalent mm)) or ('auto', 0.8 or 80%). Size 1*2 tuple.

        # returns: scan_points. key 为能量在beam model 里的index。每个能量里，按r_max外接长方形布点之后，
        # 存每个点的 theta 和 phi，用来raytrace 判断每个点有没有hit到布点区域。以及 isocenter 平面里每个扫描点的 X，Z坐标。

        # 根据最大最小水等效选能。
        available_energy, selected_index, returnMessage = self.selectEnergies(wed_bound=wed_bound, bm=bm, energy_option=energy_option)

        r_max = SAD * np.squeeze(np.tan(thetamax))  # thetamax is given is radian.
        MAXFIELD = 110  # 22x22 cm^2 field
        if r_max > MAXFIELD:
            self.logger.info("thetamax too large! please check isocenter position!")
            self.logger.info("fix field size to {:d} mm".format(2 * MAXFIELD))
            r_max = MAXFIELD
        # r_max_at_snout = (SAD - SnAD) * np.squeeze(np.sin(thetamax))
        scan_points = dict()
        for __selected_index in selected_index:
            scan_points[__selected_index] = dict()
            # self.logger.info('Energy {} MeV has sigma1 {}mm, sigma2 {}mm, sigma3 {}mm.'.format(str(selected_energy[__selected_index]),\
            #                                                                         str(spot_size1), str(spot_size2), str(spot_size3)))
            if spot_option[0] == "auto":
                # max_idx      = np.where(bm.triGaussian['dose'][__selected_index] == np.amax(bm.triGaussian['dose'][__selected_index]))[0]
                # sigma_interp = np.interp(np.arange(0, 4000, 1), np.arange(0, 4000, 20), bm.triGaussian['sigma1'][__selected_index])  # TODO:间隔暂时写死。
                # spot_size1   = sigma_interp[max_idx]
                spot_size_x, spot_size_z = bm.getSpotSize(
                    ene=available_energy[__selected_index],
                    depth=bm.getPeakPos(available_energy[__selected_index], "peak"),
                    nozzleRS=bm.getRS(available_energy[__selected_index]),
                    nGauss=bm.triGaussian["nGauss"],
                )

                FWHM_x = 2.35 * spot_size_x
                FWHM_z = 2.35 * spot_size_z
                # spot_spacing_x = FWHM * 2 / 3  # automatic spot spacing = 2/3 * spot size.
                # spot_spacing_z = FWHM * 2 / 3
                spot_spacing_x = FWHM_x * spot_option[1][0]  # automatic spot spacing = 2/3 * spot size.
                spot_spacing_z = FWHM_z * spot_option[1][1]
            elif spot_option[0] == "fixed":
                spot_spacing_x = spot_option[1][0]
                spot_spacing_z = spot_option[1][1]
            self.logger.info(
                "Energy {}MeV, spot spacing x {}mm, spot spacing z {}mm.".format(
                    str(available_energy[__selected_index]),
                    str(spot_spacing_x),
                    str(spot_spacing_z),
                )
            )
            # 以 r_max 画外接正方形。
            halfgrid = r_max // spot_spacing_x
            spot_position_grid_x = (np.arange(halfgrid * 2 + 1) - halfgrid) * spot_spacing_x
            halfgrid = r_max // spot_spacing_z
            spot_position_grid_z = (np.arange(halfgrid * 2 + 1) - halfgrid) * spot_spacing_z
            xv, zv = np.meshgrid(spot_position_grid_x, spot_position_grid_z)

            theta = np.arctan((np.sqrt(xv**2 + zv**2)) / SAD)
            assert 0.5 * np.pi > theta.all() >= 0
            phi = np.arctan2(zv, xv)
            assert np.pi >= phi.all() >= -1 * np.pi

            __dum_ind = np.where(theta <= thetamax)
            theta = theta[__dum_ind]
            phi = phi[__dum_ind]
            spot_x = xv[__dum_ind]
            spot_z = zv[__dum_ind]

            scan_points[__selected_index]["theta"] = theta
            scan_points[__selected_index]["phi"] = phi
            scan_points[__selected_index]["x"] = spot_x
            scan_points[__selected_index]["z"] = spot_z
            nSpot = np.size(theta)
            scan_points[__selected_index]["energy"] = (
                np.ones(
                    nSpot,
                )
                * available_energy[__selected_index]
            )
            scan_points[__selected_index]["scaleFactor"] = bm.getnPerMUInterp(scan_points[__selected_index]["energy"])
            scan_points[__selected_index]["range"] = np.ones(
                nSpot,
            ) * bm.getPeakPos(available_energy[__selected_index], "R80") - bm.getRS(scan_points[__selected_index]["energy"])
            scan_points[__selected_index]["spotId"] = np.arange(nSpot).astype("int16")
            scan_points[__selected_index]["spot_spacing_x"] = np.ones(nSpot) * spot_spacing_x
            scan_points[__selected_index]["spot_spacing_z"] = np.ones(nSpot) * spot_spacing_z

        return scan_points, returnMessage

    def caldose_raytrace(
        self,
        bm: BEAM_MODEL,
        scan_points,
        alongBeamWEQ,
        crossBeamWEQ,
        rs=0,
        mode="Coarse",
        startProgress=-1,
        endProgress=1,
    ):
        # bm: beam model.
        # scan_points: 根据raytrace 后筛选的能够hit target 的spots，包含每个能量的 theta, phi, x, z.
        # alongBeamWEQ: spot 入射轴的水等效。M * N. M = number of voxels in external_contour_linear. N = Number of spots.
        # crossBeamWEQ: spot 垂直于入射轴的水等效距离。M * N. M = number of voxels in external_contour_linear. N = Number of spots.

        # return: fluence_mat in sparse matrix

        if mode == "Fine":
            cutoff = 0.00005
        elif mode == "Coarse":
            cutoff = 0.0004
        else:
            cutoff = 0.001

        # 准备 fluence_mat nnz array。
        fluence_data = np.zeros((alongBeamWEQ.nnz,), dtype=np.float32)

        # 算剂量。
        available_energy = np.array(bm.triGaussian["meaEneList"]).astype("float32")
        energy_idx_list = np.sort(list(scan_points.keys()))[::-1]  # make sure descending order
        alongBeamWEQ.data = alongBeamWEQ.data + rs
        self.logger.info("Start")
        deltaP = (endProgress - startProgress) / len(energy_idx_list)
        iSpot = 0
        for energy_idx in energy_idx_list:
            nSpot = len(scan_points[energy_idx]["theta"])
            for jj in range(iSpot, iSpot + nSpot):
                if alongBeamWEQ.indptr[jj + 1] - alongBeamWEQ.indptr[jj] > 1:
                    bm.caldose2(
                        fluence_data,
                        alongBeamWEQ.indptr,
                        jj,
                        available_energy[energy_idx],
                        alongBeamWEQ.data,
                        crossBeamWEQ.data,
                        5,
                        cutoff,
                    )

            # wait(allTask, return_when=ALL_COMPLETED)
            # NetWork.sendStatus(None, self.task_id, "Calculating the dose map.", startProgress+deltaP)

            self.logger.info("Calculating the dose map {:.2f}.".format(startProgress))
            NetWork.sendStatus(None, self.task_id, "Calculating the dose map.", startProgress)

            startProgress = startProgress + deltaP
            iSpot = iSpot + nSpot
            self.logger.info("dose for {} MeV/u done".format(available_energy[energy_idx]))

        # with ThreadPoolExecutor() as t:
        #     iSpot = 0
        #     for energy_idx in energy_idx_list:
        #         nSpot = len(scan_points[energy_idx]['theta'])
        #         allTask = [t.submit(bm.caldose2, fluence_data, alongBeamWEQ.indptr, jj, available_energy[energy_idx], alongBeamWEQ.data, crossBeamWEQ.data, 5, cutoff) \
        #             for jj in range(iSpot, iSpot+nSpot) if alongBeamWEQ.indptr[jj+1]-alongBeamWEQ.indptr[jj]>1]

        #         # wait(allTask, return_when=ALL_COMPLETED)
        #         # NetWork.sendStatus(None, self.task_id, "Calculating the dose map.", startProgress+deltaP)

        #         iTask = 0
        #         for task in as_completed(allTask):
        #             iTask = iTask + 1
        #             if(startProgress>0 and iTask%200==1):
        #                 self.logger.info("Calculating the dose map {:.2f}.".format(startProgress+iTask*deltaP/nSpot))
        #                 NetWork.sendStatus(None, self.task_id, "Calculating the dose map.", startProgress+iTask*deltaP/nSpot)
        #         for task in allTask:
        #             if(task.exception()!=None):
        #                 raise RuntimeError("Computing resource is NOT enough {}".format(task.exception()))

        #         startProgress = startProgress + deltaP
        #         iSpot = iSpot+nSpot
        #         self.logger.info("dose for {} MeV/u done".format(available_energy[energy_idx]))

        self.logger.info("Done")
        fluence_mat = csc_matrix(
            (fluence_data, alongBeamWEQ.indices, alongBeamWEQ.indptr),
            shape=alongBeamWEQ.shape,
        )
        fluence_mat.eliminate_zeros()
        self.logger.info("nnz number after eliminating small dose regions {} max {}".format(fluence_mat.nnz, np.max(fluence_data)))
        return fluence_mat

    def split_subspot(self, bm: BEAM_MODEL, beamparadata, enelist, sigmaThreshold=2):
        """
        If sigma at iso > 2 mm, split subspot.
        """
        num_energy = len(enelist)
        profilePara = np.vstack(
            (
                enelist,
                np.ones(
                    num_energy,
                ),
                np.ones(
                    num_energy,
                ),
                np.sqrt(beamparadata[:, 0] / 2),
                np.sqrt(beamparadata[:, 0] / 2),
            )
        ).transpose()

        # split_flag = np.zeros((np.shape(profilePara)[0],))
        # for Ei in range(np.shape(beamparadata)[0]):
        #     if beamparadata[Ei, 0] > 1:
        #         split_flag[Ei] = int(1)
        # split_flag = np.where(split_flag)[0]
        split_flag = np.sqrt(beamparadata[:, 0] / 2) >= sigmaThreshold

        __subspotdata = bm.splitSubSpots(
            profilePara=profilePara[split_flag, :],
            precision=5,
            nGauss=1,
            subspot_type="square",
        )
        subspotdata = np.zeros((num_energy, __subspotdata.shape[1], __subspotdata.shape[2]))
        subspotdata[split_flag] = __subspotdata
        subspotdata[~split_flag, 0, 2] = 1
        subspotdata[~split_flag, 0, 3] = np.sqrt(beamparadata[~split_flag, 0] / 2)
        subspotdata[~split_flag, 0, 4] = np.sqrt(beamparadata[~split_flag, 0] / 2)

        return subspotdata

    def interpSubspotData(self, bm, enelist, plotFlag=1):
        measEneList = bm.triGaussian["meaEneList"]
        subspotData = bm.triGaussian["subspotData"]
        subspotDataInterp = np.zeros(
            (
                len(enelist),
                bm.triGaussian["subspotData"].shape[1],
                bm.triGaussian["subspotData"].shape[2],
            )
        )

        for energy_i in range(len(enelist)):
            energy = enelist[energy_i]
            if energy in measEneList:
                _idx = np.where(measEneList == energy)
                subspotDataInterp[energy_i, :] = subspotData[_idx, :]
            else:
                _idx = np.argmin(np.abs(measEneList - energy))  # TODO measEne, eneList small to max order.
                if measEneList[_idx] < energy:
                    energy1, energy2 = measEneList[_idx], measEneList[_idx + 1]
                    idx = [_idx, _idx + 1]
                else:
                    energy1, energy2 = measEneList[_idx - 1], measEneList[_idx]
                    idx = [_idx - 1, _idx]
                assert energy1 < energy < energy2
                non_zero_idx_subspot1 = np.where(subspotData[idx[0], :, 2])
                non_zero_idx_subspot2 = np.where(subspotData[idx[1], :, 2])
                non_zero_idx_subspot = np.union1d(non_zero_idx_subspot1, non_zero_idx_subspot2)

                energy_dim = np.array([energy1, energy2])
                points = (energy_dim, non_zero_idx_subspot)
                interpolated_w = interpn(
                    points,
                    np.vstack(
                        (
                            subspotData[idx[1], non_zero_idx_subspot, 2],
                            subspotData[idx[0], non_zero_idx_subspot, 2],
                        )
                    ),
                    (energy, non_zero_idx_subspot),
                )

                subspotDataInterp[energy_i, non_zero_idx_subspot, :] = subspotData[idx[0], non_zero_idx_subspot, :]
                subspotDataInterp[energy_i, non_zero_idx_subspot, 2] = interpolated_w

                if plotFlag:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotData[idx[0], non_zero_idx_subspot, 2],
                        ".-",
                        label=measEneList[idx[0]],
                    )
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotData[idx[1], non_zero_idx_subspot, 2],
                        ".-",
                        label=measEneList[idx[1]],
                    )
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotDataInterp[energy_i, non_zero_idx_subspot, 2],
                        ".-",
                        label=enelist[energy_i],
                    )
                    plt.legend()
                    plt.grid()
                    plt.savefig("/data/yunzhou_figs/proton/interp_w" + str(energy) + ".png")
                    plt.close()
        return subspotDataInterp

    def interpSubspotDataSigmaTable(self, bm, enelist, beamparadata, plotFlag=0):
        subspotData = bm.triGaussian["subspotData"]
        subspotData_sigmalist = bm.triGaussian["subspotData_sigmalist"]
        subspotDataInterp = np.zeros(
            (
                len(enelist),
                bm.triGaussian["subspotData"].shape[1],
                bm.triGaussian["subspotData"].shape[2],
            )
        )

        for energy_i in range(len(enelist)):
            energy = enelist[energy_i]
            sigma = np.sqrt(beamparadata[energy_i, 0] / 2)
            if sigma in subspotData_sigmalist:
                _idx = np.where(subspotData_sigmalist == sigma)
                subspotDataInterp[energy_i, :] = subspotData[_idx, :]
            else:
                _idx = np.argmin(np.abs(subspotData_sigmalist - sigma))  # TODO measEne, eneList small to max order.
                if subspotData_sigmalist[_idx] < sigma:
                    sigma1, sigma2 = (
                        subspotData_sigmalist[_idx],
                        subspotData_sigmalist[_idx + 1],
                    )  # Here "energy" is sigma.
                    idx = [_idx, _idx + 1]
                else:
                    sigma1, sigma2 = (
                        subspotData_sigmalist[_idx - 1],
                        subspotData_sigmalist[_idx],
                    )
                    idx = [_idx - 1, _idx]
                assert sigma1 < sigma < sigma2
                non_zero_idx_subspot1 = np.where(subspotData[idx[0], :, 2])
                non_zero_idx_subspot2 = np.where(subspotData[idx[1], :, 2])
                non_zero_idx_subspot = np.union1d(non_zero_idx_subspot1, non_zero_idx_subspot2)

                energy_dim = np.array([sigma1, sigma2])
                points = (energy_dim, non_zero_idx_subspot)
                interpolated_w = interpn(
                    points,
                    np.vstack(
                        (
                            subspotData[idx[1], non_zero_idx_subspot, 2],
                            subspotData[idx[0], non_zero_idx_subspot, 2],
                        )
                    ),
                    (sigma, non_zero_idx_subspot),
                )

                subspotDataInterp[energy_i, non_zero_idx_subspot, :] = subspotData[idx[0], non_zero_idx_subspot, :]
                subspotDataInterp[energy_i, non_zero_idx_subspot, 2] = interpolated_w

                if plotFlag:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotData[idx[0], non_zero_idx_subspot, 2],
                        ".-",
                        label=subspotData_sigmalist[idx[0]],
                    )
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotData[idx[1], non_zero_idx_subspot, 2],
                        ".-",
                        label=subspotData_sigmalist[idx[1]],
                    )
                    plt.plot(
                        non_zero_idx_subspot,
                        subspotDataInterp[energy_i, non_zero_idx_subspot, 2],
                        ".-",
                        label=np.round(sigma, 2),
                    )
                    plt.legend()
                    plt.grid()
                    plt.title(str(energy))
                    plt.savefig("/data/yunzhou_figs/proton/interp_w" + str(energy) + ".png")
                    plt.close()
        return subspotDataInterp

    def transformation_matrix_between_vectors(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)  # 归一化v1
        v2 = v2 / np.linalg.norm(v2)  # 归一化v2

        # 计算叉乘矩阵（skew-symmetric matrix）
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)

        # 构建旋转矩阵
        if s == 0:  # 如果s为0，表示v1和v2平行或反平行
            if c > 0:
                return np.eye(3)  # 相同方向，不需要旋转
            else:
                return -np.eye(3)  # 相反方向，旋转180度

        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
        return R

    def caldose_raytrace_all(
        self,
        nFrac,
        beam,
        sadx,
        sady,
        ext_linear,
        ext_linear_opt,
        longitudalCutoff,
        bm: BEAM_MODEL,
        scan_points,
        rs: float = 0.0,
        mode="Coarse",
        startProgress: float = 0,
        deltaProgress: float = 1,
        calType="Opt",
        cal_fluence_map=True,
        dose_type="bio",
    ):
        function_start_time = time.time()
        x_all = []
        z_all = []
        ene_all = []
        spot_spacing_x_all = []
        spot_spacing_z_all = []

        for energy_idx in scan_points.keys():
            x_all.extend(scan_points[energy_idx]["x"])
            z_all.extend(scan_points[energy_idx]["z"])
            ene_all.extend(scan_points[energy_idx]["energy"])
            if "spot_spacing_x" in scan_points[energy_idx] and "spot_spacing_z" in scan_points[energy_idx]:
                spot_spacing_x_all.extend(scan_points[energy_idx]["spot_spacing_x"])
                spot_spacing_z_all.extend(scan_points[energy_idx]["spot_spacing_z"])

        npermu = bm.getnPerMUInterp(ene_all)
        nBeam = len(x_all)
        if spot_spacing_x_all or spot_spacing_z_all:
            if len(spot_spacing_x_all) != nBeam or len(spot_spacing_z_all) != nBeam:
                raise RuntimeError(
                    "spot_spacing_x/z length mismatch: spot_spacing_x={}, spot_spacing_z={}, nBeam={}".format(
                        len(spot_spacing_x_all),
                        len(spot_spacing_z_all),
                        nBeam,
                    )
                )
            spot_spacing_x_all = np.asarray(spot_spacing_x_all, dtype=np.float32)
            spot_spacing_z_all = np.asarray(spot_spacing_z_all, dtype=np.float32)
            if (not np.all(np.isfinite(spot_spacing_x_all))) or (not np.all(np.isfinite(spot_spacing_z_all))):
                raise RuntimeError("spot_spacing_x/z must be finite physical PB spacing values")
            if np.any(spot_spacing_x_all <= 0) or np.any(spot_spacing_z_all <= 0):
                raise RuntimeError("spot_spacing_x/z must be positive physical PB spacing values")
        else:
            spot_spacing_x_all = None
            spot_spacing_z_all = None
        nROI = np.size(ext_linear)
        sad = (sadx + sady) * 0.5

        # interpolation for weq
        xlim = np.max(np.abs(x_all) + 10).astype(int) + 1
        ylim = np.max(np.abs(z_all) + 10).astype(int) + 1
        minRes = np.min(self.doseGrid.resolution)
        interpy, interpx = np.meshgrid(np.arange(-ylim, ylim + 1), np.arange(-xlim, xlim + 1))
        # interpy, interpx = np.meshgrid(np.arange(-ylim, ylim + 1, minRes), np.arange(-xlim, xlim + 1, minRes))
        # interpx, interpy = np.meshgrid(np.arange(-xlim, xlim+1), np.arange(-ylim, ylim+1))
        interpSource = (sadx - sad) / sadx * interpx.flatten() * self.bmxdir
        interpSource = interpSource + (sady - sad) / sady * interpy.flatten() * self.bmydir
        interpSource = interpSource - sad * self.bmdir + self.isocenter
        interpBeamDir = self.isocenter + interpx.flatten() * self.bmxdir + interpy.flatten() * self.bmydir - interpSource
        bmZ = np.cross(np.squeeze(self.bmxdir), np.squeeze(self.bmydir))
        bmZ = bmZ / np.linalg.norm(bmZ)
        bmZ = bmZ.astype(np.float32)
        interpBeamDir = interpBeamDir / np.sqrt(np.sum(np.square(interpBeamDir), 0, keepdims=True))
        nMaxStep = 10000
        rayweq = np.zeros((9 + np.size(interpx) * nMaxStep,), dtype=np.float32)
        rayweq[3:9] = np.array([-ylim, 1, 2 * ylim + 1, -xlim, 1, 2 * xlim + 1])

        # if (bm.modality == PtclType.Protons):
        cuPrepareWEQ(
            rayweq,
            nMaxStep,
            interpSource.flatten(order="F"),
            interpBeamDir.flatten(order="F"),
            self.doseGrid.data,
            self.doseGrid.corner,
            self.doseGrid.resolution,
            self.doseGrid.dims,
            0,
        )

        idbeamxy = np.zeros((nBeam, 2))
        idbeamxy[:, 0] = np.array(x_all) + xlim + 0.5
        idbeamxy[:, 1] = np.array(z_all) + ylim + 0.5

        # Diagnostic logging: spot spacing, sample spot coords and idbeamxy
        try:
            first_energy = list(scan_points.keys())[0] if len(scan_points) > 0 else None
            if first_energy is not None and 'spot_spacing_x' in scan_points[first_energy]:
                self.logger.info("DIAG sample spot_spacing_x: %s", np.array(scan_points[first_energy]['spot_spacing_x'])[:5].tolist())
                self.logger.info("DIAG sample spot_spacing_z: %s", np.array(scan_points[first_energy]['spot_spacing_z'])[:5].tolist())
        except Exception as _e:
            self.logger.warning("DIAG unable to read spot_spacing from scan_points: %s", _e)
        try:
            self.logger.info("DIAG x_all sample: %s", np.array(x_all)[:20].tolist())
            self.logger.info("DIAG z_all sample: %s", np.array(z_all)[:20].tolist())
            self.logger.info("DIAG idbeamxy sample: %s", idbeamxy[:20].tolist())
        except Exception:
            pass
        try:
            header = rayweq[3:9]
            self.logger.info("DIAG rayweq header: %s", header.tolist())
        except Exception:
            pass

        sourcePos = (sadx - sad) / sadx * np.array(x_all) * self.bmxdir
        sourcePos = sourcePos + (sady - sad) / sady * np.array(z_all) * self.bmydir
        sourcePos = sourcePos - sad * self.bmdir + self.isocenter

        beamdir = self.isocenter + np.array(x_all) * self.bmxdir + np.array(z_all) * self.bmydir - sourcePos
        beamdir = beamdir / np.sqrt(np.sum(np.square(beamdir), 0, keepdims=True))

        roiIdx = np.array(np.unravel_index(ext_linear, self.doseGrid.dims))
        roiIdx = roiIdx.flatten(order="F")

        if mode == "Fine":
            cutoff = 0.0002
            sparse_ration = 1.0
        elif mode == "Coarse":
            cutoff = 0.0002
            sparse_ration = 0.20
        else:
            cutoff = 0.001

        crossCut = cutoff * np.ones((nBeam,))
        beamParaPos: float = 0.0  # 进入计算的beam parameter的虚拟面位置，相对于isocenter，iec坐标系
        # 实际默认为0，需测试对碳离子剂量计算的影响
        if bm.triGaussian["variableEnergyMode"] == "Discrete":
            enelist = np.array(bm.triGaussian["meaEneList"]).astype("float32")
            idddata = np.array(bm.triGaussian["dose"])
            idddepth = bm.triGaussian["IDDDepth"]
            iddsetting = np.array(
                [
                    idddepth[0] - rs,
                    idddepth[1] - idddepth[0],
                    np.size(idddepth),
                ]
            )
            profiledata = np.array(bm.triGaussian["profile"])
            profiledepth = bm.triGaussian["profileDepth"]
            profilesetting = np.array(
                [
                    profiledepth[0] - rs,
                    profiledepth[1] - profiledepth[0],
                    np.size(profiledepth),
                ]
            )
            beamparadata = bm.getBeamPara(bm.commissionLoc, bm.latPara, beamParaPos)
            beamparadata = beamparadata.astype(np.float32)
        elif bm.triGaussian["variableEnergyMode"] == "Continuous":
            enelist = np.unique(ene_all)
            idddata, profiledata, latparas, rsDeltaLatPara = bm.getIDDFromMeasureList(enelist)
            bm.rsDeltaLatPara = rsDeltaLatPara
            idddepth = bm.triGaussian["IDDDepth"]
            iddsetting = np.array(
                [
                    idddepth[0] - rs,
                    idddepth[1] - idddepth[0],
                    np.size(idddepth),
                ]
            )
            profiledepth = bm.triGaussian["profileDepth"]
            profilesetting = np.array(
                [
                    profiledepth[0] - rs,
                    profiledepth[1] - profiledepth[0],
                    np.size(profiledepth),
                ]
            )
            beamParaPos = 0
            beamparadata = bm.getBeamPara(bm.commissionLoc, latparas, beamParaPos)
        else:
            raise RuntimeError("Not supported variableEnergyType")

        if bm.modality == PtclType.Unknow:  # by default no subspots splitting in carbon therapy
            subspotdata = self.split_subspot(bm=bm, beamparadata=beamparadata, enelist=enelist)
        else:
            subspotdata = np.zeros((beamparadata.shape[0], 1, 5))  # for test 1
            subspotdata[:, 0, 2] = 1
            subspotdata[:, 0, 3] = np.sqrt(beamparadata[:, 0] / 2)
            subspotdata[:, 0, 4] = np.sqrt(beamparadata[:, 0] / 2)

        ctCubeSize = np.squeeze(self.doseGrid.resolution) * self.doseGrid.dims

        diameter = 80
        maxTraceSepth = int((ctCubeSize[0] ** 2 + ctCubeSize[1] ** 2 + ctCubeSize[2] ** 2) ** 0.5 / np.min(self.doseGrid.resolution))
        weq = np.zeros((nBeam, maxTraceSepth), dtype=np.float32)
        spacingRes = 1.5
        depthRes = minRes

        numVoxPerSpot = int(diameter / spacingRes + 1) ** 2 * int(weq.size / nBeam * np.min(self.doseGrid.resolution) / depthRes + 1)

        tmpSourcePos = sourcePos.flatten(order="F")
        self.sourcePos = tmpSourcePos[:3]
        tmpBeamDir = beamdir.flatten(order="F")
        layerEnergy = np.array([key for key, _ in groupby(ene_all)])
        layerInfo = np.array(
            [(np.sum(ene_all == element)) for element in layerEnergy],
            dtype=np.int32,
        )
        all_energies = np.array(ene_all)

        nnz_v = np.zeros((1,), dtype=np.uint64)
        beam["all_energies"] = all_energies
        beam["water_equivalence"] = rayweq
        beam["source_pos"] = self.sourcePos
        beam["beam_xdir"] = self.bmxdir
        beam["beam_ydir"] = self.bmydir
        beam["beam_dir"] = tmpBeamDir
        beam["longitudal_cutoff"] = longitudalCutoff
        beam["energy_list"] = enelist
        beam["idd_data"] = idddata
        beam["profile_data"] = profiledata
        beam["idd_setting"] = iddsetting
        beam["profile_setting"] = profilesetting
        beam["beam_para_data"] = beamparadata
        beam["subspot_data"] = subspotdata
        beam["layer_info"] = layerInfo
        beam["layer_energy"] = layerEnergy
        beam["sad"] = sad
        beam["npermu"] = npermu
        beam["number_particle"] = npermu * beam["weight_vector"]
        beam["idbeamxy"] = idbeamxy
        if spot_spacing_x_all is not None and spot_spacing_z_all is not None:
            beam["spot_spacing_x"] = spot_spacing_x_all
            beam["spot_spacing_z"] = spot_spacing_z_all
        if calType == "Dose" or calType == "QA" or calType == "Scale" or calType == "DoseRecalculation":
            finalDose = np.zeros(
                (
                    self.doseGrid.dims[0],
                    self.doseGrid.dims[1],
                    self.doseGrid.dims[2],
                ),
                dtype=np.float32,
            )
            start_time = time.time()
            if dose_type == "bio":
                # 由RBEcal.calFinalRBEMapAndDose计算final dose，避免重复计算
                end_time = time.time()
                duration = end_time - start_time
                csv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'cuFinalDose_timing.csv')
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                        duration,
                        beam.get('beamName', ''),
                        calType,
                        dose_type,
                        'skip_no_cuFinalDose',
                    ])
                return finalDose
            #########################################################################################
            # finalDose：   final dose counter（最终的剂量网格）
            # rayweq：      water equivalent matrix(每个spot的cumulate 水等效)
            # roiIdx:       region of interest index  (ROI格子对应的xyz下标索引)
            # all_energies: energies of each spot. vector size: (每个spot的能量)
            # sourcePos:    snout position.每个spot的源的位置）
            # tmpBeamDir:   direction of each spot（每个spot的方向）
            # self.bmxdir:  x axis direction of virtual plan in global coordinate system.（虚平面x轴在全局坐标系下的矢量）
            # self.bmydir:  y axis direction of virtual plan in global coordinate system.（虚平面y轴在全局坐标系下的矢量）
            # self.doseGrid.corner: grid corner.（剂量网格左下角在全局坐标系的位置）
            # self.doseGrid.resolution: grid resolution（剂量网格的分辨率）
            # self.doseGrid.dims: grid dimension（剂量网格的维度）
            # longitudalCutoff: longitudal cut off(这个能量能打到的最远距离的一个截断)
            # enelist: energy list of machine(机器能打的所有能量)
            # idddata: idd data(idd数据)
            # iddsetting: idd data(idd数据)
            # profiledata: profile data(profile数据)
            # profilesetting: profile data(profile数据)
            # beamparadata: beam model parameters(束流模型数据)
            # subspotdata: sub spots parameters(子束分解sigma数据)
            # layerInfo: spots number of each energy layer(每个能量层打多少个点的数据)
            # layerEnergy: energy of each energy layer.(每个能量层的能量)
            # idbeamxy：子束在V平面上的位置
            # nPar：这是一个（nBeam, ）长的向量，记录每个spot打了多少粒子，也就是weights * 标定因子 ions/MU
            # sad：sad距离
            # 0.0: gaussian weight 的 cut off 设置成0
            # beamParaPos:暂时不需要关心
            # 0：使用0号GPU
            #########################################################################################
            if rayweq[2] < 0:
                error_message = f"Beam {beam['beamName']} does not intersect the patient outline."
                raise RuntimeError(error_message)

            nPtlcsPerBm = (npermu * beam["weight_vector"] / nFrac).astype(np.int32)
            # 裁剪到 int32 有效范围
            int32_max = np.iinfo(np.int32).max  # 2147483647
            tmpNperMin = 0
            numParticlesPerBeam = np.clip(nPtlcsPerBm, tmpNperMin, int32_max).astype(np.int32)

            start_time = time.time()
            cuFinalDose(
                finalDose,
                rayweq,
                roiIdx,
                all_energies,
                self.sourcePos,
                tmpBeamDir,
                self.bmxdir,
                self.bmydir,
                self.doseGrid.corner,
                self.doseGrid.resolution,
                self.doseGrid.dims,
                longitudalCutoff,
                enelist,
                idddata,
                iddsetting,
                profiledata,
                profilesetting,
                beamparadata,
                subspotdata,
                layerInfo,
                layerEnergy,
                idbeamxy,
                numParticlesPerBeam,
                sad,
                0.00005,
                beamParaPos,
                0,
                spotSpacingX=spot_spacing_x_all,
                spotSpacingZ=spot_spacing_z_all,
            )
            end_time = time.time()
            duration = end_time - start_time
            csv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'cuFinalDose_timing.csv')
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                    duration,
                    beam.get('beamName', ''),
                    calType,
                    dose_type,
                    'cuFinalDose',
                ])
            if np.any(finalDose < 0):
                raise ValueError("final dose negative!")
            return finalDose
        elif not cal_fluence_map:
            # 不在这里计算fluence_map, 只计算每行的norm
            # cutoff = 0.0001
            cutoff = 0.00005  # 和finaldose一致
            roiIdx = np.array(np.unravel_index(ext_linear_opt, self.doseGrid.dims))
            roiIdx = roiIdx.flatten(order="F")
            nROI = np.size(ext_linear_opt)
            grad_norm = np.zeros(nROI, dtype=np.float32)
            succ = cuCalDoseNorm(
                grad_norm,
                rayweq,
                roiIdx,
                all_energies,
                self.sourcePos,
                tmpBeamDir,
                self.bmxdir,
                self.bmydir,
                self.doseGrid.corner,
                self.doseGrid.resolution,
                self.doseGrid.dims,
                longitudalCutoff,
                enelist,
                idddata,
                iddsetting,
                profiledata,
                profilesetting,
                beamparadata,
                subspotdata,
                layerInfo,
                layerEnergy,
                idbeamxy,
                sad,
                cutoff,
                beamParaPos,
                0,
            )
            end_time = time.time()
            duration = end_time - function_start_time
            csv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'cuFinalDose_timing.csv')
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                    duration,
                    beam.get('beamName', ''),
                    calType,
                    dose_type,
                    'opt_grad_norm',
                ])
            return grad_norm
        else:
            # 只生成优化相关的voxel的fluence_map
            roiIdx = np.array(np.unravel_index(ext_linear_opt, self.doseGrid.dims))
            roiIdx = roiIdx.flatten(order="F")
            nROI = np.size(ext_linear_opt)
            while True:
                cscValues = np.zeros(
                    (int(numVoxPerSpot * nBeam * sparse_ration),),
                    dtype=np.float32,
                )
                cscRowInd = np.zeros(
                    (int(numVoxPerSpot * nBeam * sparse_ration),),
                    dtype=np.int32,
                )
                cscPtr = np.zeros((int(nBeam + 1),), dtype=np.int32)
                succ = cuCalDose3(
                    cscValues,
                    cscPtr,
                    cscRowInd,
                    rayweq,
                    roiIdx,
                    all_energies,
                    self.sourcePos,
                    tmpBeamDir,
                    self.bmxdir,
                    self.bmydir,
                    self.doseGrid.corner,
                    self.doseGrid.resolution,
                    self.doseGrid.dims,
                    longitudalCutoff,
                    enelist,
                    idddata,
                    iddsetting,
                    profiledata,
                    profilesetting,
                    beamparadata,
                    subspotdata,
                    layerInfo,
                    layerEnergy,
                    nnz_v,
                    idbeamxy,
                    sad,
                    0.00005,
                    beamParaPos,
                    cscValues.size,
                    0,
                )

                cscPtr = cscPtr.astype(np.int64)
                cscPtr = np.cumsum(cscPtr)

                if nnz_v[0] > 0 and succ:
                    cscValues = cscValues[: nnz_v[0]]
                    cscRowInd = cscRowInd[: nnz_v[0]]
                    fluenceMap = csc_matrix((cscValues, cscRowInd, cscPtr), shape=(nROI, nBeam))
                    del cscValues, cscRowInd, cscPtr
                    gc.collect()
                    break
                else:
                    sparse_ration += 0.2

                if nnz_v[0] == 0:
                    self.logger.error("There is no dose deposition within the region of interest. Please check the validity of the plan.")
                    raise RuntimeError("There is no dose deposition within the region of interest. Please check the validity of the plan.")

        end_time = time.time()
        duration = end_time - function_start_time
        csv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'cuFinalDose_timing.csv')
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                duration,
                beam.get('beamName', ''),
                calType,
                dose_type,
                'opt_fluence_map',
            ])

        for colidx in range(fluenceMap.shape[1]):
            ind1 = fluenceMap.indptr[colidx]
            ind2 = fluenceMap.indptr[colidx + 1]
            if ind2 <= ind1:
                continue
            threshold = np.max(fluenceMap.data[ind1:ind2]) * 0.00005
            fluenceMap.data[ind1:ind2] = fluenceMap.data[ind1:ind2] * (fluenceMap.data[ind1:ind2] > threshold) * npermu[colidx]

        print("*************", np.amax(fluenceMap.data))
        # return fluenceMap, cscValues, cscRowInd, cscPtr
        return fluenceMap

    # 函数错误 wedcross未定义
    # def caldose_raytrace_all_proton(self, sadx, sady, ext_linear, longitudalCutoff, bm: BEAM_MODEL, scan_points, rs=0,
    #                                 mode="Coarse", startProgress=0, deltaProgress=1):
    #     try:
    #         subspotdata = bm.triGaussian["subspotData"]
    #     except:
    #         self.logger.warning("No subspot data in beam model. Assume no subspot.")
    #         # subspotdata = np.zeros((beamparadata.shape[0], 1, 5))  # for test 1
    #         # subspotdata[:, 0, 2] = 1
    #         # subspotdata[:, 0, 3] = np.sqrt(beamparadata[:, 0] / 2)
    #         # subspotdata[:, 0, 4] = np.sqrt(beamparadata[:, 0] / 2)
    #         # subspotdata = np.zeros((beamparadata.shape[0], 2, 5)) # for test 2
    #         # subspotdata[:, 0, 2] = 0.5
    #         # subspotdata[:, 0, 3] = np.sqrt(beamparadata[:,0]/2)
    #         # subspotdata[:, 0, 4] = np.sqrt(beamparadata[:,0]/2)
    #         # subspotdata[:, 1, 2] = 0.5
    #         # subspotdata[:, 1, 3] = np.sqrt(beamparadata[:,0]/2)
    #         # subspotdata[:, 1, 4] = np.sqrt(beamparadata[:,0]/2)

    #     x_all = []
    #     z_all = []
    #     ene_all = []
    #     for energy_idx in list(scan_points.keys())[0:1]:
    #         x_all.extend(scan_points[energy_idx]['x'])
    #         z_all.extend(scan_points[energy_idx]['z'])
    #         # ene_all.extend(scan_points[energy_idx]['energy'])
    #         ene_all.extend(len(scan_points[energy_idx]['x']) * [70])
    #         # non_zero_subspot = np.where(subspotdata[energy_idx, :, 2] != 0)
    #         # for sub_idx in range(len(scan_points[energy_idx]['x'])):
    #         #     x_sub_all.extend(subspotdata[energy_idx, non_zero_subspot, 0].squeeze() + scan_points[energy_idx]['x'][sub_idx])
    #         #     z_sub_all.extend(subspotdata[energy_idx, non_zero_subspot, 1].squeeze() + scan_points[energy_idx]['z'][sub_idx])
    #         #     ene_sub_all.extend(np.ones(len(non_zero_subspot),) * ene_all[0])

    #     npermu = bm.getnPerMUInterp(ene_all)
    #     nBeam = len(x_all)
    #     nROI = np.size(ext_linear)
    #     sad = (sadx + sady) * 0.5

    #     # interpolation for weq
    #     # xlim = np.max(np.abs(x_all) + 10).astype(int) + 1
    #     # ylim = np.max(np.abs(z_all) + 10).astype(int) + 1
    #     # interpy, interpx = np.meshgrid(np.arange(-ylim, ylim + 1), np.arange(-xlim, xlim + 1))
    #     # # interpx, interpy = np.meshgrid(np.arange(-xlim, xlim+1), np.arange(-ylim, ylim+1))
    #     # interpSource = (sadx - sad) / sadx * interpx.flatten() * self.bmxdir
    #     # interpSource = interpSource + (sady - sad) / sady * interpy.flatten() * self.bmydir
    #     # interpSource = interpSource - sad * self.bmdir + self.isocenter
    #     # interpBeamDir = self.isocenter + interpx.flatten() * self.bmxdir + interpy.flatten() * self.bmydir - interpSource
    #     # interpBeamDir = interpBeamDir / np.sqrt(np.sum(np.square(interpBeamDir), 0, keepdims=True))
    #     # nMaxStep = 10000
    #     # rayweq = np.zeros((9 + np.size(interpx) * nMaxStep,), dtype=np.float32)
    #     # rayweq[3:9] = np.array([-ylim, 1, 2 * ylim + 1, -xlim, 1, 2 * xlim + 1])
    #     # start = time.time()
    #     # cuPrepareWEQ(rayweq, nMaxStep, interpSource.flatten(order="F"), interpBeamDir.flatten(order="F"),
    #     #              self.doseGrid.data,
    #     #              self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, 0)
    #     # print("cuPrepareWEQ took {}".format((time.time() - start)))
    #     ####
    #     # interpolation for weq subspots.
    #     # xlim = np.max(np.abs(x_sub_all) + 10).astype(int) + 1
    #     # ylim = np.max(np.abs(z_sub_all) + 10).astype(int) + 1
    #     # interpy, interpx = np.meshgrid(np.arange(-ylim, ylim + 1), np.arange(-xlim, xlim + 1))
    #     # # interpx, interpy = np.meshgrid(np.arange(-xlim, xlim+1), np.arange(-ylim, ylim+1))
    #     # interpSource = (sadx - sad) / sadx * interpx.flatten() * self.bmxdir
    #     # interpSource = interpSource + (sady - sad) / sady * interpy.flatten() * self.bmydir
    #     # interpSource = interpSource - sad * self.bmdir + self.isocenter
    #     # interpBeamDir = self.isocenter + interpx.flatten() * self.bmxdir + interpy.flatten() * self.bmydir - interpSource
    #     # interpBeamDir = interpBeamDir / np.sqrt(np.sum(np.square(interpBeamDir), 0, keepdims=True))
    #     # nMaxStep = 10000
    #     # rayweq_sub = np.zeros((9 + np.size(interpx) * nMaxStep,), dtype=np.float32)
    #     # rayweq_sub[3:9] = np.array([-ylim, 1, 2 * ylim + 1, -xlim, 1, 2 * xlim + 1])
    #     # start = time.time()
    #     # cuPrepareWEQ(rayweq_sub, nMaxStep, interpSource.flatten(order="F"), interpBeamDir.flatten(order="F"),
    #     #              self.doseGrid.data,
    #     #              self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, 0)
    #     # print("cuPrepareWEQ took {}".format((time.time() - start)))

    #     # idbeamxy = np.zeros((nBeam, 2))
    #     # idbeamxy[:, 0] = np.array(x_all) + xlim + 0.5
    #     # idbeamxy[:, 1] = np.array(z_all) + ylim + 0.5
    #     #
    #     # sourcePos = (sadx - sad) / sadx * np.array(x_all) * self.bmxdir
    #     # sourcePos = sourcePos + (sady - sad) / sady * np.array(z_all) * self.bmydir
    #     # sourcePos = sourcePos - sad * self.bmdir + self.isocenter
    #     #
    #     # beamdir = self.isocenter + np.array(x_all) * self.bmxdir + np.array(z_all) * self.bmydir - sourcePos
    #     # beamdir = beamdir / np.sqrt(np.sum(np.square(beamdir), 0, keepdims=True))
    #     #
    #     # roiIdx = np.array(np.unravel_index(ext_linear, self.doseGrid.dims))
    #     # roiIdx = roiIdx.flatten(order="F")  # test it!

    #     if (mode == "Fine"):
    #         cutoff = 0.00005
    #     elif (mode == "Coarse"):
    #         cutoff = 0.0002
    #     else:
    #         cutoff = 0.001
    #     crossCut = cutoff * np.ones((nBeam,))

    #     enelist = np.array(bm.triGaussian['meaEneList']).astype('float32')
    #     idddata = np.array(bm.triGaussian["dose"])
    #     idddepth = bm.triGaussian["IDDDepth"]
    #     iddsetting = np.array([idddepth[0] - rs, idddepth[1] - idddepth[0], np.size(idddepth)])
    #     profiledata = np.array(bm.triGaussian["latPara"])
    #     # profiledepth = bm.triGaussian["profileDepth"]
    #     # profilesetting = np.array([profiledepth[0] - rs, profiledepth[1] - profiledepth[0], np.size(profiledepth)])
    #     # beamparadata = bm.getBeamPara(bm.commissionLoc, bm.latPara, 150)
    #     nGauss = bm.triGaussian["nGauss"]

    #     fluence_map = np.zeros((nROI, nBeam))
    #     for spot_i in range(len(x_all)):
    #         energy = ene_all[spot_i]
    #         energy_idx = np.where(enelist == energy)
    #         # Get x,y coords of subspots relative to main spot.
    #         non_zero_subspot = np.where(subspotdata[energy_idx, :, 2].squeeze() != 0)
    #         x_sub = subspotdata[energy_idx, non_zero_subspot, 0].squeeze()
    #         y_sub = subspotdata[energy_idx, non_zero_subspot, 1].squeeze()
    #         # absolute x,y coords of subspots
    #         x_sub_all = x_all[spot_i] + x_sub
    #         z_sub_all = z_all[spot_i] + y_sub

    #         xlim = np.max(np.abs(x_sub_all) + 10).astype(int) + 1
    #         ylim = np.max(np.abs(z_sub_all) + 10).astype(int) + 1
    #         interpy, interpx = np.meshgrid(np.arange(-ylim, ylim + 1), np.arange(-xlim, xlim + 1))
    #         # interpx, interpy = np.meshgrid(np.arange(-xlim, xlim+1), np.arange(-ylim, ylim+1))
    #         interpSource = (sadx - sad) / sadx * interpx.flatten() * self.bmxdir
    #         interpSource = interpSource + (sady - sad) / sady * interpy.flatten() * self.bmydir
    #         interpSource = interpSource - sad * self.bmdir + self.isocenter
    #         interpBeamDir = self.isocenter + interpx.flatten() * self.bmxdir + interpy.flatten() * self.bmydir - interpSource
    #         interpBeamDir = interpBeamDir / np.sqrt(np.sum(np.square(interpBeamDir), 0, keepdims=True))
    #         nMaxStep = 10000
    #         rayweq_sub = np.zeros((9 + np.size(interpx) * nMaxStep,), dtype=np.float32)  # TODO  9mm?
    #         rayweq_sub[3:9] = np.array([-ylim, 1, 2 * ylim + 1, -xlim, 1, 2 * xlim + 1])
    #         start = time.time()
    #         cuPrepareWEQ(rayweq_sub, nMaxStep, interpSource.flatten(order="F"), interpBeamDir.flatten(order="F"),
    #                      self.doseGrid.data,
    #                      self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims, 0)

    #         idd = np.interp(rayweq_sub[:100], bm.triGaussian["IDDDepth"], bm.triGaussian["IDDData"][spot_i, :])

    #         radialR = wedcross

    #         gaussianWeight = np.zeros(radialR.shape)
    #         tmpdose = np.zeros(np.shape(idd * gaussianWeight))
    #         profilePara = profiledata[energy_idx]

    #         # sub_xi, sub_zi = x_sub_all[spot_i], z_sub_all[spot_i]
    #         # [sub_wi, sub_sigmaxi, sub_sigmayi] = subspotdata[energy_idx, spot_i, 2:5]
    #         for iGauss in range(nGauss):
    #             wiGauss = subspotdata[energy_idx, spot_i, iGauss * 5 + 2]
    #             sigma_x = 1e-4 * profilePara[energy_idx, 0, iGauss * 3] * rayweq_sub[energy_idx,
    #                                                                       :] ** 2 + 1e-2 * profilePara[
    #                           energy_idx, 0, iGauss * 3 + 1] * rayweq_sub[energy_idx, :] + \
    #                       profilePara[energy_idx, 0, iGauss * 3 + 2]
    #             sigma_y = 1e-4 * profilePara[energy_idx, 1, iGauss * 3] * rayweq_sub[energy_idx,
    #                                                                       :] ** 2 + 1e-2 * profilePara[
    #                           energy_idx, 1, iGauss * 3 + 1] * rayweq_sub[energy_idx, :] + \
    #                       profilePara[energy_idx, 1, iGauss * 3 + 2]
    #             gaussianWeight = gaussianWeight + calGaussiantwoRs(x=radialR, y=radialR, sigma_x=sigma_x,
    #                                                                sigma_y=sigma_y,
    #                                                                mu_x=x_sub_all, mu_y=z_sub_all) * wiGauss
    #         tmpdose += bm.triGaussian["nPerMU"] * idd * gaussianWeight
    #         fluence_map[:, spot_i] = tmpdose

    #     # relpos = self.doseGrid.getPosFromLinearIndex(ext_linear) - self.isocenter
    #     # estSparseRatio = 0.1 + np.sum(
    #     #     np.sum(np.square(np.cross(self.bmdir, relpos, axis=0)), axis=0) < 20 * np.max(beamparadata[:, 0])) / nROI
    #     # self.logger.info("estSparseRatio: {}".format(estSparseRatio))
    #     # if (estSparseRatio > 1):
    #     #     estSparseRatio = 1
    #     # # estSparseRatio = 0.16
    #     # maxnnz = nROI * nBeam * estSparseRatio
    #     # nGroup = np.int((maxnnz * 4 * 1.5) // (1024 * 1024 * 1024)) + 1
    #     # nPerGroup = np.int(nBeam // nGroup + 1)
    #     # spotDoseValues = np.array([], dtype=np.float32)
    #     # spotDoseRowindices = np.array([], dtype=np.int32)
    #     # spotDoseColptrs = np.zeros((nBeam + 1,), dtype=np.int64)
    #     # currentNNZ = 0
    #     # estnnz = int(nROI * nPerGroup * estSparseRatio)
    #     # dosedata = np.zeros((estnnz,), dtype=np.float32)
    #     # doseindices = np.zeros((estnnz + nPerGroup + 3,), dtype=np.int32)
    #     # self.logger.info("total beam {} in {} batch".format(nBeam, nGroup))
    #     # for i in range(nGroup):
    #     #     self.logger.info("calculating the {}/{}th batch".format(i, nGroup))
    #     #     NetWork.sendStatus(None, self.task_id, "PBS calculating the {}/{}th batch".format(i, nGroup),
    #     #                        startProgress + i / nGroup * deltaProgress)
    #     #
    #     #     remainder = nBeam - i * nPerGroup
    #     #     if (remainder <= 0):
    #     #         break
    #     #     if (remainder > nPerGroup):
    #     #         remainder = nPerGroup
    #     #
    #     #     tmpene = np.array(ene_all[nPerGroup * i:nPerGroup * i + remainder])
    #     #     tmpsourcepos = sourcePos[:, nPerGroup * i:nPerGroup * i + remainder].flatten(order="F")
    #     #     tmpbmdir = beamdir[:, nPerGroup * i:nPerGroup * i + remainder].flatten(order="F")
    #     #     tmpcross = crossCut[nPerGroup * i:nPerGroup * i + remainder]
    #     #     tmplong = longitudalCutoff[nPerGroup * i:nPerGroup * i + remainder]
    #     #     # cuCalDose(dosedata, doseindices, tmpene, tmpsourcepos,
    #     #     #         tmpbmdir, self.doseGrid.data, self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims,
    #     #     #         roiIdx,  tmpcross, tmplong,
    #     #     #         enelist, idddata, iddsetting, profiledata, profilesetting, beamparadata, estSparseRatio, 0)
    #     #     tmpidbeamxy = idbeamxy[nPerGroup * i:nPerGroup * i + remainder, :]
    #     #     cuCalDose(dosedata, doseindices, tmpene, tmpsourcepos, tmpbmdir, self.bmxdir, self.bmydir,
    #     #               rayweq, tmpidbeamxy, self.doseGrid.corner, self.doseGrid.resolution, self.doseGrid.dims,
    #     #               roiIdx, tmpcross, tmplong,
    #     #               enelist, idddata, iddsetting, profiledata, profilesetting, beamparadata, subspotdata, sad,
    #     #               estSparseRatio, 0)
    #     #     nnz = doseindices[0]
    #     #     spotDoseValues = np.concatenate((spotDoseValues, dosedata[0:nnz]))
    #     #     spotDoseRowindices = np.concatenate((spotDoseRowindices, doseindices[2 + remainder:2 + remainder + nnz]))
    #     #     spotDoseColptrs[i * nPerGroup + 1:1 + i * nPerGroup + remainder] = currentNNZ + (
    #     #     doseindices[2:1 + remainder + 1]).astype(np.int64)
    #     #     currentNNZ = currentNNZ + nnz
    #     #
    #     # fluenceMap = csc_matrix((spotDoseValues, spotDoseRowindices, spotDoseColptrs), shape=(nROI, nBeam))
    #     #
    #     # for colidx in range(fluenceMap.shape[1]):
    #     #     ind1 = fluenceMap.indptr[colidx]
    #     #     ind2 = fluenceMap.indptr[colidx + 1]
    #     #     if (ind2 <= ind1):
    #     #         continue
    #     #     threshold = np.max(fluenceMap.data[ind1:ind2]) * 0.00005
    #     #     fluenceMap.data[ind1:ind2] = fluenceMap.data[ind1:ind2] * (fluenceMap.data[ind1:ind2] > threshold) * npermu[
    #     #         colidx]
    #     #     # fluenceMap.data[ind1:ind2] = fluenceMap.data[ind1:ind2]**npermu[colidx]
    #     # fluenceMap.eliminate_zeros()
    #     return fluence_map

    def split_sub_spots(self, scan_points):
        """
        Split a single spot into multiple sub spots. Used for proton PBS.
        """

        return scan_points

    def compute_single_beam(
        self,
        nFrac,
        beam,
        bm: BEAM_MODEL,
        ext_contour_linear,
        ext_contour_linear_opt,
        particle_type,
        cal_mode="Coarse",
        startProgress=0.0,
        deltaProgress=100.0,
        calType="Opt",
        cal_fluence_map=True,
        dose_type="bio",
    ):
        scan_points = beam.get("scan_points", None)
        if scan_points is None:
            self.logger.error("Beam ID {} dose not have attribute 'scan_points'.".format(beam["beamName"]))
            raise RuntimeError("Beam '{}' dose not have attribute 'scan_points'.".format(str(beam["beamName"])))

        __scan_points = scan_points

        # 获取 hit target 的水等效
        theta_all = []
        phi_all = []
        ene_all = []
        for energy_idx in __scan_points.keys():
            self.logger.info("Beam ID {} merging energy index {}".format(beam["beamName"], str(energy_idx)))
            theta_all.extend(__scan_points[energy_idx]["theta"])
            phi_all.extend(__scan_points[energy_idx]["phi"])
            ene_all.extend(__scan_points[energy_idx]["energy"])

        isocenter = beam.get("isocenter", None)
        if isocenter is None:
            self.logger.warning("Beam ID {} dose not have isocenter. Cannot proceed.".format(beam["beamName"]))
            raise RuntimeError("Beam '{}' dose not have isocenter. Cannot proceed.".format(str(beam["beamName"])))

        SAD = np.sum(np.abs(beam["source_pos"]) * 0.5)
        SADX = np.abs(beam["source_pos"][1])
        SADY = np.abs(beam["source_pos"][2])
        self.setGeometry(
            beam["gantry_angle"],
            beam["couch_angle"],
            [0, 0, 0],
            isocenter,
            SAD,
        )

        if self.serviceMode:
            NetWork.sender.sendStatus(
                None,
                self.task_id,
                "Calculating WEQ for beam '{}'".format(beam["beamName"]),
                startProgress + deltaProgress * 0.2,
            )

        longitudalCutoff = np.interp(ene_all, bm.triGaussian["meaEneList"], bm.triGaussian["R80"] * 2)
        longitudalCutoff1 = np.interp(ene_all, bm.triGaussian["meaEneList"], bm.triGaussian["R80"] + 100)
        tmpidx = longitudalCutoff < longitudalCutoff1
        longitudalCutoff[tmpidx] = longitudalCutoff1[tmpidx]

        try:
            rs_setting = beam.get("rs_setting")
            bm.setRS(rs_setting)
            rs = rs_setting["rs_weq"]
        except:
            rs = beam.get("range_shifter", 0)
        cal_result = self.caldose_raytrace_all(
            nFrac,
            beam,
            SAD,
            SAD,
            ext_contour_linear,
            ext_contour_linear_opt,
            longitudalCutoff,
            bm,
            __scan_points,
            rs=rs,
            mode=cal_mode,
            startProgress=startProgress + deltaProgress * 0.21,
            deltaProgress=deltaProgress * 0.7,
            calType=calType,
            cal_fluence_map=cal_fluence_map,
            dose_type=dose_type,
        )
        # fluence_map = self.caldose_raytrace_all_proton(SADX, SADY, ext_contour_linear, longitudalCutoff, bm, __scan_points,
        #                                         rs=rs, mode=cal_mode,
        #                                         startProgress=startProgress + deltaProgress * 0.21,
        #                                         deltaProgress=deltaProgress * 0.7)

        # transverseCutoff = -1
        # if(cal_mode=="Coarse"):
        #     transverseCutoff = 15

        # self.logger.info('Calculating water equivalent matrices for beam ID {}.'.format(beam["beamName"]))
        # amatrix, bmatrix = self.getRayAccumulatedWEQ_GPU(theta_all, phi_all, ext_contour_linear,  transverseCutoff=transverseCutoff, crossStep=0.2, parallelStep=0.5, ene=ene_all, \
        #     longitudalCutoff=longitudalCutoff,startProgress=startProgress+deltaProgress*0.21,endProgress=startProgress+deltaProgress*0.49)
        # # -1 means automatic cutoff

        # if self.serviceMode:
        #     NetWork.sender.sendStatus(None, self.task_id, "Calculating dose map for beam '{}'".format(beam["beamName"]), startProgress+deltaProgress*0.5)

        # __rs = beam.get("range_shifter",0)
        # self.logger.info('Calculating dose for beam ID {}.'.format(beam["beamName"]))
        # fluence_map = self.caldose_raytrace(bm, scan_points, amatrix, bmatrix, rs=__rs, mode=cal_mode, startProgress=startProgress+deltaProgress*0.51,endProgress=startProgress+deltaProgress*0.95) #bmatrix#
        # del amatrix, bmatrix

        self.logger.info("Gathering data for beam ID {}.".format(beam["beamName"]))
        attribute_lists = [
            "x",
            "z",
            "theta",
            "phi",
            "energy",
            "range",
            "scaleFactor",
            "spot_spacing_x",
            "spot_spacing_z",
            "weight_vector",
        ]
        for attribute in attribute_lists:
            beam[attribute] = []
            energy_list = np.sort(list(scan_points.keys()))[::-1]  # make sure descending order
            for energy_idx in energy_list:
                if attribute in scan_points[energy_idx].keys():
                    beam[attribute].extend(scan_points[energy_idx][attribute])
                else:
                    beam[attribute].extend([-1] * len(scan_points[energy_idx]["x"]))
            beam[attribute] = np.array(beam[attribute])

        if "weight_vector" not in beam.keys():
            beam["weight_vector"] = np.ones((beam["spotId"].size,))
            self.logger.warning("Beam ID {} dose not have attribute 'weight_vector', set to ones.".format(beam["beamName"]))
        if calType in ["Dose", "QA", "Scale", "DoseRecalculation"]:
            beam["final_dose"] = cal_result * nFrac
        else:
            if cal_fluence_map:
                beam["fluence_map"] = cal_result
            else:
                beam["fluence_map_row_norm"] = cal_result
            # beam['csc_values'] = cal_result[1]
            # beam['csc_row_ind'] = cal_result[2]
            # beam['csc_col_ptr'] = cal_result[3]

        self.logger.info("Dose calculation finished for beam ID {}.".format(beam["beamName"]))

        return beam
