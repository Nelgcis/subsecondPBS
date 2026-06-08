# -*- coding: utf-8 -*-

'''
版权说明：
    版权所有（c）2025，国科离子医疗科技有限公司，保留所有权利

版本号：1.0.0
生成日期：
作者：

修改日志：
    2024/09/28 - 李晶 - other: 修改最小MU的删点逻辑
    2024/10/21 - 李晶 - other: 解决网页和脚本计算结果不一致的问题
    2025/03/06 - 李晶 - feat: 用fluence_map对voxel聚类 - 修改优化fluence_map计算
    2025/03/17 - 李晶 - feat: 用HU和frac对voxel聚类，优化过程降维
    2025/04/17 - 李晶 - feat: 优化过程降维 - 处理没存参与优化的index的情况
'''

import sys
import numpy as np
import json

# from youfang.plot import calType

sys.path.append('..')
from content.tools.file_tools import load_json

from content.base_operator.base_operator import BASE
from content.tools.network.cashim_net import NetWork
from content.rbe_operator.rbecal import RBEcal
from content.dose_operator.dosecal import DOSECAL
from content.bm_operator.beamModel import BEAM_MODEL
from content.ct_operator.ctProcess import CT_PROCESS
from content.dose_operator.mcdose import MCDOSE

# forward dose calculation, one time
class FINALDOSE(BASE):
    def __init__(self, serviceMode = False, name='FINALDOSE'):
        super(FINALDOSE, self).__init__(name=name)
        self.serviceMode = serviceMode
        self.maxUncertainty = 0
        self.totalSimN = 0
        self.bm = BEAM_MODEL()
        self.mc = MCDOSE()
        self.rbecal = RBEcal()
    
    ######################################
    #2024.06.04 tlchen@cashim.cn
    #去掉本类中的rbe_method读取。全部在rbe_cal里执行
    ######################################
    def process(self, plan_info:dict, dc:DOSECAL, ct:CT_PROCESS, pbs_resolution="Coarse", task_id="123", 
                startProg=0, deltaProg=1, updateBeamInfoFlag=False):
        self.task_id = task_id
        self.dc = dc
        ext_name = plan_info["body"]
        self.ext_contour_linear = self.dc._normalize_linear_indices(
            "ct.tissues[{}]['VOI_indices_linear']".format(ext_name),
            ct.tissues[ext_name]["VOI_indices_linear"],
        )
        ct.tissues[ext_name]["VOI_indices_linear"] = self.ext_contour_linear
        # 参与优化的index
        if "VOI_indices_linear_opt" in ct.tissues[ext_name].keys():
            self.ext_contour_linear_opt = self.dc._normalize_linear_indices(
                "ct.tissues[{}]['VOI_indices_linear_opt']".format(ext_name),
                ct.tissues[ext_name]["VOI_indices_linear_opt"],
            )
            ct.tissues[ext_name]["VOI_indices_linear_opt"] = self.ext_contour_linear_opt
        else:
            # 对于没有存参与优化的index的情况
            self.ext_contour_linear_opt = self.ext_contour_linear
        # 优化过程是否计算fluence_map，默认为计算
        self.cal_fluence_map = ct.tissues[ext_name].get('cal_fluence_map',True)
        self.table = dict()
        self.table["HU_to_density"] = ct.densityTable
        self.table["HU_to_spr"] = ct.SPRTable

        dose_cal_config = plan_info["dose_cal_config"]
        self.pbs_resolution = pbs_resolution
        self.clinical_factor = dose_cal_config["clinical_factor"] 
        self.particle_type = dose_cal_config["particle_type"]
        #self.rbe_method = dose_cal_config.get("rbe_method","lq")

        self.logger.info("Particle type {} running.".format(self.particle_type))
        self.dose_type = dose_cal_config["phys_bio"]
        # self.dose_type = "phys"
        self.dose_engine = dose_cal_config["phy_dose_engine"]
        self.nuclear_correction = self.dc._coerce_bool_flag(
            "dose_cal_config['nuclear_correction']",
            dose_cal_config.get("nuclear_correction", False),
        )


        if (self.dose_engine == "monte_carlo"):
            # mc_config["nIonPerSpot"] = 100000
            # mc_config["uncertainty"] = 0.005
            mc_config = plan_info["mc_config"]
            self.nIonPerSpot = mc_config.get("nIonPerSpot", 0)
            self.uncertaintyLevel = mc_config.get("uncertainty", 0.02)
            self.ifSpots = mc_config.get("ifSpots", 1)

            if plan_info['beam_group_config']['calculation_type'] in ['Dose','Scale'] and plan_info['dose_cal_config'][
                'particle_type'] == 'Protons':
                self.ifSpots = 0

            self.ifWater = mc_config.get("ifWater", 0)

        
        self.logger.info("Read-in beam info file")
        self.updateBeaminfoFromFile(plan_info)
        
        if(updateBeamInfoFlag):
            return
        NetWork.sendStatus(None, self.task_id, "Read beam info file done.", startProg+0.05*deltaProg)

        if (self.dose_type == "bio" and plan_info["beam_group_config"]["calculation_type"] != "RBE"):
            NetWork.sendStatus(None, self.task_id, "Start calculating phy dosemap.", startProg+0.1*deltaProg)
            self.cal_phy_dose(plan_info, startProg=startProg+0.1*deltaProg, deltaProg=0.4*deltaProg, calType=plan_info["beam_group_config"]["calculation_type"])
            # if self.particle_type == 'Carbon' or plan_info["beam_group_config"]["calculation_type"] == 'Opt':
            NetWork.sendStatus(None, self.task_id, "Intialization for calculating RBE.", startProg+0.5*deltaProg)
            self.rbecal.process(plan_info, ct=ct, dc=dc)

            NetWork.sendStatus(None, self.task_id, "RBE calculation in progress in GPU. Please wait if no error.",
                               startProg + 0.6 * deltaProg)
            self.cal_RBE(plan_info, startProgress=startProg + 0.65 * deltaProg, calType=plan_info["beam_group_config"]["calculation_type"])
        else:
            NetWork.sendStatus(None, self.task_id, "Start calculating phy dosemap.", startProg+0.1*deltaProg)
            self.cal_phy_dose(plan_info, startProg=startProg+0.15*deltaProg, deltaProg=0.7*deltaProg, calType=plan_info["beam_group_config"]["calculation_type"])

            beams = plan_info["beams"]
            for beamid in beams.keys():
                beam = beams[beamid]
                beam["rbe_map"] = np.ones(dc.doseGrid.data.shape)

        return_message = ""
        return return_message, self.totalSimN, self.maxUncertainty

    def updateRBE(self, plan_info:dict, dc:DOSECAL, ct:CT_PROCESS, task_id="123"): # for RBE test only
        self.task_id = task_id
        self.rbecal.process(plan_info, ct=ct, dc=dc)
        self.cal_RBE(plan_info, startProgress = 0)

    def cal_phy_dose(self, plan_info, startProg=0, deltaProg=100, calType='Opt'):
        __beam_count = 0
        __num_beam = len(plan_info.keys())
        delta_progress = deltaProg / __num_beam
        beams = plan_info["beams"]
        if calType == 'QA':
            self.nFrac = 1
        else:
            self.nFrac = plan_info["prescription_config"].get('nFrac', 1)

        for beamid in beams.keys(): 
            self.logger.info('[FINAL_DOSE] Calculating dose for beam ID {}.'.format(str(beamid)))
            beam = beams[beamid]

            __beam_model = beam["beam_model"]
            bm_path = __beam_model['beam_model_path']
            __gauss_model_file = __beam_model['gauss_model_file']
            self.bm.readFileFromFolder(bm_path) # load model for each beam
            self.bm.readTriGaussian(__gauss_model_file)

            beam["SAD"] = np.sum(np.abs(beam["source_pos"])*0.5)
            NetWork.sendStatus(None, self.task_id,
                                          "Calculating dose for beam '{}' ( {} / {} )".\
                                          format(beam['beamName'], __beam_count+1, __num_beam),
                                          int(startProg + delta_progress * 0.1))
            if(self.dose_engine=="monte_carlo"):
                beams[beamid], uncertainty, simN = self.mc.process(beam=beams[beamid], dc=self.dc, \
                                                                   bm=self.bm, calROIIndex=self.ext_contour_linear,
                                                                   table=self.table, nIonPerSpot=self.nIonPerSpot,
                                                                   uncertaintyLevel=self.uncertaintyLevel,
                                                                   ifSpots=self.ifSpots,
                                                                   ifWater=self.ifWater, task_id=self.task_id,
                                                                   startProg=startProg + delta_progress * 0.1,
                                                                   deltaProg=0.9 * delta_progress,
                                                                   particle_type=self.particle_type)
                if(self.maxUncertainty<uncertainty):
                    self.maxUncertainty = uncertainty
                self.totalSimN += simN
            elif(self.dose_engine == "gaussian"):
                beams[beamid] = self.dc.compute_single_beam(self.nFrac, beam=beam, bm=self.bm,
                                                            ext_contour_linear=self.ext_contour_linear,
                                                            ext_contour_linear_opt=self.ext_contour_linear_opt,
                                                            particle_type=self.particle_type,
                                                            cal_mode=self.pbs_resolution,
                                                            startProgress=startProg + delta_progress * 0.1,
                                                            deltaProgress=0.9 * delta_progress, calType=calType,
                                                            cal_fluence_map=self.cal_fluence_map,
                                                            nuclear_correction=self.nuclear_correction)
            else:
                raise RuntimeError("Unsupported method for physical dose")
            startProg += delta_progress
        # if self.serviceMode:
        NetWork.sendStatus(None, task_id=self.task_id, msg=json.dumps({"totalIons": self.totalSimN}), process=-1)

    def cal_RBE(self, plan_info, startProgress = 0, calType='Opt'):
        if calType in ['Opt', 'RBE']:
            if self.rbecal.RBEmethod != 5:
                self.rbecal.selectBeamForCal(-1)
                self.rbecal.updateCalIdx()
                if(self.rbecal.RBEmethod==3):
                    estimatedTime = 0.5 * self.rbecal.dosemap.nnz/2**20
                    estimatedTime = estimatedTime*self.rbecal.nRepeat/1000
                    NetWork.sendStatus(None, self.task_id, "Estimated time {:.2f} minutes.".format(estimatedTime), startProgress)

        self.rbecal.calRBEWithMode(2, calType=calType)
        # if plan_info["dose_cal_config"]["rbe_method"] != "Constant1.1":
        #     self.rbecal.calRBEWithMode(2)
        # else:
        #     self.rbecal.rbemap = np.ones(self.dc.doseGrid.dims) * 1.1

        beams = plan_info["beams"]
        for beamid in beams.keys():
            beam = beams[beamid]
            # self.rbecal.selectBeamForCal(beamid)
            # self.rbecal.updateCalIdx()
            # self.rbecal.calRBEWithMode(2)
            beam["rbe_map"] = self.rbecal.rbemap

    def updateBeaminfoFromFile(self, plan_info):
        beams = plan_info["beams"]
        for beam_id in beams.keys():
            __beam_model = beams[beam_id]["beam_model"]
            bm_path = __beam_model['beam_model_path']
            __gauss_model_file = __beam_model['gauss_model_file']
            self.bm.readFileFromFolder(bm_path)
            self.bm.readTriGaussian(__gauss_model_file)
            SAD = np.sum(np.abs(beams[beam_id]["source_pos"])*0.5)
            energy_spot_info = load_json(beams[beam_id]["jsonPath"])

            # energy_spot_info = load_json("/data/env/dev-env/tps//data/app/tps/work//case_186/plan_233/radiationSet_236/algorithm/energy_spot_info_beam_368.json")
            # 测试单点 subspot，可以修改单点坐标
            # energy_spot_info = energy_spot_info[1]
            # energy_spot_info["spot_position_and_weight"] = [{'x': '1', 'z': '0', 'value': '6889.3726', 'np': '20842418.8181'}]
            # energy_spot_info = [energy_spot_info]
            calEneList = self.bm.triGaussian['meaEneList']
            if (self.bm.triGaussian["variableEnergyMode"] == "Continuous"):
                calEneList = []
                for __energy_layer in energy_spot_info:
                    __energy = float(__energy_layer['energy'])
                    calEneList.append(__energy)
                calEneList = np.sort(np.unique(calEneList))[::-1]
            scan_points = dict()
            for __energy_layer in energy_spot_info:
                __energy =    float(__energy_layer['energy']) #399.92#361.34#330.09 #261.03 #190.19 #160.86 #120.23 #
                energy_iidd = int(__energy_layer["energy_id"])
                energy_idx = np.argmin(abs(np.array(calEneList) - __energy))
                if energy_iidd not in scan_points.keys():
                    scan_points[energy_iidd] = dict()
                    scan_points[energy_iidd]['theta'] = np.array([])
                    scan_points[energy_iidd]['phi'] = np.array([])
                    scan_points[energy_iidd]['x'] = np.array([])
                    scan_points[energy_iidd]['z'] = np.array([])
                    scan_points[energy_iidd]['energy'] = np.array([])
                    scan_points[energy_iidd]['scaleFactor'] = np.array([])
                    scan_points[energy_iidd]['range'] = np.array([])
                    scan_points[energy_iidd]['spotId'] = np.array([])
                    scan_points[energy_iidd]['weight_vector'] = np.array([])
                    scan_points[energy_iidd]['spot_spacing_x'] = np.array([])
                    scan_points[energy_iidd]['spot_spacing_z'] = np.array([])
                    scan_points[energy_iidd]['energy_idx'] = np.array([])
                else:
                    self.logger.info("Layers of the same energy {} MeV.".format(__energy))
                
                spotSpacing = __energy_layer.get("spot_spacing","[0,0]").strip("\"[]").split(",")
                x_list = list()
                z_list = list()
                weight_list = list()
                # spotId_list = list()
                for spot in __energy_layer['spot_position_and_weight']:
                    x_list.append(float(spot['x']))
                    z_list.append(-float(spot['z']))
                    weight_list.append(float(spot['value']))
                    # spotId_list.append(int(spot['id']))

                theta = np.arctan((np.sqrt(np.array(x_list) ** 2 + np.array(z_list) ** 2)) / SAD)
                assert 0.5 * np.pi > theta.all() >= 0
                phi = np.arctan2(np.array(z_list), np.array(x_list))
                assert np.pi >= phi.all() >= -1 * np.pi


                scan_points[energy_iidd]['theta'] = np.hstack((scan_points[energy_iidd]['theta'], theta))
                scan_points[energy_iidd]['phi'] = np.hstack((scan_points[energy_iidd]['phi'], phi))
                scan_points[energy_iidd]['x'] = np.hstack((scan_points[energy_iidd]['x'], np.array(x_list)))
                scan_points[energy_iidd]['z'] = np.hstack((scan_points[energy_iidd]['z'], np.array(z_list)))
                scan_points[energy_iidd]['energy'] = np.hstack((scan_points[energy_iidd]['energy'], np.ones(np.size(theta), ) * __energy))
                scan_points[energy_iidd]['scaleFactor'] = np.hstack((scan_points[energy_iidd]['scaleFactor'], self.bm.getnPerMUInterp(np.ones(np.size(theta), ) * __energy)))
                scan_points[energy_iidd]['range'] = np.hstack((scan_points[energy_iidd]['range'], np.ones(np.size(theta), ) * \
                                                         self.bm.getPeakPos(__energy, mode="R80")))
                scan_points[energy_iidd]['spotId'] = np.hstack((scan_points[energy_iidd]['spotId'], np.arange(np.size(theta)).astype('int16')))
                scan_points[energy_iidd]['weight_vector'] = np.hstack((scan_points[energy_iidd]['weight_vector'], np.array(weight_list)))
                scan_points[energy_iidd]['spot_spacing_x'] = np.hstack((scan_points[energy_iidd]['spot_spacing_x'], np.ones(np.size(theta), ) *float(spotSpacing[0])))
                scan_points[energy_iidd]['spot_spacing_z'] = np.hstack((scan_points[energy_iidd]['spot_spacing_z'], np.ones(np.size(theta), ) *float(spotSpacing[1])))
                scan_points[energy_iidd]['energy_idx'] = np.hstack((scan_points[energy_iidd]['energy_idx'], energy_idx))
                # Combine spots of the same energy and the same (x,z) coordinates.
                # __coords = np.array(np.vstack((scan_points[energy_idx]['x'], scan_points[energy_idx]['z']))).transpose()
                # _, unique_indices = np.unique(__coords, axis=0, return_index=True)
                # if np.size(unique_indices) == np.size(x_list):
                #     continue
                # else:
                #     rep_spot_indices = set(np.arange(len(x_list))) - set(unique_indices)
                #     rep_spot_indices_sorted = list()
                #     for rep_spot_index in rep_spot_indices:
                #         __ind = set(np.where(__coords[:, 0] == __coords[rep_spot_index, 0])[0]) & \
                #                 set(np.where(__coords[:, 1] == __coords[rep_spot_index, 1])[0])
                #         rep_spot_indices_sorted.append(list(__ind))
                #     for spot_index_sorted in rep_spot_indices_sorted:
                #         scan_points[energy_idx]['x'][spot_index_sorted[1:]] = np.nan
                #         scan_points[energy_idx]['z'][spot_index_sorted[1:]] = np.nan
                #         scan_points[energy_idx]['theta'][spot_index_sorted[1:]] = np.nan
                #         scan_points[energy_idx]['phi'][spot_index_sorted[1:]] = np.nan
                #         scan_points[energy_idx]['weight_vector'][spot_index_sorted[0]] = np.sum(np.array(weight_list)[spot_index_sorted])
                #         scan_points[energy_idx]['weight_vector'][spot_index_sorted[1:]] = np.nan
                #     remain_idx = np.logical_not(np.isnan(scan_points[energy_idx]['x']))
                #     scan_points[energy_idx]['x'] = scan_points[energy_idx]['x'][remain_idx]
                #     scan_points[energy_idx]['z'] = scan_points[energy_idx]['z'][remain_idx]
                #     scan_points[energy_idx]['theta'] = scan_points[energy_idx]['theta'][remain_idx]
                #     scan_points[energy_idx]['phi'] = scan_points[energy_idx]['phi'][remain_idx]
                #     scan_points[energy_idx]['weight_vector'] = scan_points[energy_idx]['weight_vector'][remain_idx]
                #     scan_points[energy_idx]['energy'] = scan_points[energy_idx]['energy'][remain_idx]
                #     scan_points[energy_idx]['scaleFactor'] = scan_points[energy_idx]['scaleFactor'][remain_idx]
                #     scan_points[energy_idx]['range'] = scan_points[energy_idx]['range'][remain_idx]
                #     scan_points[energy_idx]['spotId'] = scan_points[energy_idx]['spotId'][remain_idx]
                #     scan_points[energy_idx]['spot_spacing_x'] = scan_points[energy_idx]['spot_spacing_x'][remain_idx]
                #     scan_points[energy_idx]['spot_spacing_z'] = scan_points[energy_idx]['spot_spacing_z'][remain_idx]

            # order scan_points from max energy to min energy, because compute_single_beam gets theta, phi from scan_points. 2023.11.29.
            energy_list = np.sort(list(scan_points.keys()))[::-1]
            __scan_points = dict()
            for energy_idx in energy_list:
                __scan_points[energy_idx] = scan_points[energy_idx]
            beams[beam_id]['scan_points'] = __scan_points

            attribute_lists = ["x", "z", "theta", "phi", "energy", \
                           "range", "scaleFactor", "spot_spacing_x", "spot_spacing_z", "weight_vector", "spotId", "energy_idx"]
            beam = beams[beam_id]
            for attribute in attribute_lists:
                beam[attribute] = []
                energy_list = np.sort(list(scan_points.keys()))[::-1]  # make sure descending order
                for energy_idx in energy_list:
                    if (attribute in scan_points[energy_idx].keys()):
                        beam[attribute].extend(scan_points[energy_idx][attribute])
                    else:
                        beam[attribute].extend([-1] * len(scan_points[energy_idx]["x"]))
                beam[attribute] = np.array(beam[attribute])
            
            # 统计每个能量层的分界点 - 用于优化删点
            spot_interval_energy_layer = [0]
            num_spots = 0  # 每个能量层起始的index
            for energy_idx in energy_list:
                num_spots+=len(scan_points[energy_idx]['weight_vector'])
                spot_interval_energy_layer.append(num_spots)
            beam['spot_interval_energy_layer'] = spot_interval_energy_layer
