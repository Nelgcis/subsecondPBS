#include "common/common.cuh"
#include <cuda_runtime.h>
#include "cooperative_groups.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"
#include "thrust/execution_policy.h"
#include "thrust/sort.h"
#include "thrust/functional.h"
#include <thrust/for_each.h>

#include "omp.h"
namespace cg = cooperative_groups;


// #define ptx_optimize    //whether use ptx optimization or not
// #define tex_optimize    //whether use tex optimization or not
// #define coalescing_optimize //whether use coalesce optimization or not


#ifdef ptx_optimize
__device__ float calGaussianR(float r2, float sigma2) {
    float result;
    asm volatile (
        "{\n"
        "   .reg .f32             tmp, denom;\n"
        "   add.f32               denom, %1, %3;\n"                                  // denom = sigma2 + 1e-5
        "   neg.f32               tmp, %2;\n"                                        // tmp = -r2
        "   div.rn.f32            tmp, tmp, denom;\n"                                // tmp = -r2/denom
				"   mul.f32               tmp, tmp, %5;\n"                                   // tmp = -r2/denom * log(2)
        "   ex2.approx.ftz.f32    tmp, tmp;\n"                                       // tmp = exp(tmp)
        "   mul.f32               denom, denom, %4;\n"                               // π*denom
        "   rcp.approx.ftz.f32    denom, denom;\n"                                   // 1/(π*denom)
        "   mul.f32               %0, tmp, denom;\n" 
        "}\n"
        : "=f"(result)
        : "f"(sigma2), "f"(r2),"f"(0.00001f),"f"(3.1415926f),"f"(1.442695f)
    );
    return result;
}
#else
__device__ float
calGaussianR(float r2, float sigma2) {
    return expf(-r2 / (sigma2 + 0.00001f)) / 3.1415926f / (sigma2 + 0.00001f);
}
#endif


__global__ void calFluenceMapAlphaBetaKernel(float* dalpha,
                                             float* dbeta,
                                             float* ddosecube,
                                             float* ddose,
                                             float* weights,
                                             float* rbe2,
                                             float* counter,
                                             vec3i* roiIndex,
                                             vec3f* beamDirect,
                                             vec3f* dsource,
                                             int* drowIndex,
                                             int* colPtr,
                                             int* deneId,
                                             vec3f rayweqSetting,
                                             int nSpots,
                                             Grid doseGrid,
                                             float rs,
                                             float estDimension,
                                             cudaTextureObject_t rayweqData,
                                             cudaTextureObject_t letData,
                                             cudaTextureObject_t lqData)
{
	cg::grid_group g = cg::this_grid();
	int gid          = g.thread_rank();
//    printf("i am in kernel!\n");
	while (gid < nSpots)
	{
		int nVoxels = colPtr[gid+1] - colPtr[gid];
		for (int i = 0; i < nVoxels; i++)
		{
			int   index     = colPtr[gid] + i;
			int   rowIndex  = drowIndex[index];

			int   eneId     = deneId[gid];
			vec3f source    = dsource[gid];
			vec3f direction = beamDirect[gid];
			float weight    = weights[gid];

			vec3f pos = vec3f(roiIndex[rowIndex].x * doseGrid.resolution.x + doseGrid.corner.x,
			                  roiIndex[rowIndex].y * doseGrid.resolution.y + doseGrid.corner.y,
			                  roiIndex[rowIndex].z * doseGrid.resolution.z + doseGrid.corner.z);

			int absId = roiIndex[rowIndex].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[rowIndex].y * doseGrid.dims.z + roiIndex[rowIndex].z;

			float projectedLength = dot(direction, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			float weqDepth = tex2D<float>(rayweqData, idx + 0.5f, float(gid) + 0.5f);

			float let  = 0;
			float let2 = 0;

			for (int j = 0; j < 5; j++)
			{
				float tmplet = tex2D<float>(letData, (weqDepth + rs + (float(j) - 2.0f) * estDimension / 5.0f) * 2.0f, float(eneId));
				let2 = let2 + tmplet * tmplet;
				let  = let + tmplet;
			}

			let = let2 / let;
			float alpha, beta, tmpDose;
			alpha = tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 0.5f, 0.5f);
			beta  = sqrtf(tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 1.5f, 0.5f));
			tmpDose = ddose[index] * weight;
			atomicAdd(dalpha + absId, tmpDose * alpha);
			atomicAdd(dbeta + absId, tmpDose * beta);
			atomicAdd(ddosecube + absId, tmpDose);
			atomicAdd(rbe2 + absId, let*tmpDose);
//			atomicAdd(counter + absId, 1.0f);
//            printf("let:%f alpha:%f beta:%f dose:%f\n", let, alpha, beta, tmpDose);
		}
		gid += g.size();
	}
}

__global__ void calDoseSumSquareKernel(float  *d_dose,   //在 GPU 上并行计算每个beam对体素产生的剂量平方和，用于统计或目标函数评估
							  vec3f  *d_beamDirect,
							  int    num_beam,
							  vec3f  source,
							  vec3f  bmxdir,
							  vec3f  bmydir,
							  vec3i  *roiIndex,
							  int    num_roi,
							  Grid   doseGrid,
							  cudaTextureObject_t rayweqData,
							  cudaTextureObject_t iddData,
							  cudaTextureObject_t profileData,
							  cudaTextureObject_t subspotData,
							  vec3f rayweqSetting,
							  vec3f iddDepth,
							  vec3f profileDepth,
							  float *d_idbeamxy,
							  int   nsubspot,
							  int   nGauss,
							  int   eneIdx,
							  int   beamOffset,
							  float beamParaPos,
							  float longitudalCutoff,
							  float transCutoff,
							  float rtheta,
							  float theta2,
							  float radius_cutoff,
							  float sad) 
{
	cg::grid_group g = cg::this_grid(); //利用CUDA Cooperative Groups将整个grid按体素和beam分别分组
	cg::thread_block b = cg::this_thread_block();

	dim3 grid_index    = b.group_index();
	dim3 block_dim     = b.group_dim();
	dim3 thread_index  = b.thread_index();

	uint64_t voxelIndex = thread_index.x + block_dim.x * grid_index.x; //每个线程首先通过voxelIndex在体素集合roiIndex中迭代
	uint64_t roiThreadNum = g.group_dim().x * b.group_dim().x;
	uint64_t beamThreadNum = g.group_dim().y * b.group_dim().y;

	while (voxelIndex < num_roi) {
		vec3f pos = vec3f(roiIndex[voxelIndex].x * doseGrid.resolution.x + doseGrid.corner.x,
			roiIndex[voxelIndex].y * doseGrid.resolution.y + doseGrid.corner.y,
			roiIndex[voxelIndex].z * doseGrid.resolution.z + doseGrid.corner.z);
		uint64_t beamIndex = thread_index.y + block_dim.y * grid_index.y; //对每个体素，再通过beamIndex遍历所有射束
		while (beamIndex < num_beam) {
			vec3f beamDirect = d_beamDirect[beamIndex];
			float ix = d_idbeamxy[beamIndex * 2];
			float iy = d_idbeamxy[beamIndex * 2 + 1];
			float dose = 0;
			for (int isubspot = 0; isubspot < nsubspot; isubspot++) {
				float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx)); 
				//从三维纹理subspotData中读取每个子射斑的偏移(deltax, deltay)、权重subspotweight及横向展宽参数 (sigmax, sigmay)
				float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));
	
				float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
				if (subspotweight < 0.001f) continue;
				float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
				float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
				float r2 = sigmax * sigmax + sigmay * sigmay;
	
				vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir; 
				//子射斑方向subspotDirection结合射束方向beamDirect与两个正交基bmxdir, bmydir归一化后，用于计算投影深度projectedLength与横向距离平方crossDis2
				subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
				float projectedLength = dot(subspotDirection, pos - source);
				float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
				vec3f target = source + subspotDirection * projectedLength;
				float crossDis2 = dot(pos - target, pos - target);

				//水当量深度与剂量展宽: 通过三维纹理 rayweqData 在索引 (idx, iy+deltay, ix+deltax)处获取等效水深 weqDepth，并计算物理深度 phyDepth
					float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
				float phyDepth = projectedLength - (sad - beamParaPos);
				
	
				if (weqDepth < longitudalCutoff) {
					float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
					float initR2 = r2 + 2.f * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
					float gaussianWeight = 0.f;
	
					//若 weqDepth 小于纵向截止 longitudalCutoff，则在三维纹理 profileData 上按 nGauss 个高斯成分累加 横向高斯权重 gaussianWeight
					for (int j = 0; j < nGauss; j++)
					{
						float w = tex3D<float>(profileData, float(j) + 0.5, profileDepthIdx, float(eneIdx + 0.5));
						float sigma = tex3D<float>(profileData, float(j + nGauss) + 0.5, profileDepthIdx, float(eneIdx + 0.5));
						sigma = sigma * sigma + initR2; // 计算点位置处的展宽
						// = 水带来的展宽+空气里的
						gaussianWeight += calGaussianR(crossDis2, sigma) * w;
					}
	
					if (gaussianWeight > transCutoff)
					{
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose += idd * gaussianWeight * overallWeight * subspotweight;
					}
				}
			}
			//剂量累加与平方和: 若 gaussianWeight > transCutoff，再结合二维纹理 iddData（内含剂量分布）和整体权重 overallWeight 计算本次贡献，并通过：
			atomicAdd(d_dose + voxelIndex, dose * dose);
			beamIndex += beamThreadNum;
		}
		voxelIndex += roiThreadNum;
	}
}


//此函数使用压缩稀疏列Compressed Sparse Column格式，通过数组 cscIndices, colindices, cscIndptr 存储非零元位置和值索引
__global__ void calDoseKernel(float  *dose, //构建稀疏剂量矩阵（dose deposition coefficients），以支持优化器对照射点权重的调整
							  int    *cscIndices, //csc：compressed sparse column
							  int    *colindices,
							  int    *cscIndptr, //每列（射束）的非零计数前缀和
							  int    totalnnz, //nnz：非零元素数量
							  int    *sumNNZ,
							  vec3f  *d_beamDirect,
							  int    num_beam,
							  vec3f  source,
							  vec3f  bmxdir,
							  vec3f  bmydir,
							  vec3i  *roiIndex,
							  int    num_roi,
							  Grid   doseGrid,
							  cudaTextureObject_t rayweqData,
							  cudaTextureObject_t iddData,
							  cudaTextureObject_t profileData,
							  cudaTextureObject_t subspotData,
							  vec3f rayweqSetting,
							  vec3f iddDepth,
							  vec3f profileDepth,
							  float *d_idbeamxy,
							  int   nsubspot,
							  int   nGauss,
							  int   eneIdx,
							  int   beamOffset,
							  float beamParaPos,
							  float longitudalCutoff,
							  float transCutoff,
							  float rtheta,
							  float theta2,
							  float radius_cutoff,
							  float sad) {
	cg::grid_group g = cg::this_grid();

	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(
		num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		//	int beamid       = gtx + beamOffset;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];


		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int nnz = 0;
		bool increaseNNZFlag = true;
		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001f) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			//		printf("source:%f %f %f sad:%f beampos:%f\n", source.x, source.y, source.z, sad, beamParaPos);
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);
			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2.f * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0.f;
				for (int j = 0; j < nGauss; j++)
				{
#ifdef tex_optimize
					float w = tex3D<float>(profileData, 
                                 float(j) + 0.5f,     
                                 float(profileDepthIdx) + 0.5f, 
                                 float(eneIdx) + 0.5f);

          float sigma = tex3D<float>(profileData, 
                                     float(j + nGauss) + 0.5f, 
                                     float(profileDepthIdx) + 0.5f, 
                                     float(eneIdx) + 0.5f);

#else
          float w = tex3D<float>(profileData, j + 0.5, profileDepthIdx, eneIdx + 0.5);
					float sigma = tex3D<float>(profileData, j + nGauss + 0.5, profileDepthIdx, eneIdx + 0.5);
#endif
					
          sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;	
				}
				// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
				if (gaussianWeight > transCutoff)
				{
					if (increaseNNZFlag)
					{
						nnz = atomicAdd(sumNNZ, 1);
						atomicAdd(&(cscIndptr[gtx + 1]), 1);
						increaseNNZFlag = false;
					}
					if (nnz < totalnnz)
					{
						cscIndices[nnz] = gty;
						colindices[nnz] = gtx + beamOffset;
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose[nnz] += idd * gaussianWeight * overallWeight * subspotweight;
					}
				}
			}
		}
		gid += g.size();
	}
}

__global__ void calDoseKernel_(float  *dose,
							  int    *cscIndices,
							  int    *colindices,
							  int    *cscIndptr,
								float  *d_idd_per_depth,//suplementary
							  int    totalnnz,
							  int    *sumNNZ,
							  vec3f  *d_beamDirect,
							  int    num_beam,
							  vec3f  source,
							  vec3f  bmxdir,
							  vec3f  bmydir,
							  vec3i  *roiIndex,
							  int    num_roi,
							  Grid   doseGrid,
							  cudaTextureObject_t rayweqData,
							  cudaTextureObject_t iddData,
							  cudaTextureObject_t profileData,
							  cudaTextureObject_t subspotData,
							  vec3f rayweqSetting,
							  vec3f iddDepth,
							  vec3f profileDepth,
							  float *d_idbeamxy,
							  int   nsubspot,
							  int   nGauss,
							  int   eneIdx,
							  int   beamOffset,
							  float beamParaPos,
							  float longitudalCutoff,
							  float transCutoff,
							  float rtheta,
							  float theta2,
							  float radius_cutoff,
							  float sad) {
	cg::grid_group g = cg::this_grid();

	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(
		num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		//	int beamid       = gtx + beamOffset;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];


		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int nnz = 0;
		bool increaseNNZFlag = true;
		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001f) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			//		printf("source:%f %f %f sad:%f beampos:%f\n", source.x, source.y, source.z, sad, beamParaPos);
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);
			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				for (int j = 0; j < nGauss; j++)
				{
					float w = tex3D<float>(profileData, float(j + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					float sigma = tex3D<float>(profileData, float(j + nGauss + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
				}
				// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
				if (gaussianWeight > transCutoff)
				{
					if (increaseNNZFlag)
					{
						nnz = atomicAdd(sumNNZ, 1);
						atomicAdd(&(cscIndptr[gtx + 1]), 1);
						increaseNNZFlag = false;
					}
					if (nnz < totalnnz)
					{
						cscIndices[nnz] = gty;
						colindices[nnz] = gtx + beamOffset;
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose[nnz] += idd * gaussianWeight * overallWeight * subspotweight;

            atomicAdd(d_idd_per_depth + int(iddDepthIdx), idd * gaussianWeight * overallWeight * subspotweight);  //suplementary

					}
				}
			}
		}
		gid += g.size();
	}
}

__global__ void calFinalDoseKernel(float *dose,
								   vec3f *d_beamDirect,
								   int   num_beam,
								   vec3f source,
								   vec3f bmxdir,
								   vec3f bmydir,
								   vec3i *roiIndex,
								   int   num_roi,
								   Grid  doseGrid,
								   cudaTextureObject_t rayweqData,
								   cudaTextureObject_t iddData,
								   cudaTextureObject_t profileData,
								   cudaTextureObject_t subspotData,
								   vec3f rayweqSetting,
								   vec3f iddDepth,
								   vec3f profileDepth,
								   float *d_idbeamxy,
								   int   *d_numParticles,
								   int   nsubspot,
								   int   nGauss,
								   int   eneIdx,
								   int   beamOffset,
								   float beamParaPos,
								   float longitudalCutoff,
								   float transCutoff,
								   float rtheta,
								   float theta2,
								   float r22,
								   float sad) {

  cg::grid_group g   = cg::this_grid();
  cg::thread_block b = cg::this_thread_block();
  dim3 grid_index    = b.group_index();
  dim3 block_dim     = b.group_dim();
  dim3 thread_index  = b.thread_index();

  unsigned int gtx = thread_index.x + block_dim.x * grid_index.x;
  while (gtx < num_beam) {
//	int beamid = gtx + beamOffset;
	vec3f beamDirect = d_beamDirect[gtx];
	unsigned int gty = thread_index.y + block_dim.y * grid_index.y;

	float ix = d_idbeamxy[gtx * 2];
	float iy = d_idbeamxy[gtx * 2 + 1];

	while (gty < num_roi) {

	  vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
						roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
						roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

	  int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z + roiIndex[gty].z;

//	  int nnz   = 0;
//	  bool increaseNNZFlag = true;
	  for (int isubspot = 0; isubspot < nsubspot; isubspot++) {
		float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
		float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

		float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
		if (subspotweight < 0.001) continue;
		float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
		float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
		float r2     = sigmax * sigmax + sigmay * sigmay;
		vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
		subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
		float projectedLength = dot(subspotDirection, pos - source);
		float idx       = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
		vec3f target    = source + subspotDirection * projectedLength;
		float crossDis2 = dot(pos - target, pos - target);

		float weqDepth  = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
		float phyDepth  = projectedLength - (sad - beamParaPos);

	  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d longitudal:%f\n", weqDepth, eneIdx, longitudalCutoff);
	  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d ", weqDepth, eneIdx);

		if (weqDepth < longitudalCutoff) {
		  float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
		  float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
		  float gaussianWeight = 0;
		  float sigma;
		  for (int j = 0; j < nGauss; j++) {
			float w = tex3D<float>(profileData, float(j + 0.5), profileDepthIdx, float(eneIdx + 0.5));
			sigma = tex3D<float>(profileData, float(j + nGauss + 0.5), profileDepthIdx, float(eneIdx + 0.5));
			sigma = sigma * sigma + initR2; // 计算点位置处的展宽
			// = 水带来的展宽+空气里的
			gaussianWeight += calGaussianR(crossDis2, sigma) * w;
		  }
		  // if (gtx == 0 && gty == 127422) printf("profileDepthId:%f sigma:%f initR2:%f gaussianweight:%e\n", profileDepthIdx, sigma, initR2,
		  //        gaussianWeight);
		  // if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  if (gaussianWeight > transCutoff) {
			float iddDepthIdx   = (weqDepth - iddDepth.x) / iddDepth.y;
			float idd           = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
			float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f idddepthidx:%f eneIdx:%d idd:%e\n", weqDepth, iddDepthIdx, eneIdx, idd);
		  	// if (gtx == 0 && gty == 127422) printf("%e %e %e %e %d gty:%d\n", idd, gaussianWeight, overallWeight, subspotweight, d_numParticles[gtx], gty);
			atomicAdd(dose + absId, idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx]);
		  }
		}
	  }
	  gty += g.group_dim().y * b.group_dim().y;
	}
	gtx += g.group_dim().x * b.group_dim().x;
  }
}

__global__ void cuCalFinalDoseAndRBEKernel(
																					 float* dose,
																					 float* rbe2,
																					 float* counter,
																					 float* dalpha,
																					 float* dbeta,
#ifdef coalescing_optimize
                                           float* d_beamDirect_x,
																					 float* d_beamDirect_y,
																					 float* d_beamDirect_z,
#else
                                           vec3f* d_beamDirect,
#endif
                                           int num_beam,
                                           vec3f source,
                                           vec3f bmxdir,
                                           vec3f bmydir,
                                           vec3i* roiIndex,
                                           int num_roi,
                                           Grid doseGrid,
                                           cudaTextureObject_t rayweqData,
                                           cudaTextureObject_t iddData,
                                           cudaTextureObject_t profileData,
                                           cudaTextureObject_t subspotData,
                                           cudaTextureObject_t letData,
                                           cudaTextureObject_t lqData,
                                           vec3f rayweqSetting,
                                           vec3f iddDepth,
                                           vec3f profileDepth,
                                           float* d_idbeamxy,
                                           int* d_numParticles,
                                           int nsubspot,
                                           int nGauss,
                                           int eneIdx,
                                           int fieldId,
                                           float beamParaPos,
                                           float longitudalCutoff,
                                           float transCutoff,
                                           float rtheta,
                                           float theta2,
                                           float estDimension,
                                           float sad,
                                           float rs)
{
	cg::grid_group g = cg::this_grid();


	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;

#ifdef coalescing_optimize
		vec3f beamDirect = vec3f(d_beamDirect_x[gtx], d_beamDirect_y[gtx], d_beamDirect_z[gtx]);
		float ix = reinterpret_cast<float2*>(&d_idbeamxy[gtx * 2])[0].x;
		float iy = reinterpret_cast<float2*>(&d_idbeamxy[gtx * 2])[0].y;
#else
		vec3f beamDirect = d_beamDirect[gtx];
		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];
#endif

		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z +
			roiIndex[gty].z;

		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);

			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);

			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				float let = 0;
				float let2 = 0;
				float sigma;
				for (int j = 0; j < nGauss; j++)
				{
#ifdef tex_optimize
					float w = tex3D<float>(profileData, 
                                 float(j) + 0.5f,     
                                 float(profileDepthIdx) + 0.5f, 
                                 float(eneIdx) + 0.5f);

          sigma = tex3D<float>(profileData, 
                                     float(j + nGauss) + 0.5f, 
                                     float(profileDepthIdx) + 0.5f, 
                                     float(eneIdx) + 0.5f);

#else
          float w = tex3D<float>(profileData, j + 0.5, profileDepthIdx, eneIdx + 0.5);
					sigma = tex3D<float>(profileData, j + nGauss + 0.5, profileDepthIdx, eneIdx + 0.5);
#endif
					sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
					float tmplet = tex2D<float>(letData, (weqDepth + rs + (j - 2.0f) * estDimension / 5.0f) * 2.0f,
					                            eneIdx);
					let2 = let2 + tmplet * tmplet;
					let = let + tmplet;
				}
				let = let2 / let;

				float alpha, beta;
				alpha = tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 0.5f, 0.5f);
				beta = sqrtf(tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 1.5f, 0.5f));

				if (gaussianWeight > transCutoff)
				{
					float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
					float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
					float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));

					float tmpDose = idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx];
					atomicAdd(dose + absId + fieldId * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z, tmpDose);
					atomicAdd(rbe2 + absId, let*tmpDose);
//					atomicAdd(counter + absId, 1.0f);
					atomicAdd(dalpha + absId, tmpDose * alpha);
					atomicAdd(dbeta + absId, tmpDose * beta);
				}
			}
		}
		gid += g.size();
	}
}

__global__ void LQRBEKernel(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap,
                            float* rbe2, float* counter, float* phyDose, int nField)
{// alpha as input is alpha for each voxel with dose as weight
	// alpha as output is RBE
	// int id = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();
	int id = g.thread_rank();
	// printf("i am in kernel!\n");
	while(id < num)
	{
		int absId  = roiIndex[id].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[id].y * doseGrid.dims.z + roiIndex[id].z;
		float dose = 0.0f;
#pragma unroll
		for(int i = 0; i < nField; i++)
			dose += phyDose[i * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z + absId];
		// float dose = phyDose[absId];
		if(dose==0 || alphamap[absId]==0 || betamap[absId]==0)
		{
			rbemap[absId]=1;
			// printf("dose:%e alpha:%f beta:%f\n", dose, alphamap[absId], betamap[absId]);
		}
		else{

			float alpha = alphamap[absId]/dose;
			float beta  = betamap[absId]/dose;
			beta       *= beta; // average sqrt(beta)

			beta = (-alpha+sqrtf(alpha*alpha-4*-2.302585f*beta))/beta*0.5f;// Note, beta reused!

			rbemap[absId] = 3.988565f/beta;
		}
		if (dose > 0.0f)
			rbe2[absId] /= dose;
		id += g.size();
	}
}

__global__ void cuCalFinalDoseAndLEMKernel(float* dose,
                                           float* rbe2,
                                           float* counter,
                                           float* dalpha,
                                           float* dbeta,
                                           vec3f* d_beamDirect,
                                           int num_beam,
                                           vec3f source,
                                           vec3f bmxdir,
                                           vec3f bmydir,
                                           vec3i* roiIndex,
                                           int num_roi,
                                           Grid doseGrid,
                                           cudaTextureObject_t rayweqData,
                                           cudaTextureObject_t iddData,
                                           cudaTextureObject_t profileData,
                                           cudaTextureObject_t subspotData,
                                           cudaTextureObject_t letData,
                                           cudaTextureObject_t lqData,
                                           vec3f rayweqSetting,
                                           vec3f iddDepth,
                                           vec3f profileDepth,
                                           float* d_idbeamxy,
                                           int* d_numParticles,
                                           int nsubspot,
                                           int nGauss,
                                           int eneIdx,
                                           int fieldId,
                                           float beamParaPos,
                                           float longitudalCutoff,
                                           float transCutoff,
                                           float rtheta,
                                           float theta2,
                                           float estDimension,
                                           float sad,
                                           float rs)
{
	cg::grid_group g = cg::this_grid();


	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];


		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z +
			roiIndex[gty].z;

		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);

			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);

			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				float let = 0;
				float let2 = 0;
				float sigma;
				for (int j = 0; j < nGauss; j++)
				{
					float w = tex3D<float>(profileData, float(j + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					sigma = tex3D<float>(profileData, float(j + nGauss + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
					float tmplet = tex2D<float>(letData, (weqDepth + rs + float(j - 2.0f) * estDimension / 5.0f) * 2.0f,
					                            eneIdx);
					let2 = let2 + tmplet * tmplet;
					let = let + tmplet;
				}
				let = let2 / let;

				float alpha, beta;
				alpha = tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 0.5f, 0.5f);
				beta = sqrtf(tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 1.5f, 0.5f));

				if (gaussianWeight > transCutoff)
				{
					float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
					float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
					float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));

					float tmpDose = idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx];
					atomicAdd(dose + absId + fieldId * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z, tmpDose);
					atomicAdd(rbe2 + absId, let*tmpDose);
//					atomicAdd(counter + absId, 1.0f);
					atomicAdd(dalpha + absId, tmpDose * alpha);
					atomicAdd(dbeta + absId, tmpDose * beta);
				}
			}
		}
		gid += g.size();
	}
}

__global__ void cuCalFinalDoseAndMKMKernel(float* dose,
                                           float* Z1Dmix,
                                           vec3f* d_beamDirect,
                                           int num_beam,
                                           vec3f source,
                                           vec3f bmxdir,
                                           vec3f bmydir,
                                           vec3i* roiIndex,
                                           int num_roi,
                                           Grid doseGrid,
                                           cudaTextureObject_t rayweqData,
                                           cudaTextureObject_t iddData,
                                           cudaTextureObject_t profileData,
                                           cudaTextureObject_t subspotData,
                                           cudaTextureObject_t Z1DijData,
                                           vec3f rayweqSetting,
                                           vec3f iddDepth,
                                           vec3f profileDepth,
                                           float* d_idbeamxy,
                                           int*   d_numParticles,
                                           int    nsubspot,
                                           int    nGauss,
                                           int    eneIdx,
                                           int    fieldId,
                                           float  beamParaPos,
                                           float  longitudalCutoff,
                                           float  transCutoff,
                                           float  rtheta,
                                           float  theta2,
                                           float  estDimension,
                                           float  sad,
                                           float  rs,
                                           float  energy)
{
	cg::grid_group g = cg::this_grid();

	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];

		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z +
			roiIndex[gty].z;

		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);

			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);

			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				float sigma;
				for (int j = 0; j < nGauss; j++)
				{

#ifdef tex_optimize
					float w = tex3D<float>(profileData, 
                                 float(j) + 0.5f,     
                                 float(profileDepthIdx) + 0.5f, 
                                 float(eneIdx) + 0.5f);

          sigma = tex3D<float>(profileData, 
                                     float(j + nGauss) + 0.5f, 
                                     float(profileDepthIdx) + 0.5f, 
                                     float(eneIdx) + 0.5f);

#else
          float w = tex3D<float>(profileData, j + 0.5, profileDepthIdx, eneIdx + 0.5);
					sigma = tex3D<float>(profileData, j + nGauss + 0.5, profileDepthIdx, eneIdx + 0.5);
#endif
					sigma   = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
				}

				if (gaussianWeight > transCutoff)
				{
					float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
					float iddDepthIdForZij = weqDepth + rs;
					float eneIdForZij = (energy - 80.0f)/2.5f;
					float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
					float Zij = tex2D<float>(Z1DijData, iddDepthIdForZij  + 0.5f, float(eneIdForZij + 0.5));
					// if (energy > 151 && energy < 152 && weqDepth > 40 && weqDepth < 60)
					// 	printf("ene id: %f energy:%f depth id:%f weq:%f zij:%f\n", eneIdForZij, energy, iddDepthIdForZij, weqDepth, Zij);
					float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
					float tmpDose  = idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx];
					// float tmpZ1Dij = Zij * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx];
					atomicAdd(dose + absId + fieldId * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z, tmpDose);
					atomicAdd(Z1Dmix + absId, tmpDose * Zij);
				}
			}
		}
		gid += g.size();
	}
}

__global__ void lemRBEMapKernel(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap,
							float* rbe2, float* counter, float* phyDose, int nField)
{
	cg::grid_group g = cg::this_grid();
	int id           = g.thread_rank();
	while (id < num)
	{
		id += g.size();
		int  absId = roiIndex[id].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[id].y * doseGrid.dims.z + roiIndex[id].z;
		float dose = 0.0f;
#pragma unroll
		for(int i = 0; i < nField; i++)
			dose += phyDose[i * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z + absId];
		if(dose==0 || alphamap[absId]==0 || betamap[absId]==0)
		{
			rbemap[absId]=1;
			// printf("dose:%e alpha:%f beta:%f\n", dose, alphamap[absId], betamap[absId]);
		}
		else{

			float alpha = alphamap[absId]/dose;
			float beta  = betamap[absId]/dose;
			beta       *= beta; // average sqrt(beta)

			beta = (-alpha+sqrtf(alpha*alpha-4*-2.302585f*beta))/beta*0.5f;// Note, beta reused!

			rbemap[absId] = 3.988565f/beta;
		}
		if (dose > 0.0f)
			rbe2[absId] /= dose;
	}
}

__global__ void mkmRBEMapKernel(int num, Grid doseGrid, vec3i* roiIndex, float* rbemap1, float* rbemap2, float* Z1Dmix,
                                float* phyDose,
                                vec2f alpha0AndBeta0, int nField)
{
	cg::grid_group g = cg::this_grid();
	int id           = g.thread_rank();
	while (id < num)
	{
		float alphar  = 0.3312;
		float betar   = 0.0593;
		int   absId   = roiIndex[id].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[id].y * doseGrid.dims.z + roiIndex[id].z;
		float dose    = 0.0f;
#pragma unroll
		for(int i = 0; i < nField; i++)
			dose += phyDose[i * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z + absId];

		// float dose    = phyDose[absId];
		// if (dose > 0)
		// {
		float Z1Dmix_ = Z1Dmix[absId] / dose;
		if (isnan(Z1Dmix_))
			Z1Dmix_ = 0.0f;


		Z1Dmix_ = 0.172f + alpha0AndBeta0.y * Z1Dmix_;

	  [[maybe_unused]]    float lnSr = -2.302585f;
	  [[maybe_unused]]		float tmp1 = alphar / 2.0f / betar;
		float tmp2 = alpha0AndBeta0.x / 2.0f / alpha0AndBeta0.y;
	  [[maybe_unused]]		float tmp3 = Z1Dmix_ / 2.0f / alpha0AndBeta0.y;

		// rbemap1[absId] = 2.5077f / (-tmp3 + sqrtf(-lnSr/2.0f/alpha0AndBeta0.y + tmp3 * tmp3));

		// rbemap1[absId] = -tmp2 + sqrtf(
		// 	tmp2 * tmp2 + (Z1Dmix_ * dose + alpha0AndBeta0.y * dose * dose) / alpha0AndBeta0.y);
		rbemap1[absId] = -tmp2 + sqrtf(
			tmp2 * tmp2 + (Z1Dmix_ * dose + alpha0AndBeta0.y * dose * dose) / alpha0AndBeta0.y);
		if (dose > 1e-3f)
			rbemap1[absId] /= dose;
		else
			rbemap1[absId] = 1.f;
		// rbemap1[absId] = 1.3298f / (-tmp3 + sqrtf(-lnSr/2.0f/alpha0AndBeta0.y + tmp3 * tmp3));

		// rbemap1[absId] = (-tmp1 + sqrtf(Sr / betar + tmp1 * tmp1)) / (-tmp3 + sqrtf(
		// 	Sr / 2.0f / alpha0AndBeta0.y + tmp3 * tmp3));

	  [[maybe_unused]]		float alpha = 0.172f + Z1Dmix_*0.0615;
	  [[maybe_unused]]		float beta  = 0.0615f;

		// if (isnan(rbemap1[absId]))
		// 	printf("tmp1:%f Sr:%f tmp3:%f %f\n", tmp1, Sr, tmp3, (-tmp3 + sqrtf(
		// 	Sr / 2.0f / alpha0AndBeta0.y + tmp3 * tmp3)));
		// rbemap2[absId] = (-tmp2 + sqrtf(
		// 	tmp2 * tmp2 + (0.172f * dose + alpha0AndBeta0.y * Z1Dmix_ * dose + alpha0AndBeta0.y * dose * dose) /
		// 	alpha0AndBeta0.y)) / dose;
		// rbemap2[absId] = 3.988565f/(-alpha+sqrt(alpha*alpha-4*-2.3026*beta))/beta*0.5;
		rbemap2[absId] = Z1Dmix_;
		// } else
		// {
		// 	rbemap1[absId] = 1;
		// 	rbemap2[absId] = 1;
		// }

		// if (isnan(rbemap1[absId]))
		// 	printf("tmp1: %f tmp2:%f tmp3:%f Sr:%f\n", );

		id          += g.size();
	}
}

__global__ void rotate3DArrayKernel(Grid new_g, float *d_new_array, cudaTextureObject_t oldArrayObj, vec4f *d_backRot) {
    cg::grid_group g = cg::this_grid();
    int id = g.thread_rank();
    int size = new_g.dims.x * new_g.dims.y * new_g.dims.z;
    while (id < size) {
        int z = id % new_g.dims.z;
        int y = id / new_g.dims.z % new_g.dims.y;
        int x = id / (new_g.dims.z * new_g.dims.y);
        vec3f center = new_g.corner + vec3f(x, y, z) * new_g.resolution;
        vec3f ori_pos = vec3f(owl::common::dot(d_backRot[0], vec4f(center.x, center.y, center.z, 1.0f)),
                              owl::common::dot(d_backRot[1], vec4f(center.x, center.y, center.z, 1.0f)),
                              owl::common::dot(d_backRot[2], vec4f(center.x, center.y, center.z, 1.0f)));
        d_new_array[id] = tex3D<float>(oldArrayObj, ori_pos.z, ori_pos.y, ori_pos.x);
        id += g.size();
    }
}

__global__ void setROIKernel(vec2i *d_roiFlag, vec3i *d_roiInd, vec3f *d_center, Grid grid, int size) {
    cg::grid_group g = cg::this_grid();
    int id = g.thread_rank();
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (id < size) {
        int absId = d_roiInd[id].x * grid.dims.y * grid.dims.z + d_roiInd[id].y * grid.dims.z + d_roiInd[id].z;
        d_roiFlag[absId].x = 1;
        d_roiFlag[absId].y = id;
        d_center[id] = grid.corner + vec3f(d_roiInd[id]) * grid.resolution + 0.5f * grid.resolution;
        id += g.size();
    }
}

void calFluenceMapAlphaBeta(float* dalpha,
							float* dbeta,
							float* ddosecube,
							float* ddose,
							float* weights,
							float* rbe2,
							float* counter,
							vec3i* roiIndex,
							vec3f* beamDirect,
							vec3f* dsource,
							int* drowIndex,
							int* colPtr,
							int* deneId,
							vec3f rayweqSetting,
							int nSpots,
							Grid doseGrid,
							float rs,
							float estDimension,
							cudaTextureObject_t rayweqData,
							cudaTextureObject_t letData,
							cudaTextureObject_t lqData,
							int gpuid)
{
	void* args[] = {
		(void*)&dalpha, (void*)&dbeta, (void*)&ddosecube, (void*)&ddose, (void*)&weights, (void *)&rbe2, (void *)&counter, (void*)&roiIndex,
		(void*)&beamDirect, (void*)&dsource, (void*)&drowIndex, (void*)&colPtr, (void*)&deneId, (void*)&rayweqSetting,
		(void*)&nSpots, (void*)&doseGrid, (void*)&rs, (void*)&estDimension, (void*)&rayweqData, (void*)&letData, (void*)&lqData
	};
	int batchSize = queryBatchSize(calFluenceMapAlphaBetaKernel, gpuid);
	launchCudaKernel(calFluenceMapAlphaBetaKernel, batchSize, args);
}


void calFinalDose(float *dose,
				  vec3f *d_beamDirect,
				  int   num_beam,
				  vec3f source,
				  vec3f bmxdir,
				  vec3f bmydir,
				  vec3i *roiIndex,
				  int   num_roi,
				  Grid  doseGrid,
				  cudaTextureObject_t rayweqData,
				  cudaTextureObject_t iddData,
				  cudaTextureObject_t profileData,
				  cudaTextureObject_t subspotData,
				  vec3f rayweqSetting,
				  vec3f iddDepth,
				  vec3f profileDepth,
				  float *d_idbeamxy,
				  int   *d_numParticles,
				  int   nsubspot,
				  int   nGauss,
				  int   eneIdx,
				  int   beamOffset,
				  float beamParaPos,
				  float longitudalCutoff,
				  float transCutoff,
				  float rtheta,
				  float theta2,
				  float r2,
				  float sad,
				  int   gpuid) {
  void *args[] = {(void *)&dose, (void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir,
				  (void *)&bmydir, (void *)&roiIndex, (void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData,
				  (void *)&iddData, (void *)&profileData, (void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
				  (void *)&profileDepth, (void *)&d_idbeamxy, (void *)&d_numParticles, (void *)&nsubspot,
				  (void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
				  (void *)&longitudalCutoff, (void *)&transCutoff,
				  (void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};
  int batchSize = queryBatchSize(calFinalDoseKernel, gpuid);
  launchCudaKernel2D(calFinalDoseKernel, batchSize, args);
}

void calFinalDoseAndRBE(float* dose,
												float* rbe2,
												float* counter,
												float* dalpha,
												float* dbeta,
                        vec3f* d_beamDirect,
                        int    num_beam,
                        vec3f  source,
                        vec3f  bmxdir,
                        vec3f  bmydir,
                        vec3i* roiIndex,
                        int    num_roi,
                        Grid   doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t letData,
                        cudaTextureObject_t lqData,
                        vec3f  rayweqSetting,
                        vec3f  iddDepth,
                        vec3f  profileDepth,
                        float* d_idbeamxy,
                        int*   d_numParticles,
                        int    nsubspot,
                        int    nGauss,
                        int    eneIdx,
                        int    fieldId,
                        float  beamParaPos,
                        float  longitudalCutoff,
                        float  transCutoff,
                        float  rtheta,
                        float  theta2,
                        float  r2,
                        float  estDimension,
                        float  sad,
                        float  rs,
                        int    gpuid)
{

#ifdef coalescing_optimize
    thrust::device_vector<float> d_beamDirect_x(num_beam);
		thrust::device_vector<float> d_beamDirect_y(num_beam);		
		thrust::device_vector<float> d_beamDirect_z(num_beam);
		auto d_beamDirect_xptr = d_beamDirect_x.data().get();
		auto d_beamDirect_yptr = d_beamDirect_y.data().get();
		auto d_beamDirect_zptr = d_beamDirect_z.data().get();
		thrust::for_each(
			thrust::device,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(num_beam),
			[d_beamDirect_xptr,
			 d_beamDirect_yptr,
			 d_beamDirect_zptr,
			 d_beamDirect] __device__ (int i) {
				d_beamDirect_xptr[i] = d_beamDirect[i].x;
				d_beamDirect_yptr[i] = d_beamDirect[i].y;
				d_beamDirect_zptr[i] = d_beamDirect[i].z;
			});

#endif
	
	void* args[] = {
		(void*)&dose, (void*)&rbe2, (void*)&counter, (void*)&dalpha, (void*)&dbeta, 
#ifdef coalescing_optimize
		(void*)&(d_beamDirect_xptr), (void*)&(d_beamDirect_yptr), (void*)&(d_beamDirect_zptr),
#else
		(void*)&d_beamDirect, 
#endif
		(void*)&num_beam,
		(void*)&source, (void*)&bmxdir, (void*)&bmydir, (void*)&roiIndex, (void*)&num_roi, (void*)&doseGrid, (void*)&rayweqData,
		(void*)&iddData, (void*)&profileData, (void*)&subspotData, (void*)&letData, (void*)&lqData, (void*)&rayweqSetting,
		(void*)&iddDepth, (void*)&profileDepth, (void*)&d_idbeamxy, (void*)&d_numParticles, (void*)&nsubspot,
		(void*)&nGauss, (void*)&eneIdx, (void *)&fieldId, (void*)&beamParaPos,(void*)&longitudalCutoff, (void*)&transCutoff,
		(void*)&rtheta, (void*)&theta2, (void*)&estDimension,(void*)&sad, (void*)&rs,
	};
	int batchSize = queryBatchSize(cuCalFinalDoseAndRBEKernel, gpuid);
	cudaFuncSetCacheConfig(cuCalFinalDoseAndRBEKernel, cudaFuncCachePreferL1);
	launchCudaKernel(cuCalFinalDoseAndRBEKernel, batchSize, args);
}

void calFinalDoseAndMKM(float* dose,
                        float* Z1Dmix,
                        vec3f* d_beamDirect,
                        int    num_beam,
                        vec3f  source,
                        vec3f  bmxdir,
                        vec3f  bmydir,
                        vec3i* roiIndex,
                        int    num_roi,
                        Grid   doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t Z1DijData,
                        vec3f  rayweqSetting,
                        vec3f  iddDepth,
                        vec3f  profileDepth,
                        float* d_idbeamxy,
                        int*   d_numParticles,
                        int    nsubspot,
                        int    nGauss,
                        int    eneIdx,
                        int    fieldId,
                        float  beamParaPos,
                        float  longitudalCutoff,
                        float  transCutoff,
                        float  rtheta,
                        float  theta2,
                        float  r2,
                        float  estDimension,
                        float  sad,
                        float  rs,
                        float  energy,
                        int    gpuid)
{
	void* args[] = {
		(void*)&dose, (void*)&Z1Dmix, (void*)&d_beamDirect, (void*)&num_beam, (void*)&source,
		(void*)&bmxdir, (void*)&bmydir, (void*)&roiIndex, (void*)&num_roi, (void*)&doseGrid, (void*)&rayweqData,
		(void*)&iddData, (void*)&profileData, (void*)&subspotData, (void*)&Z1DijData, (void*)&rayweqSetting,
		(void*)&iddDepth, (void*)&profileDepth, (void*)&d_idbeamxy, (void*)&d_numParticles, (void*)&nsubspot,
		(void*)&nGauss, (void*)&eneIdx, (void *)&fieldId, (void*)&beamParaPos,(void*)&longitudalCutoff, (void*)&transCutoff,
		(void*)&rtheta, (void*)&theta2, (void*)&estDimension,(void*)&sad, (void*)&rs, (void*)&energy
	};
	int batchSize = queryBatchSize(cuCalFinalDoseAndMKMKernel, gpuid);
	launchCudaKernel(cuCalFinalDoseAndMKMKernel, batchSize, args);
}

template<typename T>
struct sqrt_op
{
	__device__ 
	T operator()(T x)
	{
		return sqrtf(x);
	}
};

void calSqrt(float *first, float *last, float *result)
{
	thrust::transform(thrust::device, first, last, result, sqrt_op<float>());
}

void calDoseSumSquares(float *d_doseNorm,
	vec3f *d_beamDirect,
	int num_beam,
	vec3f source,
	vec3f bmxdir,
	vec3f bmydir,
	vec3i *roiIndex,
	int num_roi,
	Grid doseGrid,
	cudaTextureObject_t rayweqData,
	cudaTextureObject_t iddData,
	cudaTextureObject_t profileData,
	cudaTextureObject_t subspotData,
	vec3f rayweqSetting,
	vec3f iddDepth,
	vec3f profileDepth,
	float *d_idbeamxy,
	int nsubspot,
	int nGauss,
	int eneIdx,
	int beamOffset,
	float beamParaPos,
	float longitudalCutoff,
	float transCutoff,
	float rtheta,
	float theta2,
	float r2,
	float sad,
	int gpuid)
{
	void *args[] = {(void *)&d_doseNorm,
		(void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir, (void *)&bmydir,
		(void *)&roiIndex,
		(void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData, (void *)&iddData, (void *)&profileData,
		(void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
		(void *)&profileDepth, (void *)&d_idbeamxy, (void *)&nsubspot,
		(void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
		(void *)&longitudalCutoff, (void *)&transCutoff,
		(void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};
	
	int batchSize = queryBatchSize(calDoseSumSquareKernel, gpuid);
	launchCudaKernel2D(calDoseSumSquareKernel, batchSize, args);
}

int calOneLayerDoseOld(float *dose,
					   int *cscIndices,
					   int *colindices,
					   int *cscIndptr,
					   int totalnnz,
					   int *sumNNZ,
					   vec3f *d_beamDirect,
					   int num_beam,
					   vec3f source,
					   vec3f bmxdir,
					   vec3f bmydir,
					   vec3i *roiIndex,
					   int num_roi,
					   Grid doseGrid,
					   cudaTextureObject_t rayweqData,
					   cudaTextureObject_t iddData,
					   cudaTextureObject_t profileData,
					   cudaTextureObject_t subspotData,
					   vec3f rayweqSetting,
					   vec3f iddDepth,
					   vec3f profileDepth,
					   float *d_idbeamxy,
					   int nsubspot,
					   int nGauss,
					   int eneIdx,
					   int beamOffset,
					   float beamParaPos,
					   float longitudalCutoff,
					   float transCutoff,
					   float rtheta,
					   float theta2,
					   float r2,
					   float sad,
					   int gpuid) {

  void *args[] = {(void *)&dose, (void *)&cscIndices,
				  (void *)&colindices,
				  (void *)&cscIndptr, (void *)&totalnnz, (void *)&sumNNZ,
				  (void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir, (void *)&bmydir,
				  (void *)&roiIndex,
				  (void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData, (void *)&iddData, (void *)&profileData,
				  (void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
				  (void *)&profileDepth, (void *)&d_idbeamxy, (void *)&nsubspot,
				  (void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
				  (void *)&longitudalCutoff, (void *)&transCutoff,
				  (void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};

  int batchSize = queryBatchSize(calDoseKernel, gpuid);
  launchCudaKernel(calDoseKernel, batchSize, args);
  int *d_colInd1;
  int h_NNZ;
  checkCudaErrors(cudaMemcpy(&h_NNZ, sumNNZ, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc((void **)&d_colInd1, sizeof(int) * h_NNZ));
  checkCudaErrors(cudaMemcpy(d_colInd1, colindices, sizeof(int) * h_NNZ, cudaMemcpyDeviceToDevice));

  thrust::sort_by_key(thrust::device, colindices, colindices + h_NNZ, cscIndices);
  thrust::sort_by_key(thrust::device, d_colInd1, d_colInd1 + h_NNZ, dose);

  std::vector<int> h_cscIndptr(num_beam);
  checkCudaErrors(cudaMemcpy(h_cscIndptr.data(), cscIndptr+1, sizeof(int) * num_beam, cudaMemcpyDeviceToHost));
  int start = 0;
  int end   = 0;

  std::vector<std::thread> threads;
	threads.reserve(num_beam);
	for (int i = 0; i < num_beam; ++i)
	{
			end += h_cscIndptr[i];
			if (end > start)
			{
					threads.emplace_back([=]() {
							thrust::sort_by_key(thrust::device,
																	cscIndices + start,
																	cscIndices + end,
																	dose + start);
					});
			}
			start += h_cscIndptr[i];
	}

	for (auto &t : threads)
	{
			t.join();
	}

  checkCudaErrors(cudaFree(d_colInd1));
  return h_NNZ;
}

int calOneLayerDoseOld_(float *dose,
					   int *cscIndices,
					   int *colindices,
					   int *cscIndptr,
						 float* d_idd_per_depth, //suplementary
					   int totalnnz,
					   int *sumNNZ,
					   vec3f *d_beamDirect,
					   int num_beam,
					   vec3f source,
					   vec3f bmxdir,
					   vec3f bmydir,
					   vec3i *roiIndex,
					   int num_roi,
					   Grid doseGrid,
					   cudaTextureObject_t rayweqData,
					   cudaTextureObject_t iddData,
					   cudaTextureObject_t profileData,
					   cudaTextureObject_t subspotData,
					   vec3f rayweqSetting,
					   vec3f iddDepth,
					   vec3f profileDepth,
					   float *d_idbeamxy,
					   int nsubspot,
					   int nGauss,
					   int eneIdx,
					   int beamOffset,
					   float beamParaPos,
					   float longitudalCutoff,
					   float transCutoff,
					   float rtheta,
					   float theta2,
					   float r2,
					   float sad,
					   int gpuid) {

  void *args[] = {(void *)&dose, (void *)&cscIndices,
				  (void *)&colindices,
				  (void *)&cscIndptr,(void *)&d_idd_per_depth, (void *)&totalnnz, (void *)&sumNNZ,//d_idd_per_depth suplementary
				  (void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir, (void *)&bmydir,
				  (void *)&roiIndex,
				  (void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData, (void *)&iddData, (void *)&profileData,
				  (void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
				  (void *)&profileDepth, (void *)&d_idbeamxy, (void *)&nsubspot,
				  (void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
				  (void *)&longitudalCutoff, (void *)&transCutoff,
				  (void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};

  int batchSize = queryBatchSize(calDoseKernel_, gpuid);
  launchCudaKernel(calDoseKernel_, batchSize, args);
  int *d_colInd1;
  int h_NNZ;
  checkCudaErrors(cudaMemcpy(&h_NNZ, sumNNZ, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc((void **)&d_colInd1, sizeof(int) * h_NNZ));
  checkCudaErrors(cudaMemcpy(d_colInd1, colindices, sizeof(int) * h_NNZ, cudaMemcpyDeviceToDevice));

  thrust::sort_by_key(thrust::device, colindices, colindices + h_NNZ, cscIndices);
  thrust::sort_by_key(thrust::device, d_colInd1, d_colInd1 + h_NNZ, dose);

  std::vector<int> h_cscIndptr(num_beam);
  checkCudaErrors(cudaMemcpy(h_cscIndptr.data(), cscIndptr+1, sizeof(int) * num_beam, cudaMemcpyDeviceToHost));
  int start = 0;
  int end   = 0;
  for (int i = 0; i < num_beam; ++i)
  {
      end += h_cscIndptr[i];
      if (end > start)
      {
          thrust::sort_by_key(thrust::device,
						                  cscIndices + start,
						                  cscIndices + end,
                              dose + start);
      }
      start += h_cscIndptr[i];
  }

  checkCudaErrors(cudaFree(d_colInd1));
  return h_NNZ;
}

void RBEMap(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap, float* rbe2,
            float* counter, float* phyDose, int nField, int gpuid)
{
	void* args[] = {
		(void*)&num, (void*)&doseGrid, (void*)&roiIndex, (void*)&alphamap, (void*)&betamap, (void*)&rbemap, (void*)&rbe2,
		(void*)&counter, (void*)&phyDose, (void*)&nField
	};
	int batchSize = queryBatchSize(LQRBEKernel, gpuid);
	launchCudaKernel(LQRBEKernel, batchSize, args);
}

void mkmRBEMap(int num, Grid doseGrid, vec3i* roiIndex, float* rbemap1, float* rbemap2, float* Z1Dmix,
               float* phyDose,
               vec2f alpha0AndBeta0, int nField, int gpuid)
{
	void* args[] = {
		(void*)&num, (void*)&doseGrid, (void*)&roiIndex, (void*)&rbemap1, (void*)&rbemap2, (void*)&Z1Dmix,
		(void*)&phyDose, (void*)&alpha0AndBeta0, (void*)&nField,
	};
	int batchSize = queryBatchSize(mkmRBEMapKernel, gpuid);
	launchCudaKernel(mkmRBEMapKernel, batchSize, args);
}

void setROI(vec2i *d_roiFlag, vec3i *d_roiInd, vec3f *d_center, Grid grid, int size) {
    int batchSize = queryBatchSize(setROIKernel, 0);
    void *args[] = {(void *)&d_roiFlag, (void *)&d_roiInd, (void *)&d_center, (void *)&grid, (void *)&size};
    launchCudaKernel(setROIKernel, batchSize, args);
}

int rotate3DArray(Grid &new_g,
                  const Grid &old_g,
                  float *h_Array_new,
                  float *h_Array_old,
                  float *d_rot_forward,
                  int type,
                  int gpuId) {
    auto *h_rot_forward = new vec4f[4];
    auto *h_rot_backward = new vec4f[4];

    vec4f *d_rot_backward;
    float *d_array_new;

    cudaArray_t old3DArray;
    cudaTextureObject_t old3DArrayObj;

    weq::create3DTexture(h_Array_old,
                         &old3DArray,
                         &old3DArrayObj,
                         weq::LINEAR,
                         weq::R,
                         old_g.dims.z,
                         old_g.dims.y,
                         old_g.dims.x,
                         type,
                         gpuId);

    h_rot_forward[0] = vec4f(d_rot_forward[0], d_rot_forward[1], d_rot_forward[2], d_rot_forward[3]);
    h_rot_forward[1] = vec4f(d_rot_forward[4], d_rot_forward[5], d_rot_forward[6], d_rot_forward[7]);
    h_rot_forward[2] = vec4f(d_rot_forward[8], d_rot_forward[9], d_rot_forward[10], d_rot_forward[11]);
    h_rot_forward[3] = vec4f(d_rot_forward[12], d_rot_forward[13], d_rot_forward[14], d_rot_forward[15]);

    h_rot_backward[0] = vec4f(d_rot_forward[0], d_rot_forward[4], d_rot_forward[8], -d_rot_forward[3]);
    h_rot_backward[1] = vec4f(d_rot_forward[1], d_rot_forward[5], d_rot_forward[9], -d_rot_forward[7]);
    h_rot_backward[2] = vec4f(d_rot_forward[2], d_rot_forward[6], d_rot_forward[10], -d_rot_forward[11]);
    h_rot_backward[3] = vec4f(d_rot_forward[12], d_rot_forward[13], d_rot_forward[14], d_rot_forward[15]);

    vec3f corner_1 = old_g.corner;
    vec3f corner_2 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x, old_g.corner.y, old_g.corner.z);
    vec3f corner_3 = vec3f(old_g.corner.x, old_g.corner.y + old_g.resolution.y * old_g.dims.y, old_g.corner.z);
    vec3f corner_4 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z);

    vec3f corner_5 = vec3f(old_g.corner.x, old_g.corner.y, old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_6 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_7 = vec3f(old_g.corner.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_8 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);

    vec3f new_corner_1 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)));
    vec3f new_corner_2 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)));
    vec3f new_corner_3 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)));
    vec3f new_corner_4 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)));

    vec3f new_corner_5 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)));
    vec3f new_corner_6 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)));
    vec3f new_corner_7 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)));
    vec3f new_corner_8 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)));

    float
            new_corner_x[8] = {new_corner_1.x, new_corner_2.x, new_corner_3.x, new_corner_4.x, new_corner_5.x, new_corner_6.x,
                               new_corner_7.x, new_corner_8.x};
    float
            new_corner_y[8] = {new_corner_1.y, new_corner_2.y, new_corner_3.y, new_corner_4.y, new_corner_5.y, new_corner_6.y,
                               new_corner_7.y, new_corner_8.y};
    float
            new_corner_z[8] = {new_corner_1.z, new_corner_2.z, new_corner_3.z, new_corner_4.z, new_corner_5.z, new_corner_6.z,
                               new_corner_7.z, new_corner_8.z};

    float min_x = *std::min_element(new_corner_x, new_corner_x + 8);
    float min_y = *std::min_element(new_corner_y, new_corner_y + 8);
    float min_z = *std::min_element(new_corner_z, new_corner_z + 8);

    float max_x = *std::max_element(new_corner_x, new_corner_x + 8);
    float max_y = *std::max_element(new_corner_y, new_corner_y + 8);
    float max_z = *std::max_element(new_corner_z, new_corner_z + 8);

    new_g.corner = vec3f(min_x, min_y, min_z);
    new_g.upperCorner = vec3f(max_x, max_y, max_z);

    vec3f dims = (new_g.upperCorner - new_g.corner) / old_g.resolution + 1.0f;

    new_g.dims = vec3i(int(dims.x), int(dims.y), int(dims.z));
    new_g.resolution = old_g.resolution;
    checkCudaErrors(cudaMalloc((void **)&d_rot_backward, sizeof(vec4f) * 4));
    checkCudaErrors(cudaMalloc((void **)&d_array_new, sizeof(float) * new_g.dims.x * new_g.dims.y * new_g.dims.z));
    checkCudaErrors(cudaMemcpy(d_rot_backward, h_rot_backward, sizeof(vec4f) * 4, cudaMemcpyHostToDevice));

    int batchSize = queryBatchSize(rotate3DArrayKernel, gpuId);
    void *args[] = {(void *)&new_g, (void *)&d_array_new, (void *)&old3DArrayObj, (void *)&d_rot_backward};

    launchCudaKernel(rotate3DArrayKernel, batchSize, args);

    checkCudaErrors(cudaMemcpy(h_Array_new,
                               d_array_new,
                               sizeof(float) * new_g.dims.x * new_g.dims.y * new_g.dims.z,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_rot_backward));
    checkCudaErrors(cudaFree(d_array_new));
    checkCudaErrors(cudaFreeArray(old3DArray));
    checkCudaErrors(cudaDestroyTextureObject(old3DArrayObj));

    delete[] h_rot_forward;
    delete[] h_rot_backward;
    return new_g.dims.x * new_g.dims.y * new_g.dims.z;
}