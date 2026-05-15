//
// Created by 王子路 on 2022/5/24.
//

#ifndef CUDACMC__MACRO_H_
#define CUDACMC__MACRO_H_

#define INF 1.0e20
#define WATERDENSITY 1.0 // g/cm^3
#define MP 938.272046	//proton mass, in MeV
#define CP 1.00000 //proton charge
#define ME 0.510998928  //electron mass, in MeV
#define MO 14903.3460795634 //oxygen mass in MeV
#define MINELECTRONENERGY 0.1 // MeV
#define TWOPIRE2MENEW 0.08515495201157892 //2pi*r_e^2*m_e*n_{ew}, where r_e in cm, m_e in eV, n_ew = 3.34e23/cm^3
#define XW 36.514 	//radiation length of water, in cm
#define PI 3.1415926535897932384626433
#define SECONDPARTICLEVOLUME 10000
#define EMINPOI 1.0	//minimun energy used in p-o inelastic event, in MeV
#define EBIND 3.0	//initial binding energy used in p-o inelastic, in MeV
#define MAXSTEP 0.32 //in cm
#define MAXENERGYRATIO 0.2 //Max energy decay ratio of initial energy in a step
#define MINSECONDENERGY 1.0 //Min proton energy to transport
#define ZERO 1e-7
#define EPSILON 1e-5
#define MC 11174.86339 //carbon mass in MeV
#define CC 6.0000      //carbon charge
#define ES 25          //MeV, ES parameter for secondary particles
#define MINCARBONENERGY 5.0 //Min carbon energy to transport in MeV
#define PPETHRESHOLD 10.0 // energy threshold of proton proton interaction

// ROI and convolution related macros
#define ROI_MARGIN_DEFAULT 5.0f  // Default margin for ROI calculation
#define WEIGHT_CUTOFF 1e-6f      // Weight threshold for subspot filtering
#define SIGMA_CUTOFF_DEFAULT 3.0f // Default sigma cutoff for convolution
#define BP_DEPTH_CUTOFF 1.05f    // Depth cutoff as a percentage relative to the Bragg peak depth
#define POETHRESHOLD 7.0 // energy threshold of proton oxygen elastic interaction
#define POITHRESHOLD 20.0 // energy threshold of proton oxygen inelastic interaction
#define NDOSECOUNTERS 1 // number of dosecounters
#define PARTICLESCORETYPES 8//
#define OXYGEN 15.9994 // oxygen atomic mass
#define HYDROGEN 1.00794 //hydrogen atomic mass
#define BEAM_PARTICLE_CUTOFF 5 //ray weight (by particles num)
#define MAX_TRACE_STEPS 1000 //ray tracing from effective source(x0,y0) to the last inside, maximum total steps
//#define GRIDDIM 256
//#define BLOCKDIM 128
#define SECONDARYCATEGORIES 8
#define EXCITATION 75.0e-6
#define MEV2JOULES 1.6021773e-13
#define BATCHSIZE 65536

// ROI and margin constants
#define ROI_MARGIN_X 5.0f  // ROI margin in X direction (cm)
#define ROI_MARGIN_Y 5.0f  // ROI margin in Y direction (cm)  
#define ROI_MARGIN_Z 2.0f  // ROI margin in Z direction (cm)
#define MIN(a,b) (a > b ? b : a)
#define MIN3(a,b,c) (a > b ? b : a) > c ? c : (a > b ? b : a)
#define ABS(a) a > 0 ? a : -a
// #define WEIGHT_CUTOFF 0.001f  // Commented out to avoid redefinition
#define SIGMA_CUTOFF 3.0f
#define ROI_MARGIN 2.0f  // ROI边界扩展margin (cm)

// Superposition算法相关常量
#define MAX_SUPERP_RADIUS 32
#define SUPERP_TILE_X 32  // 等于warp大小
#define SUPERP_TILE_Y 8   // 优化内存合并访问
#define MIN_TILES_IN_BATCH 16
#define KS_SIGMA_CUTOFF 3.0f

#define __IFGPU__ 1
#define __LINUX__ 1

#define __SCOREDOSE2WATER__ 0
//	if __SCOREDOSE2WATER__ is turned on, 1--on, 0--off
#define __ONLYEM__ 0
//	if only EM interaction is turned on, 1--on, 0--off

#endif //CUDACMC__MACRO_H_

// Minimal float3 helpers if not already available
#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER __host__ __device__
#endif


#ifndef HALF
#define HALF 0.5f
#endif
