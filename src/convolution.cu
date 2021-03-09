#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cufft.h>

extern "C" {
#include "cudaFIR.h"
}

static __global__ void cufftComplexPointwiseMul(cufftComplex *, const cufftComplex *, const cufftComplex *);
static __global__ void AddOverlapScale(cufftReal *a, const cufftReal *b, float scale);
static __global__ void ShiftAndPad(const cufftReal *a, cufftReal *b, int shift ,int len);

static  int cudaFIRintialised=0;
static  cufftReal *d_signal;
static  cufftComplex *d_signal_fft;
static  cufftComplex *d_filter_fft;
static  cufftComplex *d_tmp_fft;
static  cufftReal *d_tmp_signal;
static  cufftReal *d_convolved_signal[2];
static  cufftHandle fplan,bplan;

#define NBTHREADS 256
#define NBCHANN (cvparam->nbch)
#define PART_SIZE (cvparam->partsz)
#define FILTER_NPART (cvparam->nbpart)
#define FFT_SIZE (2*PART_SIZE)
#define FFT_CSIZE (((FFT_SIZE/2+512)/512)*512)

int readFilter(char *filterpath,conv_param_t *cvparam)
{
        FILE *fd;
        int size;
        float *filter;
  	cudaError_t cerr;

        fd=fopen(filterpath,"r");
        if(fd==NULL) return -1;

	fseek(fd, 0, SEEK_END);
	size=ftell(fd)/sizeof(float)/cvparam->nbch;
	rewind(fd);
	cvparam->nbpart = (size+cvparam->partsz-1)/cvparam->partsz;
       	fprintf(stderr,"cudaFIR using filter %s sz:%d nb part:%d\n",filterpath,size,cvparam->nbpart);

        size=cvparam->nbch*cvparam->partsz*cvparam->nbpart;

        filter=(float*)calloc(size,sizeof(float));
        if(filter==NULL) {
                fclose(fd);
                return -1;
        }

        if(fread(filter,sizeof(float),size,fd)==0) {
       		fprintf(stderr,"cudaFIR read filter error \n");
                free(filter);
                fclose(fd);
                return -1;
        }
        fclose(fd);

  cudaStreamSynchronize(0);

  cudaMalloc((void **)(&(d_filter_fft)), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN*cvparam->nbpart);
  // compute fft filter parts
  for(int n=0; n < cvparam->nbpart ; n++ ) {
  	// Copy host memory to device
  	cudaMemcpy(d_signal, &(filter[n*NBCHANN*PART_SIZE]), sizeof(float)*PART_SIZE*NBCHANN, cudaMemcpyHostToDevice);
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: Memcpy filter\n");
		return -1;
  	}

  	cufftExecR2C(fplan, d_signal, &(d_filter_fft[n*NBCHANN*FFT_CSIZE]));
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: ExecR2C %d\n",n);
		return -1;
  	}
  }
  free(filter);

  cudaMalloc((void **)(&(d_convolved_signal[0])), sizeof(cufftReal)*NBCHANN*(FILTER_NPART+1)*PART_SIZE);
  cudaMalloc((void **)(&(d_convolved_signal[1])), sizeof(cufftReal)*NBCHANN*(FILTER_NPART+1)*PART_SIZE);
  cudaMemset(d_convolved_signal[0], 0, sizeof(cufftReal)*NBCHANN*(FILTER_NPART+1)*PART_SIZE);
  cudaMemset(d_convolved_signal[1], 0, sizeof(cufftReal)*NBCHANN*(FILTER_NPART+1)*PART_SIZE);

  cudaHostAlloc(&(cvparam->inoutbuff[0]),sizeof(float)*FFT_SIZE*NBCHANN,0);
  if ((cerr=cudaGetLastError()) != cudaSuccess){
	fprintf(stderr, "cudaFIR  : cudaAlllocHost error %s %s\n",cudaGetErrorName(cerr),cudaGetErrorString(cerr));
	return -1;
  }
  cudaHostAlloc(&(cvparam->inoutbuff[1]),sizeof(float)*FFT_SIZE*NBCHANN,0);
  if ((cerr=cudaGetLastError()) != cudaSuccess){
	fprintf(stderr, "cudaFIR  : cudaAlllocHost error %s %s\n",cudaGetErrorName(cerr),cudaGetErrorString(cerr));
	return -1;
  }
  memset(cvparam->inoutbuff[0],0,sizeof(float)*FFT_SIZE*NBCHANN);
  memset(cvparam->inoutbuff[1],0,sizeof(float)*FFT_SIZE*NBCHANN);

  cvparam->nbf=0;

  return 0;
}

void freeFilter(conv_param_t *cvparam) { 

  cudaStreamSynchronize(0);

  if(d_filter_fft) cudaFree(d_filter_fft);
  if(d_convolved_signal[0]) cudaFree(d_convolved_signal[0]);
  if(d_convolved_signal[1]) cudaFree(d_convolved_signal[1]);
  if(cvparam->inoutbuff[0]) cudaFreeHost(cvparam->inoutbuff[0]);
  if(cvparam->inoutbuff[1]) cudaFreeHost(cvparam->inoutbuff[1]);

  d_filter_fft=NULL;
  d_convolved_signal[0]=NULL;
  d_convolved_signal[1]=NULL;
  cvparam->inoutbuff[0]=NULL;
  cvparam->inoutbuff[1]=NULL;
}


int initConvolve(conv_param_t *cvparam) { 

  if(cudaFIRintialised) return 0;

  if(cvparam->partsz%NBTHREADS) {
	fprintf(stderr, "cudaFIR : partsz must be a multiple of %d\n",NBTHREADS);
	return -1;
  }

  // Allocate device memory 
  cudaMalloc((void **)(&d_signal), sizeof(cufftReal)*FFT_SIZE*NBCHANN);
  cudaMalloc((void **)(&d_signal_fft), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN);
  cudaMalloc((void **)(&d_tmp_fft), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN);
  cudaMalloc((void **)(&d_tmp_signal), sizeof(cufftReal)*FFT_SIZE*NBCHANN);

  // CUFFT plan 
  int inembed=1;
  int onembed=1;
  int fftsz=FFT_SIZE; 

  cufftPlanMany(&fplan,1, &fftsz, &inembed, NBCHANN, inembed, &onembed, NBCHANN, onembed, CUFFT_R2C, NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: Plan1d\n");
	return -1;
  }
  cufftPlanMany(&bplan,1, &fftsz, &inembed, NBCHANN, inembed, &onembed, NBCHANN, onembed, CUFFT_C2R, NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: Plan1d\n");
	return -1;
  }

  // pad signals
  cudaMemset(d_signal+PART_SIZE*NBCHANN, 0, sizeof(cufftReal) * PART_SIZE * NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: MulAndScale\n");
	return -1;
  }

  d_convolved_signal[0]=NULL;
  d_convolved_signal[1]=NULL;
  d_filter_fft=NULL;
  cvparam->inoutbuff[0]=NULL;
  cvparam->inoutbuff[1]=NULL;

  cudaFIRintialised=1;
  return 0;
}

void waitConvolve(void)
{
	cudaStreamSynchronize(0);
}

int cudaConvolve(conv_param_t *cvparam) { 
  int bk=cvparam->nbf^1;
  int nbk;

  cudaMemcpyAsync(d_signal,cvparam->inoutbuff[bk], sizeof(float)*PART_SIZE*NBCHANN, cudaMemcpyHostToDevice,0);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: fail to Memcpy d_signal\n");
	return -1;
  }

  /* signal FFT */
  cufftExecR2C(fplan, d_signal, d_signal_fft);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: ExecR2C\n");
	return -1;
  }

  for(int n=0; n < FILTER_NPART ; n++ ) {
	/*  signal fft * filter part fft */
  	cufftComplexPointwiseMul<<<(FFT_CSIZE*NBCHANN/NBTHREADS),NBTHREADS>>>(d_tmp_fft, d_signal_fft, &(d_filter_fft[n*FFT_CSIZE*NBCHANN]));
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: Mul\n");
		return -1;
  	}

	/* ifft */
  	cufftExecC2R(bplan, d_tmp_fft, d_tmp_signal);
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: ExecC2R\n");
		return -1;
  	}

	/* overlap and add result */
  	AddOverlapScale<<<FFT_SIZE*NBCHANN/NBTHREADS,NBTHREADS>>>(&(d_convolved_signal[bk][n*PART_SIZE*NBCHANN]), d_tmp_signal, 1.0/(float)FFT_SIZE);
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: AddOverlap\n");
		return -1;
  	}

  }

  /* result device to host copy  */
  cudaMemcpyAsync(cvparam->inoutbuff[bk], d_convolved_signal[bk], sizeof(float) * PART_SIZE * NBCHANN, cudaMemcpyDeviceToHost,0);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: fail to Memcpy d_convolved\n");
	return -1;
  }

  nbk=bk^1;
  // shift
  ShiftAndPad<<<(((FILTER_NPART+1)*PART_SIZE)*NBCHANN/NBTHREADS),NBTHREADS>>>(d_convolved_signal[bk], d_convolved_signal[nbk],PART_SIZE*NBCHANN, FILTER_NPART*PART_SIZE*NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: Shift kernel error\n");
	return -1;
    }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////
static __global__ void cufftComplexPointwiseMul(cufftComplex *a, const cufftComplex *b, const cufftComplex *c) {
  const int tID = blockIdx.x * blockDim.x + threadIdx.x;

    a[tID] = cuCmulf(b[tID], c[tID]);
}

static __global__ void AddOverlapScale(cufftReal *a, const cufftReal *b, float scale) {
  const int tID = blockIdx.x * blockDim.x + threadIdx.x;

    a[tID] = a[tID]+ b[tID]*scale;
}

static __global__ void ShiftAndPad(const cufftReal *a, cufftReal *b, int shift ,int len) {
  const int tID = blockIdx.x * blockDim.x + threadIdx.x;

  if(tID<len)
    b[tID] = a[tID+shift];
  else
    b[tID]=0;

}
