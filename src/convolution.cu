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

static  cufftReal *d_signal;
static  cufftComplex *d_signal_fft;
static  cufftComplex *d_filter_fft[NBFILTER];
static  cufftComplex *d_tmp_fft;
static  cufftReal *d_tmp_signal;
static  cufftReal *d_convolved_signal[2];
static  cufftHandle fplan,bplan;

static int bk=0,nbk;

#define PART_SIZE (cvparam->partsz)
#define FILTER_NPART (cvparam->nbpart)
#define NBCHANN (cvparam->nbch)
#define FFT_SIZE (2*PART_SIZE)
#define FFT_CSIZE (((FFT_SIZE/2+512)/512)*512)
#define FILTER_SIZE (PART_SIZE*FILTER_NPART)
#define NBTHREADS 256

int readFilter(char *filterpath,conv_param_t *cvparam,int nf)
{
        FILE *fd;
        int n,size;
        float *filter;

        fd=fopen(filterpath,"r");
        if(fd==NULL) return -1;

        fprintf(stderr,"Loading filter %s\n",filterpath);

	if(cvparam->nbpart==0) {
		fseek(fd, 0, SEEK_END);
		cvparam->nbpart = (ftell(fd)/sizeof(float)/cvparam->nbch+cvparam->partsz-1)/cvparam->partsz;
		rewind(fd);
	}

        size=cvparam->nbch*cvparam->partsz*cvparam->nbpart;

        filter=(float*)calloc(size,sizeof(float));
        if(filter==NULL) {
                fclose(fd);
                return -1;
        }

        if((n=fread(filter,sizeof(float),size,fd))==0) {
                free(filter);
                fclose(fd);
                return -1;
        }
        fclose(fd);

        if(n!=size) {
                fprintf(stderr,"Warning %s : too short filter (padded)\n",filterpath);
        }

        addFilter(cvparam,filter,nf);

        free(filter);
        return 0;
}

int initConvolve(conv_param_t *cvparam) { 
  int n;

  if(cvparam->partsz%NBTHREADS) {
	fprintf(stderr, "cudaFIR : partsz must be a multiple of %d\n",NBTHREADS);
	return -1;
  }

  cudaHostAlloc(&(cvparam->inoutbuff),sizeof(float)*FFT_SIZE*NBCHANN,0);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR  : cudaAlllocHost error\n");
	return -1;
  }
  cudaHostGetDevicePointer((void **)(&d_signal),cvparam->inoutbuff,0);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR  : GetDevicePointer error\n");
	return -1;
  }

  // Allocate device memory 
  cudaMalloc((void **)(&d_signal_fft), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN);
  cudaMalloc((void **)(&d_tmp_fft), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN);
  cudaMalloc((void **)(&d_tmp_signal), sizeof(cufftReal)*FFT_SIZE*NBCHANN);
  cudaMalloc((void **)(&(d_convolved_signal[0])), sizeof(cufftReal)*(FILTER_SIZE+PART_SIZE)*NBCHANN);
  cudaMalloc((void **)(&(d_convolved_signal[1])), sizeof(cufftReal)*(FILTER_SIZE+PART_SIZE)*NBCHANN);

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

  cudaMemset(d_convolved_signal[0], 0, sizeof(cufftReal)*(FILTER_SIZE+PART_SIZE)*NBCHANN);
  cudaMemset(d_convolved_signal[1], 0, sizeof(cufftReal)*(FILTER_SIZE+PART_SIZE)*NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: Memset\n");
	return -1;
  }

  for(n=0;n<NBFILTER;n++)
  	d_filter_fft[n]=NULL;

  return 0;
}

int addFilter(conv_param_t *cvparam, float *h_filter, int nf) { 

  cudaMalloc((void **)(&(d_filter_fft[nf])), sizeof(cufftComplex)*FFT_CSIZE*NBCHANN*FILTER_NPART);

  // compute fft filter parts
  for(int n=0; n < FILTER_NPART ; n++ ) {
  	// Copy host memory to device
  	cudaMemcpy(d_signal, &(h_filter[n*NBCHANN*PART_SIZE]), sizeof(float)*PART_SIZE*NBCHANN, cudaMemcpyHostToDevice);
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: Memcpy filter\n");
		return -1;
  	}

  	cufftExecR2C(fplan, d_signal, &(d_filter_fft[nf][n*NBCHANN*FFT_CSIZE]));
  	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "cudaFIR: ExecR2C\n");
		return -1;
  	}
  }

  return 0;
}

void waitConvolve(void)
{
	cudaStreamSynchronize(0);
}

int cudaConvolve(conv_param_t *cvparam) { 

  /* signal FFT */
  cufftExecR2C(fplan, d_signal, d_signal_fft);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: ExecR2C\n");
	return -1;
  }

  for(int n=0; n < FILTER_NPART ; n++ ) {
	/*  signal fft * filter part fft */
  	cufftComplexPointwiseMul<<<(FFT_CSIZE*NBCHANN/NBTHREADS),NBTHREADS>>>(d_tmp_fft, d_signal_fft, &(d_filter_fft[cvparam->nf][n*FFT_CSIZE*NBCHANN]));
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
  cudaMemcpyAsync(d_signal, d_convolved_signal[bk], sizeof(float) * PART_SIZE * NBCHANN, cudaMemcpyDeviceToDevice,0);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: fail to Memcpy d_convolved\n");
	return -1;
  }

  // shift
  nbk=bk^1;
  ShiftAndPad<<<((FILTER_SIZE+PART_SIZE)*NBCHANN/NBTHREADS),NBTHREADS>>>(d_convolved_signal[bk], d_convolved_signal[nbk],PART_SIZE*NBCHANN, FILTER_SIZE*NBCHANN);
  if (cudaGetLastError() != cudaSuccess){
	fprintf(stderr, "cudaFIR: Shift kernel error\n");
	return -1;
    }
  bk=nbk;

  return 0;
}

void freeFilter(void)
{
  int n;

  // Destroy CUFFT context
  cufftDestroy(fplan);
  cufftDestroy(bplan);

  cudaFree(d_signal);
  cudaFree(d_signal_fft);
  for(n=0;n<NBFILTER;n++)
  	if(d_filter_fft[n])
  		cudaFree(d_filter_fft[n]);
  cudaFree(d_tmp_fft);
  cudaFree(d_tmp_signal);
  cudaFree(d_convolved_signal[0]);
  cudaFree(d_convolved_signal[1]);
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
