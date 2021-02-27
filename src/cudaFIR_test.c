#include <alsa/asoundlib.h>
#include <alsa/pcm_external.h>
#include "cudaFIR.h"

static int filter_FS[NBFILTER]={ 44100, 48000, 88200, 96000 , 176400, 192000, 352800 ,384000, 705800 , 768000 };
static char *filter_FSstr[NBFILTER]={ "-44k", "-48k", "-88k", "-96k" , "-176k", "-192k", "-352k" , "-384k", "-705k" , "-768k" };

static int inoutidx=0;
static snd_pcm_sframes_t dsp_transfer(conv_param_t *cvparam,snd_pcm_format_t fm, void *src,int *dst,snd_pcm_uframes_t size)
{
	long int n,nb;

	nb=cvparam->partsz-inoutidx;
	if(nb>size) nb=size;

	if(inoutidx==0) 
		waitConvolve();

	// copy  out
	for (n=0;n < nb*cvparam->nbch; n++) {
		dst[n]=(int)cvparam->inoutbuff[inoutidx*cvparam->nbch+n];
	}

	// copy in 
	for (n=0;n < nb*cvparam->nbch; n++) {
		     switch(fm) {
			case SND_PCM_FORMAT_S16 :
				cvparam->inoutbuff[inoutidx*cvparam->nbch+n]=(float)(((short*)src)[n]<<16);
				break;
			case SND_PCM_FORMAT_S32 :
				cvparam->inoutbuff[inoutidx*cvparam->nbch+n]=(float)((int*)src)[n];
				break;
			}
	}
	inoutidx+=nb;

	if(inoutidx>=cvparam->partsz) {
		cudaConvolve(cvparam);
		inoutidx=0;
	}

	return nb;
}

int main(int argc,char **argv)
{
	conv_param_t *cvparam;
	snd_config_t *sconf = NULL;
	int n;
	char *filterpath;

	int idx;
	int inbuff[20000];

    	int size,result;
	struct timespec start_time,stop_time;

	FILE *infd,*outfd;

        if(argc<4) {
		fprintf(stderr,"%s filter infile outfile\n",argv[0]);
		return 1;
	}	

	cvparam = calloc(1, sizeof(*cvparam));
	if (!cvparam)
		return -ENOMEM;

	cvparam->nf=2;
	cvparam->partsz=4096;
	cvparam->nbch=2;
	size=262144;

	cvparam->nbpart=(size+cvparam->partsz-1)/cvparam->partsz;

	initConvolve(cvparam);

	filterpath=malloc(strlen(argv[1])+10);
	for(n=0;n<NBFILTER;n++) {
		strcpy(filterpath,argv[1]);
		strcat(filterpath,filter_FSstr[n]);
		strcat(filterpath,".raw");
		readFilter(filterpath,cvparam,n);
	}

	infd=fopen(argv[2],"r");
	if(infd==NULL) return -1;
	outfd=fopen(argv[3],"w+");
	if(outfd==NULL) return -1;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID,&start_time);

	idx=0;
	while(!feof(infd)) {
		int outbuff[20000];
		int rn,wn;
		// just to simulate alsa call
		rn=fread(&(inbuff[idx*cvparam->nbch]),cvparam->nbch*sizeof(int),1065-idx,infd);
		wn=dsp_transfer(cvparam,SND_PCM_FORMAT_S32,inbuff,outbuff,1065);
		fwrite(outbuff,cvparam->nbch*sizeof(int),wn,outfd);
		idx=1065-wn;
		if(idx) memmove(inbuff,&(inbuff[wn*cvparam->nbch]),idx*cvparam->nbch*sizeof(int));
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID,&stop_time);

	fprintf(stderr,"computation time %ld\n",(stop_time.tv_sec-start_time.tv_sec)*1000L+(stop_time.tv_nsec-start_time.tv_nsec)/1000000L);
	fclose(infd);
	fclose(outfd);

	freeFilter();

	return 0;
}

