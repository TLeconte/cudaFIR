#include <alsa/asoundlib.h>
#include <alsa/pcm_external.h>
#include "cudaFIR.h"

static const int filter_FS[NBFILTER]={ 44100, 48000, 88200, 96000 , 176400, 192000, 352800 ,384000, 705800 , 768000 };
static const char *filter_FSstr[NBFILTER]={ "-44k", "-48k", "-88k", "-96k" , "-176k", "-192k", "-352k" , "-384k", "-705k" , "-768k" };


static int inoutidx=0;
static snd_pcm_sframes_t cudaFIR_transfer(conv_param_t *cvparam,snd_pcm_format_t fm, void *src,int *dst,snd_pcm_uframes_t size)
{
	long int n,nb;

	nb=cvparam->partsz-inoutidx;
	if(nb>size) nb=size;

	// copy  out
	for (n=0;n < nb*cvparam->nbch; n++) {
		dst[n]=(int)cvparam->inoutbuff[cvparam->nbf][inoutidx*cvparam->nbch+n];
	}

	// copy in 
	for (n=0;n < nb*cvparam->nbch; n++) {
		     switch(fm) {
			case SND_PCM_FORMAT_S16 :
				cvparam->inoutbuff[cvparam->nbf][inoutidx*cvparam->nbch+n]=(float)(((short*)src)[n]<<16);
				break;
			case SND_PCM_FORMAT_S32 :
				cvparam->inoutbuff[cvparam->nbf][inoutidx*cvparam->nbch+n]=(float)((int*)src)[n];
				break;
			}
	}
	inoutidx+=nb;

	if(inoutidx>=cvparam->partsz) {
		waitConvolve();
		cvparam->nbf^=1;
		cudaConvolve(cvparam);
		inoutidx=0;
	}

	return nb;
}

int cudaFIR_init(conv_param_t *cvparam,int fs)
{
        char *filterpath;
        int n;

        fprintf(stderr,"cudaFIR : sampling rate %d\n",fs);

        for(n=0;n<NBFILTER;n++) {
                if(fs==filter_FS[n]) {
                        break;
                }
        }
        if(n==NBFILTER) {
                fprintf(stderr,"Invalid sampling rate %d\n",fs);
                return -1;
        }

        filterpath=(char*)malloc(strlen(cvparam->filterpathprefix)+16);
        strcpy(filterpath,cvparam->filterpathprefix);
        strcat(filterpath,filter_FSstr[n]);
        strcat(filterpath,".raw");

        readFilter(filterpath,cvparam);
        free(filterpath);

        return 0;
}

int main(int argc,char **argv)
{
	conv_param_t *cvparam;
	snd_config_t *sconf = NULL;
	int n;

	int idx;
	int inbuff[20000];

    	int size,result;
	struct timespec start_time,stop_time;

	FILE *infd,*outfd;

        if(argc<5) {
		fprintf(stderr,"%s filter filternb infile outfile\n",argv[0]);
		return 1;
	}	

	cvparam = calloc(1, sizeof(*cvparam));
	if (!cvparam)
		return -ENOMEM;

	cvparam->partsz=4096;
	cvparam->nbch=2;
	cvparam->filterpathprefix=strdup(argv[1]);

        initConvolve(cvparam);
        initConvolve(cvparam);
	cudaFIR_init(cvparam,atoi(argv[2]));

	infd=fopen(argv[3],"r");
	if(infd==NULL) return -1;
	outfd=fopen(argv[4],"w+");
	if(outfd==NULL) return -1;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID,&start_time);

	idx=0;
	while(!feof(infd)) {
		int outbuff[20000];
		int rn,wn;
		// just to simulate alsa call
		rn=fread(&(inbuff[idx*cvparam->nbch]),cvparam->nbch*sizeof(int),1065-idx,infd);
		wn=cudaFIR_transfer(cvparam,SND_PCM_FORMAT_S32,inbuff,outbuff,1065);
		fwrite(outbuff,cvparam->nbch*sizeof(int),wn,outfd);
		idx=1065-wn;
		if(idx) memmove(inbuff,&(inbuff[wn*cvparam->nbch]),idx*cvparam->nbch*sizeof(int));
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID,&stop_time);

	fprintf(stderr,"computation time %ld\n",(stop_time.tv_sec-start_time.tv_sec)*1000L+(stop_time.tv_nsec-start_time.tv_nsec)/1000000L);
	fclose(infd);
	fclose(outfd);

	return 0;
}

