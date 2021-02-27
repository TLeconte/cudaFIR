#include <alsa/asoundlib.h>
#include <alsa/pcm_external.h>
#include "cudaFIR.h"

typedef struct {
  snd_pcm_extplug_t ext;
  conv_param_t cuparam;
  int inoutidx;
} snd_pcm_cudaFIR_t;


static int filter_FS[NBFILTER]={ 44100, 48000, 88200, 96000 , 176400, 192000, 352800 ,384000, 705800 , 768000 };
static char *filter_FSstr[NBFILTER]={ "-44k", "-48k", "-88k", "-96k" , "-176k", "-192k", "-352k" , "-384k", "-705k" , "-768k" };

static inline void *area_addr(const snd_pcm_channel_area_t *area, snd_pcm_uframes_t offset) {
	unsigned int bitofs = area->first + area->step * offset;
	return (char *) area->addr + bitofs / 8;
}

static snd_pcm_sframes_t
cudaFIR_transfert(snd_pcm_extplug_t *ext,
	     const snd_pcm_channel_area_t *dst_areas,
	     snd_pcm_uframes_t dst_offset,
	     const snd_pcm_channel_area_t *src_areas,
	     snd_pcm_uframes_t src_offset,
	     snd_pcm_uframes_t size)
{
	snd_pcm_cudaFIR_t *cu_snd = ext->private_data;
	conv_param_t *cvparam = &(cu_snd->cuparam);
	int *src = area_addr(src_areas, src_offset);
	int *dst = area_addr(dst_areas, dst_offset);
        long int n,nb,nc;

        nb=cvparam->partsz-cu_snd->inoutidx;
        if(nb>size) nb=size;

        if(cu_snd->inoutidx==0)
             waitConvolve();

        // copy  out
        for (n=0;n < nb*cvparam->nbch; n++) {
                dst[n]=(int)cvparam->inoutbuff[cu_snd->inoutidx*cvparam->nbch+n];
        }
        // copy in
        for (n=0;n < nb*cvparam->nbch; n++) {
                     switch(ext->format) {
                        case SND_PCM_FORMAT_S16 :
                                cvparam->inoutbuff[cu_snd->inoutidx*cvparam->nbch+n]=(float)(((short*)src)[n]<<16);
                                break;
                        case SND_PCM_FORMAT_S32 :
                                cvparam->inoutbuff[cu_snd->inoutidx*cvparam->nbch+n]=(float)((int*)src)[n];
                                break;
                        }
        }

        cu_snd->inoutidx+=nb;
        if(cu_snd->inoutidx>=cvparam->partsz) {
                cudaConvolve(cvparam);
                cu_snd->inoutidx=0;
        }

        return nb;
}

static int cudaFIR_init(snd_pcm_extplug_t *ext)
{
	int n;
	snd_pcm_cudaFIR_t *cu_snd = ext->private_data;

        for(n=0;n<NBFILTER;n++) {
        	if(ext->rate==filter_FS[n]) {
       			cu_snd->cuparam.nf=n;
			break;
		}
	}
	if(n==NBFILTER) {
		SNDERR("Invalid sampling rate %d",ext->rate);
		return -1;
	}

	return 0;
}

static const snd_pcm_extplug_callback_t cudaconvolve_callback = {
	.transfer = cudaFIR_transfert,
	.init = cudaFIR_init,
};

SND_PCM_PLUGIN_DEFINE_FUNC(cudaconvolve)
{
	snd_config_iterator_t i, next;
	snd_pcm_cudaFIR_t *cu_snd;
	conv_param_t *cvparam;
	snd_config_t *sconf = NULL;
	int err;
        const int format_list[] =  { SND_PCM_FORMAT_S16 , SND_PCM_FORMAT_S32 };

    	int size,result;
	char *filterpath;
	char *filterpathprefix;

	cu_snd = calloc(1, sizeof(snd_pcm_cudaFIR_t));
	if (!cu_snd)
		return -ENOMEM;

	cu_snd->ext.version = SND_PCM_EXTPLUG_VERSION;
	cu_snd->ext.name = "Cudaconvolve Plugin";
	cu_snd->ext.callback = &cudaconvolve_callback;
	cu_snd->ext.private_data = cu_snd;

	cu_snd->cuparam.partsz=4096;
	cu_snd->cuparam.nbch=2;

	cvparam = &(cu_snd->cuparam);

	snd_config_for_each(i, next, conf) {
		snd_config_t *n = snd_config_iterator_entry(i);
		const char *id;
		if (snd_config_get_id(n, &id) < 0)
			continue;
		if (strcmp(id, "comment") == 0 || strcmp(id, "type") == 0 ||
		    strcmp(id, "hint") == 0)
			continue;
		if (strcmp(id, "slave") == 0) {
			sconf = n;
			continue;
		}
		if (strcmp(id, "filterpathprefix") == 0) {
			if(snd_config_get_string(n,(const char **)&filterpathprefix)==0) 
				continue;
			SNDERR("Invalid cudaFIR filterpathprefix");
		}
		if (strcmp(id, "channels") == 0) {
			if(snd_config_get_integer(n,(long*)&(cu_snd->cuparam.nbch))==0) 
				continue;
			SNDERR("Invalid cudaFIR channels");
		}
		if (strcmp(id, "partsz") == 0) {
			if(snd_config_get_integer(n,(long*)&(cu_snd->cuparam.partsz))==0) 
				continue;
			SNDERR("Invalid cudaFIR partsz");
		}

		SNDERR("Unknown field %s", id);
		err = -EINVAL;
	ok:
		if (err < 0)
			return err;
	}

	if (!sconf) {
		SNDERR("No slave configuration for cudaconvolve ");
		return -EINVAL;
	}

	if (!filterpath) {
		SNDERR("No filter path for cudaconvolve ");
		return -EINVAL;
	}


        initConvolve(cvparam);

        filterpath=malloc(strlen(filterpathprefix)+10);
        for(int n=0;n<NBFILTER;n++) {
                strcpy(filterpath,filterpathprefix);
                strcat(filterpath,filter_FSstr[n]);
                strcat(filterpath,".raw");
                readFilter(filterpath,cvparam,n);
        }
	free(filterpath);

	err = snd_pcm_extplug_create(&cu_snd->ext, name, root, sconf, stream, mode);
	if (err < 0) {
		free(cu_snd);
		return err;
	}

   // set in/out formats
	snd_pcm_extplug_set_param(&cu_snd->ext, SND_PCM_EXTPLUG_HW_CHANNELS, cu_snd->cuparam.nbch);
        snd_pcm_extplug_set_param_list(&cu_snd->ext, SND_PCM_EXTPLUG_HW_FORMAT,2,format_list);

	snd_pcm_extplug_set_slave_param(&cu_snd->ext,SND_PCM_EXTPLUG_HW_CHANNELS, cu_snd->cuparam.nbch);
        snd_pcm_extplug_set_slave_param(&cu_snd->ext, SND_PCM_EXTPLUG_HW_FORMAT,SND_PCM_FORMAT_S32);

	*pcmp = cu_snd->ext.pcm;
	return 0;
}

SND_PCM_PLUGIN_SYMBOL(cudaconvolve);
