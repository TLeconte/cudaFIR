
#define NBFILTER 10

typedef struct {
  int nbch;
  int partsz;
  int nbpart;
  int nf;
  float *inoutbuff;
} conv_param_t;

extern int initConvolve(conv_param_t *cvparam);
extern int addFilter(conv_param_t *cvparam, float *h_filter, int nf);
extern int cudaConvolve(conv_param_t *cvparam);
extern void waitConvolve(void);
extern int readFilter(char *filterpath,conv_param_t *cvparam,int nf);
extern void freeFilter(void);

