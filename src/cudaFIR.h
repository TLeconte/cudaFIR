
#define NBFILTER 10

typedef struct {
  int nbch;
  int partsz;
  int nf;
  int nbpart[NBFILTER];
  float *inoutbuff;
} conv_param_t;

extern int initConvolve(conv_param_t *cvparam,char *filterpath);
extern int cudaConvolve(conv_param_t *cvparam);
extern void waitConvolve(void);
extern void freeFilter(void);

