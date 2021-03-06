
#define NBFILTER 10

typedef struct {
  int nbch;
  int partsz;
  int nf;
  int nbpart[NBFILTER];
  int nbf;
  float *inoutbuff[2];
} conv_param_t;

extern int initConvolve(conv_param_t *cvparam,char *filterpath);
extern void resetConvolve(conv_param_t *cvparam);
extern void waitConvolve(void);
extern int cudaConvolve(conv_param_t *cvparam);
extern void freeConvolve(void);

