
#define NBFILTER 10

typedef struct {
  int nbch;
  int partsz;
  int nbpart;
  float *inoutbuff[2];
  int nbf;
  char *filterpathprefix;
} conv_param_t;

extern int initConvolve(conv_param_t *cvparam);
extern int readFilter(char *filterpath,conv_param_t *cvparam);
extern void freeFilter(conv_param_t *cvparam);
extern void waitConvolve(void);
extern int cudaConvolve(conv_param_t *cvparam);

