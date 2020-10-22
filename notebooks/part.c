#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum{AGGR_LOGZ, AGGR_GRAD};
#define BUF_SIZE 1024

inline void aggr_logZ(int n, char *x, double v, double *buf, int *cnt, double *logZ)
{
    buf[*cnt] = v;
    (*cnt)++;

    // Only do softmax aggregation when the buffer is full
    if (*cnt != BUF_SIZE && *cnt != 1<<n) return;

    double maxv = *logZ;
    for (int j=0; j<*cnt; j++)
        if (maxv < buf[j]) maxv = buf[j];
    double sum_exp = 0;
    for (int j=0; j<*cnt; j++) 
        sum_exp += exp(buf[j]-maxv);

    if (*logZ < maxv)
        *logZ = log(exp(*logZ-maxv) + sum_exp) + maxv;
    else
        *logZ = log1p(sum_exp) + maxv;
    *cnt = 0;
}

inline void aggr_grad(int n, char *x, double v, double *grad, int *cnt, double *logZ)
{
    double p = exp(v-*logZ);
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            grad[i*n+j] += p * ((x[i] == x[j]) * 2 - 1);
}

void dfs_expA(int n, int i, float *A, float *h, char *x, double v, int aggr_type, double *buf, int *cnt, double *logZ)
{
    if (i==-1) {
        if (aggr_type == AGGR_LOGZ) 
            aggr_logZ(n, x, v, buf, cnt, logZ);
        else
            aggr_grad(n, x, v, buf, cnt, logZ);
        return;
    }

    const float *Ai = A+i*n;

    for (int j=0; j<i; j++) h[j] += Ai[j];
    float vp = v + h[i] + Ai[i]/2;
    x[i] = 1;
    dfs_expA(n, i-1, A, h, x, vp, aggr_type, buf, cnt, logZ);

    for (int j=0; j<i; j++) h[j] -= 2*Ai[j];
    float vn = v - h[i] + Ai[i]/2;
    x[i] = 0;
    dfs_expA(n, i-1, A, h, x, vn, aggr_type, buf, cnt, logZ);

    for (int j=0; j<i; j++) h[j] += Ai[j];
}


double eval_logZ(int n, float *A, float *h)
{
    char *x = (char *) calloc(n, sizeof(*x));
    double *buf = (double *) calloc(BUF_SIZE, sizeof(*buf));
    double logZ = -INFINITY;
    int cnt = 0;
    dfs_expA(n, n-1, A, h, x, 0, AGGR_LOGZ, buf, &cnt, &logZ);
    free(buf);
    free(x);
    return logZ;
}

void eval_grad(int n, float *A, float *h, double *grad, double logZ)
{
    char *x = (char *) calloc(n, sizeof(*x));
    dfs_expA(n, n-1, A, h, x, 0, AGGR_GRAD, grad, NULL, &logZ);
    free(x);
}

#define szero myszero
#define saxpy mysaxpy
#define sdot mysdot
#define sscal mysscal
#define spositive myspositive
#define scopy myscopy
#define snrm2 mysnrm2

void szero(float*v, int l)
{
    memset(v, 0, sizeof(*v)*l);
}

void saxpy(float *restrict y, float a, const float *restrict x, int l)
{
        int m = l-3;
        int i;
        for (i = 0; i < m; i += 4)
        {
                y[i] += a * x[i];
                y[i+1] += a * x[i+1];
                y[i+2] += a * x[i+2];
                y[i+3] += a * x[i+3];
        }
        for ( ; i < l; ++i) /* clean-up loop */
                y[i] += a * x[i];
}

float sdot(const float *x, const float *y, int l)
{
        float s = 0;
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5)
                s += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] +
                        x[i+3] * y[i+3] + x[i+4] * y[i+4];

        for ( ; i < l; i++)        /* clean-up loop */
                s += x[i] * y[i];

        return s;
}

float snrm2(const float *x, int l)
{
        float xx = sdot(x, x, l);
        return sqrt(xx);
}

void sscal(float *x, float a, int l)
{
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5){
                x[i] *= a;
                x[i+1] *= a;
                x[i+2] *= a;
                x[i+3] *= a;
                x[i+4] *= a;
        }

        for ( ; i < l; i++)        /* clean-up loop */
                x[i] *= a;
}

void spositive(float *x, int l)
{
        int i;
        int m = l-4;
        for (i = 0; i < m; i += 5){
                x[i] *= (x[i]>=0);
                x[i+1] *= (x[i+1]>=0);
                x[i+2] *= (x[i+2]>=0);
                x[i+3] *= (x[i+3]>=0);
                x[i+4] *= (x[i+4]>=0);
        }
        for ( ; i < l; i++)        /* clean-up loop */
                x[i] *= (x[i]>=0);
}

void scopy(float *x, float*y, int l)
{
        memcpy(y, x, sizeof(*x)*(size_t)l);
}

float mixing(int n, int d, float *A, float *h, float *V, float eps, int max_iter)
{
    float *g = (float *) calloc(d, sizeof(*g));

    float diff = -1;
    for (int it=0; it<max_iter; it++) {
        diff = 0;
        for (int i=0; i<n; i++) {
            scopy(h+i*d, g, d);
            for (int j=0; j<n; j++)
                saxpy(g, A[i*n+j], V+j*d, d);
            float gnrm = snrm2(g, d);
            sscal(g, 1/gnrm, d);
            float gv = sdot(g, V+i*d, d);
            scopy(g, V+i*d, d);
            diff += gnrm * (1-gv);
        }
        fprintf(stderr, "it %d diff %f\n", it, diff);
        if (fabs(diff) < eps) break;
    }

    free(g);
    return diff;
}

void smaxvec(float *x, float *y, int m, int k)
{
    szero(y, m);
    for (int i=0; i<k; i++) {
        for (int j=i*m; j<(i+1)*m; j++) {
            if (y[j%m] < x[j]) y[j%m] = x[j];
            else x[j] = 0;
        }
    }
    for (int i=0; i<k; i++) {
        for (int j=i*m; j<(i+1)*m; j++) {
            if (y[j%m] > x[j]) x[j] = 0;
        }
    }
}

void shrink(float *x, float *y, int m, int k)
{
    for (int i=0; i<k; i++) {
        for (int j=i*m; j<(i+1)*m; j++) {
            if (x[j] > 0) y[j%m] += 1;
        }
    }

    //for (int j=0; j<m; j++)
    //    fprintf(stderr, "%f ", y[j]);
    //fprintf(stderr, "\n");

    for (int i=0; i<k; i++) {
        for (int j=i*m; j<(i+1)*m; j++) {
            if (y[j%m] > 0) x[j] /= y[j%m];
        }
    }
}

#define MEPS 1e-8

float mixing_plus(int n, int d, int k, float *A, float *h, float *Z, float eps, int max_iter)
{
    int m = d/k;
    float *g =   (float *) calloc(d, sizeof(*g));
    float *avg = (float *) calloc(m, sizeof(*avg));

    float diff = -1;
    for (int it=0; it<max_iter; it++) {
        diff = 0;
        for (int i=0; i<n; i++) {
            // g = \sum_j aij zj + hi
            scopy(h+i*d, g, d);
            for (int j=0; j<n; j++)
                saxpy(g, A[i*n+j], Z+j*d, d);
            szero(avg, m);

            // avg = average of k block
            for (int j=0; j<k; j++)
                saxpy(avg, 1., g+j*m, m);
            // g_blocks -= avg
            float invk = -1./k;
            for (int j=0; j<k; j++)
                saxpy(g+j*m, invk, avg, m);
            
            float gz = sdot(g, Z+i*d, d);
            spositive(g, d);
            smaxvec(g, avg, m, k);
            float gnrm = snrm2(avg, m);

            if (gnrm < MEPS) continue;

            sscal(g, 1/gnrm, d);
            //shrink(g, avg, m, k);
            scopy(g, Z+i*d, d);
            diff += gnrm - gz;
        }
        fprintf(stderr, "it %d diff %f\n", it, diff);
        if (fabs(diff) < eps) break;
    }

#if 0
    for (int i=0; i<n; i++) {
        for (int j=0; j<k; j++) {
            for (int l=j*m; l<(j+1)*m; l++) {
                fprintf(stderr, "%f ", Z[i*d+l]);
            }
            fprintf(stderr, "// ");
        }
        fprintf(stderr, "\n");
    }
#endif

    free(g);
    free(avg);
    return diff;
}