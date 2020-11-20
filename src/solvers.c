#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "solvers.h"

#define szero myszero
#define saxpy mysaxpy
#define sdot mysdot
#define sscal mysscal
#define spositive myspositive
#define scopy myscopy
#define snrm2 mysnrm2
#define MEPS 1e-8

void szero(float*v, int l)
{
    memset(v, 0, sizeof(*v)*l);
}

void scopy(float *x, float*y, int l)
{
        memcpy(y, x, sizeof(*x)*(size_t)l);
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

float M4(int n, int d, float *A, float *h, float *V, float eps, int max_iter)
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

float M4_plus(int n, int d, int k, float *A, float *h, float *Z, float eps, int max_iter)
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
    free(g);
    free(avg);
    return diff;
}
