#include "include/stat_util.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

// Define the static constants
const double StatUtil::X_HUGE = 1.0 / (2.0 * std::sqrt(StatUtil::X_SMALL));
const double StatUtil::X_MAX = std::min(std::numeric_limits<double>::max(), (1.0 / (StatUtil::PI_SQRT * StatUtil::X_MIN)));

double StatUtil::getInvCDF(double d, bool highPrecision) {
    // Kludge from Java implementation
    if (d == 0) {
        d = d + 1e-14;
    }
    if (d == 1) {
        d = d - 1e-14;
    }

    // Define break-points
    // variable for result
    double z = 0;

    if (d == 0) z = -std::numeric_limits<double>::infinity();
    else if (d == 1) z = std::numeric_limits<double>::infinity();
    else if (std::isnan(d) || d < 0 || d > 1) z = std::numeric_limits<double>::quiet_NaN();

    // Rational approximation for lower region:
    else if (d < P_LOW) {
        double q = std::sqrt(-2 * std::log(d));
        z = (((((ICDF_C[0] * q + ICDF_C[1]) * q + ICDF_C[2]) * q + ICDF_C[3]) * q + ICDF_C[4]) * q + ICDF_C[5]) /
            ((((ICDF_D[0] * q + ICDF_D[1]) * q + ICDF_D[2]) * q + ICDF_D[3]) * q + 1);
    }

    // Rational approximation for upper region:
    else if (P_HIGH < d) {
        double q = std::sqrt(-2 * std::log(1 - d));
        z = -(((((ICDF_C[0] * q + ICDF_C[1]) * q + ICDF_C[2]) * q + ICDF_C[3]) * q + ICDF_C[4]) * q + ICDF_C[5]) /
             ((((ICDF_D[0] * q + ICDF_D[1]) * q + ICDF_D[2]) * q + ICDF_D[3]) * q + 1);
    }
    // Rational approximation for central region:
    else {
        double q = d - 0.5;
        double r = q * q;
        z = (((((ICDF_A[0] * r + ICDF_A[1]) * r + ICDF_A[2]) * r + ICDF_A[3]) * r + ICDF_A[4]) * r + ICDF_A[5]) * q /
            (((((ICDF_B[0] * r + ICDF_B[1]) * r + ICDF_B[2]) * r + ICDF_B[3]) * r + ICDF_B[4]) * r + 1);
    }
    if (highPrecision) z = refine(z, d);
    if (std::isinf(z) || std::isnan(z)) {
        std::cerr << "getInvCDF(" << d << ") is infinite or NaN" << std::endl;
    }
    return z;
}

double StatUtil::normal_cdf(double x) {
    // Save the sign of x
    int sign = 1;
    if (x < 0) {
        sign = -1;
    }
    x = std::abs(x) / std::sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + NORM_CDF_p * x);
    double y = 1.0 - (((((NORM_CDF_a5 * t + NORM_CDF_a4) * t) + NORM_CDF_a3) * t + NORM_CDF_a2) * t + NORM_CDF_a1) * t * std::exp(-x * x);

    double p = 0.5 * (1.0 + sign * y);

    // Kludge from Java implementation
    if (p == 0) {
        p = p + 1e-14;
    }
    if (p == 1) {
        p = p - 1e-14;
    }
    return p;
}

double StatUtil::calerf(double X, int jint) {
    double result = 0;
    double Y = std::abs(X);
    double Y_SQ, X_NUM, X_DEN;

    if (Y <= THRESHOLD) {
        Y_SQ = 0.0;
        if (Y > X_SMALL) Y_SQ = Y * Y;
        X_NUM = ERF_A[4] * Y_SQ;
        X_DEN = Y_SQ;
        for (int i = 0; i < 3; i++) {
            X_NUM = (X_NUM + ERF_A[i]) * Y_SQ;
            X_DEN = (X_DEN + ERF_B[i]) * Y_SQ;
        }
        result = X * (X_NUM + ERF_A[3]) / (X_DEN + ERF_B[3]);
        if (jint != 0) result = 1 - result;
        if (jint == 2) result = std::exp(Y_SQ) * result;
        return result;
    }
    else if (Y <= 4.0) {
        X_NUM = ERF_C[8] * Y;
        X_DEN = Y;
        for (int i = 0; i < 7; i++) {
            X_NUM = (X_NUM + ERF_C[i]) * Y;
            X_DEN = (X_DEN + ERF_D[i]) * Y;
        }
        result = (X_NUM + ERF_C[7]) / (X_DEN + ERF_D[7]);
        if (jint != 2) {
            Y_SQ = std::round(Y * 16.0) / 16.0;
            double del = (Y - Y_SQ) * (Y + Y_SQ);
            result = std::exp(-Y_SQ * Y_SQ) * std::exp(-del) * result;
        }
    }
    else {
        result = 0.0;
        if (Y >= X_BIG && (jint != 2 || Y >= X_MAX));
        else if (Y >= X_BIG && Y >= X_HUGE) result = PI_SQRT / Y;
        else {
            Y_SQ = 1.0 / (Y * Y);
            X_NUM = ERF_P[5] * Y_SQ;
            X_DEN = Y_SQ;
            for (int i = 0; i < 4; i++) {
                X_NUM = (X_NUM + ERF_P[i]) * Y_SQ;
                X_DEN = (X_DEN + ERF_Q[i]) * Y_SQ;
            }
            result = Y_SQ * (X_NUM + ERF_P[4]) / (X_DEN + ERF_Q[4]);
            result = (PI_SQRT - result) / Y;
            if (jint != 2) {
                Y_SQ = std::round(Y * 16.0) / 16.0;
                double del = (Y - Y_SQ) * (Y + Y_SQ);
                result = std::exp(-Y_SQ * Y_SQ) * std::exp(-del) * result;
            }
        }
    }

    if (jint == 0) {
        result = (0.5 - result) + 0.5;
        if (X < 0) result = -result;
    }
    else if (jint == 1) {
        if (X < 0) result = 2.0 - result;
    }
    else {
        if (X < 0) {
            if (X < X_NEG) result = std::numeric_limits<double>::max();
            else {
                Y_SQ = std::round(X * 16.0) / 16.0;
                double del = (X - Y_SQ) * (X + Y_SQ);
                Y = std::exp(Y_SQ * Y_SQ) * std::exp(del);
                result = (Y + Y) - result;
            }
        }
    }
    return result;
}

double StatUtil::erf(double d) {
    return calerf(d, 0);
}

double StatUtil::erfc(double d) {
    return calerf(d, 1);
}

double StatUtil::erfcx(double d) {
    return calerf(d, 2);
}

double StatUtil::refine(double x, double d) {
    if (d > 0 && d < 1) {
        double e = 0.5 * erfc(-x / std::sqrt(2.0)) - d;
        double u = e * std::sqrt(2.0 * M_PI) * std::exp((x * x) / 2.0);
        x = x - u / (1.0 + x * u / 2.0);
    }
    return x;
}
