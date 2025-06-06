#ifndef STAT_UTIL_H
#define STAT_UTIL_H

#include <cmath>
#include <limits>

/**
 * Exact port of StatUtil from Java to C++
 *
 * This class contains the implementation of:
 * - Inverse Normal Cumulative Distribution Function Algorithm
 * - Error Function Algorithm
 * - Complementary Error Function Algorithm
 * - Normal Cumulative Distribution Function
 *
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/OpenSourceExtensions/StatUtil.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class StatUtil {
private:
    // Constants for getInvCDF
    static constexpr double P_LOW = 0.02425;
    static constexpr double P_HIGH = 1.0 - P_LOW;

    // Coefficients in rational approximations for getInvCDF
    static constexpr double ICDF_A[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    };

    static constexpr double ICDF_B[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    };

    static constexpr double ICDF_C[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    };

    static constexpr double ICDF_D[] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    };

    // Constants for normal_cdf
    static constexpr double NORM_CDF_a1 = 0.254829592;
    static constexpr double NORM_CDF_a2 = -0.284496736;
    static constexpr double NORM_CDF_a3 = 1.421413741;
    static constexpr double NORM_CDF_a4 = -1.453152027;
    static constexpr double NORM_CDF_a5 = 1.061405429;
    static constexpr double NORM_CDF_p = 0.3275911;

    // Constants and coefficients for erf, erfc, and erfcx
    static constexpr double ERF_A[] = {
        3.16112374387056560E00, 1.13864154151050156E02,
        3.77485237685302021E02, 3.20937758913846947E03,
        1.85777706184603153E-1
    };

    static constexpr double ERF_B[] = {
        2.36012909523441209E01, 2.44024637934444173E02,
        1.28261652607737228E03, 2.84423683343917062E03
    };

    static constexpr double ERF_C[] = {
        5.64188496988670089E-1, 8.88314979438837594E0,
        6.61191906371416295E01, 2.98635138197400131E02,
        8.81952221241769090E02, 1.71204761263407058E03,
        2.05107837782607147E03, 1.23033935479799725E03,
        2.15311535474403846E-8
    };

    static constexpr double ERF_D[] = {
        1.57449261107098347E01, 1.17693950891312499E02,
        5.37181101862009858E02, 1.62138957456669019E03,
        3.29079923573345963E03, 4.36261909014324716E03,
        3.43936767414372164E03, 1.23033935480374942E03
    };

    static constexpr double ERF_P[] = {
        3.05326634961232344E-1, 3.60344899949804439E-1,
        1.25781726111229246E-1, 1.60837851487422766E-2,
        6.58749161529837803E-4, 1.63153871373020978E-2
    };

    static constexpr double ERF_Q[] = {
        2.56852019228982242E00, 1.87295284992346047E00,
        5.27905102951428412E-1, 6.05183413124413191E-2,
        2.33520497626869185E-3
    };

    static constexpr double PI_SQRT = 1.77245385090551603; // sqrt(M_PI)
    static constexpr double THRESHOLD = 0.46875;

    // Hardware dependent constants
    static constexpr double X_MIN = std::numeric_limits<double>::min();
    static constexpr double X_INF = std::numeric_limits<double>::max();
    static constexpr double X_NEG = -9.38241396824444;
    static constexpr double X_SMALL = 1.110223024625156663E-16;
    static constexpr double X_BIG = 9.194E0;
    
    // These can't be constexpr because they use functions that aren't constexpr
    static const double X_HUGE;
    static const double X_MAX;

    /**
     * Helper function for erf, erfc, and erfcx
     */
    static double calerf(double X, int jint);

    /**
     * Refining algorithm for getInvCDF
     */
    static double refine(double x, double d);

public:
    /**
     * Get the inverse of the normal cumulative distribution function
     *
     * @param d The probability value (between 0 and 1)
     * @param highPrecision Whether to use high precision refinement
     * @return The value x such that normal_cdf(x) = d
     */
    static double getInvCDF(double d, bool highPrecision);

    /**
     * Calculate the cumulative density under a standard normal to a point of interest
     *
     * @param x The point of interest on the standard normal density support
     * @return The probability of interest
     */
    static double normal_cdf(double x);

    /**
     * Error function
     */
    static double erf(double d);

    /**
     * Complementary error function
     */
    static double erfc(double d);

    /**
     * Scaled complementary error function
     */
    static double erfcx(double d);
};

#endif // STAT_UTIL_H
