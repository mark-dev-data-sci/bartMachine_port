#ifndef EXACT_PORT_MERSENNE_TWISTER_H
#define EXACT_PORT_MERSENNE_TWISTER_H

#include <cstdint>
#include <vector>
#include <iostream>

/**
 * Exact port of MersenneTwisterFast from Java to C++
 * 
 * This is a direct port of the Java MersenneTwisterFast class from bartMachine.
 * All member variables, constants, and method signatures are preserved exactly
 * to ensure numerical equivalence.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/OpenSourceExtensions/MersenneTwisterFast.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 * Version 22, based on version MT199937(99/10/29)
 */
class ExactPortMersenneTwister {
private:
    // Period parameters - exact copies from Java
    static const int N = 624;
    static const int M = 397;
    static const int MATRIX_A = 0x9908b0df;      // constant vector a
    static const int UPPER_MASK = 0x80000000;    // most significant w-r bits
    static const int LOWER_MASK = 0x7fffffff;    // least significant r bits

    // Tempering parameters - exact copies from Java
    static const int TEMPERING_MASK_B = 0x9d2c5680;
    static const int TEMPERING_MASK_C = 0xefc60000;

    // State variables - exact copies from Java
    int* mt;                    // the array for the state vector
    int mti;                    // mti==N+1 means mt[N] is not initialized
    int* mag01;                 // array for mag01 values

    // Gaussian state variables - exact copies from Java
    double __nextNextGaussian;
    bool __haveNextNextGaussian;

public:
    // Constructors - exact signatures from Java
    ExactPortMersenneTwister();                    // Default constructor
    ExactPortMersenneTwister(int64_t seed);        // Constructor with long seed
    ExactPortMersenneTwister(const std::vector<int>& array);  // Constructor with int array

    // Destructor
    ~ExactPortMersenneTwister();

    // Copy constructor and assignment operator
    ExactPortMersenneTwister(const ExactPortMersenneTwister& other);
    ExactPortMersenneTwister& operator=(const ExactPortMersenneTwister& other);

    // Clone method - exact signature from Java
    ExactPortMersenneTwister* clone() const;

    // State comparison - exact signature from Java
    bool stateEquals(const ExactPortMersenneTwister& other) const;

    // Seed methods - exact signatures from Java
    void setSeed(int64_t seed);
    void setSeed(const std::vector<int>& array);

    // State I/O methods - exact signatures from Java
    void readState(std::istream& stream);
    void writeState(std::ostream& stream) const;

    // Core random number generation methods - exact signatures from Java
    int nextInt();
    int nextInt(int n);
    int64_t nextLong();
    int64_t nextLong(int64_t n);
    
    // Type-specific methods - exact signatures from Java
    int16_t nextShort();
    char nextChar();
    bool nextBoolean();
    bool nextBoolean(float probability);
    bool nextBoolean(double probability);
    int8_t nextByte();
    void nextBytes(std::vector<int8_t>& bytes);

    // Floating point methods - exact signatures from Java
    float nextFloat();
    float nextFloat(bool includeZero, bool includeOne);
    double nextDouble();
    double nextDouble(bool includeZero, bool includeOne);

    // Gaussian methods - exact signatures from Java
    double nextGaussian();
    void clearGaussian();

    // Test method - exact signature from Java
    static void main(const std::vector<std::string>& args);

private:
    // Helper methods for internal state management
    void initializeArrays();
    void copyFrom(const ExactPortMersenneTwister& other);
    void cleanup();
};

#endif // EXACT_PORT_MERSENNE_TWISTER_H
