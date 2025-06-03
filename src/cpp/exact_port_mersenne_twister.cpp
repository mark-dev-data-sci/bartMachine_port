#include "include/exact_port_mersenne_twister.h"
#include <stdexcept>
#include <cstring>
#include <chrono>

/**
 * Exact port of MersenneTwisterFast from Java to C++
 * 
 * This file contains the basic class structure with constructors, destructor,
 * and empty method stubs. Method implementations will be added in subsequent tasks.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/OpenSourceExtensions/MersenneTwisterFast.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */

// Default constructor - matches Java: public MersenneTwisterFast()
ExactPortMersenneTwister::ExactPortMersenneTwister() {
    // Initialize arrays
    initializeArrays();
    
    // Seed with current time (matches Java behavior)
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    setSeed(static_cast<int64_t>(millis));
}

// Constructor with seed - matches Java: public MersenneTwisterFast(long seed)
ExactPortMersenneTwister::ExactPortMersenneTwister(int64_t seed) {
    initializeArrays();
    setSeed(seed);
}

// Constructor with array - matches Java: public MersenneTwisterFast(int[] array)
ExactPortMersenneTwister::ExactPortMersenneTwister(const std::vector<int>& array) {
    initializeArrays();
    setSeed(array);
}

// Destructor
ExactPortMersenneTwister::~ExactPortMersenneTwister() {
    cleanup();
}

// Copy constructor
ExactPortMersenneTwister::ExactPortMersenneTwister(const ExactPortMersenneTwister& other) {
    initializeArrays();
    copyFrom(other);
}

// Assignment operator
ExactPortMersenneTwister& ExactPortMersenneTwister::operator=(const ExactPortMersenneTwister& other) {
    if (this != &other) {
        copyFrom(other);
    }
    return *this;
}

// Helper method to initialize arrays
void ExactPortMersenneTwister::initializeArrays() {
    mt = new int[N];
    mag01 = new int[2];
    mag01[0] = 0x0;
    mag01[1] = MATRIX_A;
    mti = N + 1;  // Initialize as uninitialized state
    __haveNextNextGaussian = false;
    __nextNextGaussian = 0.0;
}

// Helper method to copy from another instance
void ExactPortMersenneTwister::copyFrom(const ExactPortMersenneTwister& other) {
    // Copy state arrays
    std::memcpy(mt, other.mt, N * sizeof(int));
    std::memcpy(mag01, other.mag01, 2 * sizeof(int));
    
    // Copy state variables
    mti = other.mti;
    __nextNextGaussian = other.__nextNextGaussian;
    __haveNextNextGaussian = other.__haveNextNextGaussian;
}

// Helper method to cleanup memory
void ExactPortMersenneTwister::cleanup() {
    delete[] mt;
    delete[] mag01;
    mt = nullptr;
    mag01 = nullptr;
}

// Clone method - matches Java: public Object clone()
ExactPortMersenneTwister* ExactPortMersenneTwister::clone() const {
    return new ExactPortMersenneTwister(*this);
}

// State comparison - matches Java: public boolean stateEquals(MersenneTwisterFast other)
bool ExactPortMersenneTwister::stateEquals(const ExactPortMersenneTwister& other) const {
    // TODO: Implement in Task 1.2+
    return false;
}

// Seed methods - matches Java setSeed methods
void ExactPortMersenneTwister::setSeed(int64_t seed) {
    // Due to a bug in java.util.Random clear up to 1.2, we're
    // doing our own Gaussian variable.
    __haveNextNextGaussian = false;

    // Note: In C++, we don't need to reallocate since arrays are already allocated
    // mt = new int[N];  // Already allocated in initializeArrays()
    
    // mag01 = new int[2];  // Already allocated in initializeArrays()
    mag01[0] = 0x0;
    mag01[1] = MATRIX_A;

    mt[0] = static_cast<int>(seed & 0xffffffff);
    for (mti = 1; mti < N; mti++) {
        // Use unsigned right shift to match Java's >>> operator
        unsigned int prev = static_cast<unsigned int>(mt[mti-1]);
        unsigned int shifted = prev >> 30;
        mt[mti] =
            (1812433253 * (mt[mti-1] ^ shifted) + mti);
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        // mt[mti] &= 0xffffffff;
        /* for >32 bit machines */
    }
}

void ExactPortMersenneTwister::setSeed(const std::vector<int>& array) {
    // Due to a bug in java.util.Random clear up to 1.2, we're
    // doing our own Gaussian variable.
    __haveNextNextGaussian = false;

    int i, j, k;
    setSeed(19650218);
    i = 1; j = 0;
    k = (N > static_cast<int>(array.size()) ? N : static_cast<int>(array.size()));
    
    for (; k != 0; k--) {
        // Use unsigned right shift to match Java's >>> operator
        unsigned int prev = static_cast<unsigned int>(mt[i-1]);
        unsigned int shifted = prev >> 30;
        mt[i] = (mt[i] ^ ((mt[i-1] ^ shifted) * 1664525)) + array[j] + j; /* non linear */
        // mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i >= N) { mt[0] = mt[N-1]; i = 1; }
        if (j >= static_cast<int>(array.size())) j = 0;
    }
    
    for (k = N-1; k != 0; k--) {
        // Use unsigned right shift to match Java's >>> operator
        unsigned int prev = static_cast<unsigned int>(mt[i-1]);
        unsigned int shifted = prev >> 30;
        mt[i] = (mt[i] ^ ((mt[i-1] ^ shifted) * 1566083941)) - i; /* non linear */
        // mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= N) { mt[0] = mt[N-1]; i = 1; }
    }

    mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */ 
}

// State I/O methods
void ExactPortMersenneTwister::readState(std::istream& stream) {
    // TODO: Implement in later tasks
}

void ExactPortMersenneTwister::writeState(std::ostream& stream) const {
    // TODO: Implement in later tasks
}

// Core random number generation methods
int ExactPortMersenneTwister::nextInt() {
    // TODO: Implement in later tasks
    return 0;
}

int ExactPortMersenneTwister::nextInt(int n) {
    // TODO: Implement in later tasks
    return 0;
}

int64_t ExactPortMersenneTwister::nextLong() {
    // TODO: Implement in later tasks
    return 0;
}

int64_t ExactPortMersenneTwister::nextLong(int64_t n) {
    // TODO: Implement in later tasks
    return 0;
}

// Type-specific methods
int16_t ExactPortMersenneTwister::nextShort() {
    // TODO: Implement in later tasks
    return 0;
}

char ExactPortMersenneTwister::nextChar() {
    // TODO: Implement in later tasks
    return 0;
}

bool ExactPortMersenneTwister::nextBoolean() {
    int y;

    if (mti >= N) {  // generate N words at one time
        int kk;
        
        for (kk = 0; kk < N - M; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_y = static_cast<unsigned int>(y);
            unsigned int shifted = temp_y >> 1;
            mt[kk] = mt[kk+M] ^ shifted ^ mag01[y & 0x1];
        }
        
        for (; kk < N-1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_y = static_cast<unsigned int>(y);
            unsigned int shifted = temp_y >> 1;
            mt[kk] = mt[kk+(M-N)] ^ shifted ^ mag01[y & 0x1];
        }
        
        y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        // Use unsigned right shift to match Java's >>> operator
        unsigned int temp_y = static_cast<unsigned int>(y);
        unsigned int shifted = temp_y >> 1;
        mt[N-1] = mt[M-1] ^ shifted ^ mag01[y & 0x1];

        mti = 0;
    }
  
    y = mt[mti++];
    // Use unsigned right shift to match Java's >>> operator
    unsigned int temp_y = static_cast<unsigned int>(y);
    y ^= temp_y >> 11;                     // TEMPERING_SHIFT_U(y)
    y ^= (y << 7) & TEMPERING_MASK_B;      // TEMPERING_SHIFT_S(y)
    y ^= (y << 15) & TEMPERING_MASK_C;     // TEMPERING_SHIFT_T(y)
    temp_y = static_cast<unsigned int>(y);
    y ^= temp_y >> 18;                     // TEMPERING_SHIFT_L(y)

    // Use unsigned right shift to match Java's >>> operator
    temp_y = static_cast<unsigned int>(y);
    return ((temp_y >> 31) != 0);
}

bool ExactPortMersenneTwister::nextBoolean(float probability) {
    // TODO: Implement in later tasks
    return false;
}

bool ExactPortMersenneTwister::nextBoolean(double probability) {
    // TODO: Implement in later tasks
    return false;
}

int8_t ExactPortMersenneTwister::nextByte() {
    // TODO: Implement in later tasks
    return 0;
}

void ExactPortMersenneTwister::nextBytes(std::vector<int8_t>& bytes) {
    // TODO: Implement in later tasks
}

// Floating point methods
float ExactPortMersenneTwister::nextFloat() {
    // TODO: Implement in later tasks
    return 0.0f;
}

float ExactPortMersenneTwister::nextFloat(bool includeZero, bool includeOne) {
    // TODO: Implement in later tasks
    return 0.0f;
}

double ExactPortMersenneTwister::nextDouble() {
    int y;
    int z;

    if (mti >= N) {  // generate N words at one time
        int kk;
        
        for (kk = 0; kk < N - M; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_y = static_cast<unsigned int>(y);
            unsigned int shifted = temp_y >> 1;
            mt[kk] = mt[kk+M] ^ shifted ^ mag01[y & 0x1];
        }
        
        for (; kk < N-1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_y = static_cast<unsigned int>(y);
            unsigned int shifted = temp_y >> 1;
            mt[kk] = mt[kk+(M-N)] ^ shifted ^ mag01[y & 0x1];
        }
        
        y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        // Use unsigned right shift to match Java's >>> operator
        unsigned int temp_y = static_cast<unsigned int>(y);
        unsigned int shifted = temp_y >> 1;
        mt[N-1] = mt[M-1] ^ shifted ^ mag01[y & 0x1];

        mti = 0;
    }
  
    y = mt[mti++];
    // Use unsigned right shift to match Java's >>> operator
    unsigned int temp_y = static_cast<unsigned int>(y);
    y ^= temp_y >> 11;                     // TEMPERING_SHIFT_U(y)
    y ^= (y << 7) & TEMPERING_MASK_B;      // TEMPERING_SHIFT_S(y)
    y ^= (y << 15) & TEMPERING_MASK_C;     // TEMPERING_SHIFT_T(y)
    temp_y = static_cast<unsigned int>(y);
    y ^= temp_y >> 18;                     // TEMPERING_SHIFT_L(y)

    if (mti >= N) {  // generate N words at one time
        int kk;
        
        for (kk = 0; kk < N - M; kk++) {
            z = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_z = static_cast<unsigned int>(z);
            unsigned int shifted = temp_z >> 1;
            mt[kk] = mt[kk+M] ^ shifted ^ mag01[z & 0x1];
        }
        
        for (; kk < N-1; kk++) {
            z = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
            // Use unsigned right shift to match Java's >>> operator
            unsigned int temp_z = static_cast<unsigned int>(z);
            unsigned int shifted = temp_z >> 1;
            mt[kk] = mt[kk+(M-N)] ^ shifted ^ mag01[z & 0x1];
        }
        
        z = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        // Use unsigned right shift to match Java's >>> operator
        unsigned int temp_z = static_cast<unsigned int>(z);
        unsigned int shifted = temp_z >> 1;
        mt[N-1] = mt[M-1] ^ shifted ^ mag01[z & 0x1];

        mti = 0;
    }
  
    z = mt[mti++];
    // Use unsigned right shift to match Java's >>> operator
    unsigned int temp_z = static_cast<unsigned int>(z);
    z ^= temp_z >> 11;                     // TEMPERING_SHIFT_U(z)
    z ^= (z << 7) & TEMPERING_MASK_B;      // TEMPERING_SHIFT_S(z)
    z ^= (z << 15) & TEMPERING_MASK_C;     // TEMPERING_SHIFT_T(z)
    temp_z = static_cast<unsigned int>(z);
    z ^= temp_z >> 18;                     // TEMPERING_SHIFT_L(z)

    /* derived from nextDouble documentation in jdk 1.2 docs, see top */
    // Use unsigned right shift to match Java's >>> operator
    temp_y = static_cast<unsigned int>(y);
    temp_z = static_cast<unsigned int>(z);
    return ((((int64_t)(temp_y >> 6)) << 27) + (temp_z >> 5)) / (double)(INT64_C(1) << 53);
}

double ExactPortMersenneTwister::nextDouble(bool includeZero, bool includeOne) {
    double d = 0.0;
    do {
        d = nextDouble();                           // grab a value, initially from half-open [0.0, 1.0)
        if (includeOne && nextBoolean()) d += 1.0;  // if includeOne, with 1/2 probability, push to [1.0, 2.0)
    } while ((d > 1.0) ||                           // everything above 1.0 is always invalid
             (!includeZero && d == 0.0));           // if we're not including zero, 0.0 is invalid
    return d;
}

// Gaussian methods
double ExactPortMersenneTwister::nextGaussian() {
    // TODO: Implement in later tasks
    return 0.0;
}

void ExactPortMersenneTwister::clearGaussian() {
    __haveNextNextGaussian = false;
}

// Test method - matches Java: public static void main(String args[])
void ExactPortMersenneTwister::main(const std::vector<std::string>& args) {
    // TODO: Implement test method in later tasks
}
