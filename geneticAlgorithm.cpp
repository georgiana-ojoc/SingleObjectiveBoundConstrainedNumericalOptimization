#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#define ALGORITHM                   "EPRPSO"
#define DIMENSIONS                  10
#define E                           2.7182818284590452353602874713526625
#define EVALUATIONS                 200000
#define FUNCTIONS                   12
#define INF                         1.0e99
#define LEFT                        -100.0
#define PI                          3.1415926535897932384626433832795029
#define POPULATION                  100
#define PARAMETERS                  14
#define RANDOM_SEEDS                1000
#define REGENERATION_ERROR          0.1
#define RIGHT                       100.0
#define RUNS                        3
#define STOP_ERROR                  10E-8

using namespace std;
using namespace std::chrono;

double crossover = 0.3;
double mutation = 0.01;
const unsigned char cuts = 5;
const unsigned char tournament = 20;

constexpr unsigned char bits = 8;
constexpr unsigned short generations = 20;
constexpr unsigned short individuals = 20;
constexpr unsigned short runs = 1;

unsigned char naturalRandom(unsigned char size) {
    return rand() % size;
}

double realRandom01() {
    return (double) rand() / RAND_MAX;
}

double random(const double left, const double right) {
    return realRandom01() * (right - left) + left;
}

bool minimum(const double first, const double second) {
    return first < second;
}

class Function {
    double shiftData[DIMENSIONS * 10];
    double rotationData[DIMENSIONS * DIMENSIONS * 10];
    unsigned int evaluations;

    void loadMatrix() {
        char fileName[26];
        sprintf(fileName, "../input_data/M_%d_D%d.txt", number, dimensions);
        FILE *file = fopen(fileName, "r");
        unsigned short values = dimensions * dimensions;
        if (number > 8) {
            values *= 10;
        }
        for (unsigned short i = 0; i < values; i++) {
            fscanf(file, "%lf", &rotationData[i]);
        }
        fclose(file);
    }

    void loadShiftData() {
        char fileName[32];
        sprintf(fileName, "../input_data/shift_data_%d.txt", number);
        FILE *file = fopen(fileName, "r");
        if (number < 9) {
            for (unsigned char i = 0; i < dimensions; i++) {
                fscanf(file, "%lf", &shiftData[i]);
            }
        } else {
            for (unsigned char i = 0; i < 9; i++) {
                for (unsigned char j = 0; j < dimensions; j++) {
                    fscanf(file, "%lf", &shiftData[i * dimensions + j]);
                }
                fscanf(file, "%*[^\n]%*c");
            }
            for (unsigned char j = 0; j < dimensions; j++) {
                fscanf(file, "%lf", &shiftData[9 * dimensions + j]);
            }
        }
        fclose(file);
    }

    void loadShuffleData() {
        char fileName[37];
        sprintf(fileName, "../input_data/shuffle_data_%d_D%d.txt", number, dimensions);
        FILE *file = fopen(fileName, "r");
        for (unsigned char i = 0; i < dimensions; i++) {
            fscanf(file, "%d", &shuffleData[i]);
        }
        fclose(file);
    }

protected:
    const unsigned char number{};
    const unsigned char dimensions{};
    const char *name{};
    const double minimum{};

    unsigned char shuffleData[DIMENSIONS];
    double firstCopiedX[DIMENSIONS];
    double secondCopiedX[DIMENSIONS];
    double thirdCopiedX[DIMENSIONS];

    void shift(double x[], unsigned char start = 0) const {
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] -= shiftData[start + i];
        }
    }

    void shrink(double x[], double shrinkage) const {
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] *= shrinkage;
        }
    }

    void rotate(double x[], unsigned start = 0) {
        memcpy(firstCopiedX, x, dimensions * sizeof(double));
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] = 0;
            for (unsigned char j = 0; j < dimensions; j++) {
                x[i] += firstCopiedX[j] * rotationData[start + i * dimensions + j];
            }
        }
    }

    void shiftRotate(double x[], double shrinkage, bool shiftFlag = true, bool rotateFlag = true,
                     unsigned char shiftStart = 0, unsigned rotateStart = 0) {
        if (shiftFlag) {
            if (rotateFlag) {
                shift(x, shiftStart);
                shrink(x, shrinkage);
                rotate(x, rotateStart);
            } else {
                shift(x, shiftStart);
                shrink(x, shrinkage);
            }
        } else {
            if (rotateFlag) {
                shrink(x, shrinkage);
                rotate(x, rotateStart);
            } else {
                shrink(x, shrinkage);
            }
        }
    }

    double compose(unsigned char functionNumber, double x[], double y[], const double delta[], const double bias[]) {
        double max_w = 0;
        double sum_w = 0;
        double w[functionNumber];
        double result = 0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            y[i] += bias[i];
            w[i] = 0;
            for (unsigned char j = 0; j < dimensions; j++) {
                w[i] += pow(x[j] - shiftData[i * dimensions + j], 2.0);
            }
            if (w[i] != 0) {
                w[i] = sqrt(1.0 / w[i]) * exp(-w[i] / (2.0 * dimensions * pow(delta[i], 2.0)));
            } else {
                w[i] = INF;
            }
            if (w[i] > max_w) {
                max_w = w[i];
            }
            sum_w += w[i];
        }
        if (max_w == 0) {
            for (unsigned char i = 0; i < functionNumber; i++) {
                w[i] = 1;
            }
            sum_w = functionNumber;
        }
        for (unsigned char i = 0; i < functionNumber; i++) {
            result += y[i] * w[i] / sum_w;
        }
        return result;
    }

public:
    Function(const unsigned char number, const unsigned char dimensions, const char *name, const double minimum)
            : number(number), dimensions(dimensions), name(name), minimum(minimum), evaluations(0) {
        if (minimum != -1.0) {
            loadMatrix();
            loadShiftData();
            if (number > 5 && number < 9) {
                loadShuffleData();
            }
        }
        if (minimum == -1.0 && number > 8) {
            loadMatrix();
            loadShiftData();
        }
    }

    virtual ~Function() = default;

    [[nodiscard]] double getMinimum() const {
        return minimum;
    }

    [[nodiscard]] unsigned int getEvaluations() const {
        return evaluations;
    }

    void resetEvaluations() {
        evaluations = 0;
    }

    double apply(double x[]) {
        ++evaluations;
        memcpy(thirdCopiedX, x, dimensions * sizeof(double));
        return apply(thirdCopiedX, true, true, 0, 0) + minimum;
    }

    virtual double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart,
                         unsigned rotateStart) = 0;
};

class Zakharov : public Function {
public:
    explicit Zakharov(const unsigned char dimensions) : Function(1, dimensions, "Zakharov",
                                                                 300.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            sum1 += pow(x[i], 2);
            sum2 += 0.5 * (i + 1) * x[i];
        }
        return sum1 + pow(sum2, 2) + pow(sum2, 4);
    }
};

class Rosenbrock : public Function {
public:
    explicit Rosenbrock(const unsigned char number, const unsigned char dimensions, const double minimum) :
            Function(number, dimensions, "Rosenbrock", minimum) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 2.048e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        x[0] += 1.0;
        double term1;
        double term2;
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions - 1; i++) {
            x[i + 1] += 1.0;
            term1 = pow(x[i], 2) - x[i + 1];
            term2 = x[i] - 1.0;
            y += 100.0 * pow(term1, 2) + pow(term2, 2);
        }
        return y;
    }
};

class Schaffer : public Function {
public:
    explicit Schaffer(const unsigned char number, const unsigned char dimensions, const double minimum) :
            Function(number, dimensions, "Schaffer", minimum) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double term1;
        double term2;
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions - 1; i++) {
            // TODO - competition implementation was wrong
            term1 = sqrt(pow(x[i], 2) + pow(x[i + 1], 2));
            term2 = sin(50.0 * pow(term1, 0.2));
            y += sqrt(term1) * (pow(term2, 2) + 1);
        }
        return pow(y, 2) / pow(dimensions - 1, 2);
    }
};

class StepRastrigin : public Function {
public:
    explicit StepRastrigin(const unsigned char dimensions) : Function(4, dimensions,
                                                                      "Non-Continuous Rastrigin", 800.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5.12 / 100.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            y += (pow(x[i], 2) - 10.0 * (cos(2.0 * PI * x[i]) - 1.0));
        }
        return y;
    }
};

class Levy : public Function {
private:
    double w[DIMENSIONS];

public:
    explicit Levy(const unsigned char dimensions) : Function(5, dimensions, "Levy", 900.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        for (unsigned char i = 0; i < dimensions; i++) {
            // TODO - correct formula has x[i] - 1
            w[i] = x[i] / 4.0 + 1.0;
        }
        double term1 = pow((sin(PI * w[0])), 2);
        double term3 = pow(w[dimensions - 1] - 1.0, 2) * (pow(sin(2.0 * PI * w[dimensions - 1]),
                                                              2) + 1.0);
        double sum = 0.0;
        for (unsigned char i = 0; i < dimensions - 1; i++) {
            sum += pow(w[i] - 1, 2) * (10.0 * pow(sin(PI * w[i] + 1.0), 2) + 1.0);
        }
        return term1 + sum + term3;
    }
};

class BentCigar : public Function {
public:
    explicit BentCigar(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                              "Bent Cigar",
                                                                                              -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double y = x[0] * x[0];
        for (unsigned char i = 1; i < dimensions; i++) {
            y += pow(10.0, 6.0) * pow(x[i], 2);
        }
        return y;
    }
};

class HGBat : public Function {
public:
    explicit HGBat(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                          "HGBat",
                                                                                          -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] -= 1.0;
            sum1 += x[i];
            sum2 += pow(x[i], 2);
        }
        return sqrt(fabs(pow(sum2, 2) - pow(sum1, 2))) +
               (sum1 + sum2 / 2.0) / dimensions + 0.5;
    }
};

class Rastrigin : public Function {
public:
    explicit Rastrigin(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                              "Rastrigin",
                                                                                              -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5.12e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            y += pow(x[i], 2) - 10.0 * cos(2.0 * PI * x[i]) + 10.0;
        }
        return y;
    }
};

class Hybrid1 : public Function {
private:
    const unsigned char functionNumber = 3;
    unsigned char newStarts[3];

    Function *functions[3];
public:
    explicit Hybrid1(const unsigned char dimensions) : Function(6, dimensions, "Hybrid 1 (3)",
                                                                1800.0) {
        double percentages[3] = {0.4, 0.4, 0.2};
        unsigned char newDimensions[functionNumber];
        unsigned char sum = 0.0;
        for (unsigned char i = 0; i < functionNumber - 1; i++) {
            newDimensions[i] = ceil(percentages[i] * dimensions);
            sum += newDimensions[i];
        }
        newDimensions[functionNumber - 1] = dimensions - sum;
        newStarts[0] = 0;
        for (unsigned char i = 1; i < functionNumber; i++) {
            newStarts[i] = newStarts[i - 1] + newDimensions[i - 1];
        }
        functions[0] = new BentCigar(number, newDimensions[0]);
        functions[1] = new HGBat(number, newDimensions[1]);
        functions[2] = new Rastrigin(number, newDimensions[2]);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        for (unsigned char i = 0; i < dimensions; i++) {
            firstCopiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            y += functions[i]->apply(&firstCopiedX[newStarts[i]], false, false, 0, 0);
        }
        return y;
    }
};

class Katsuura : public Function {
public:
    explicit Katsuura(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                             "Katsuura",
                                                                                             -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double term1;
        double term2;
        double term3 = 10.0 / pow(dimensions, 1.2);
        double sum;
        double y = 1.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            sum = 0.0;
            for (unsigned char j = 1; j <= 32; j++) {
                term1 = pow(2.0, j);
                term2 = term1 * x[i];
                sum += fabs(term2 - floor(term2 + 0.5)) / term1;
            }
            y *= pow(1.0 + (i + 1) * sum, term3);
        }
        term1 = 10.0 / pow(dimensions, 2);
        return term1 * (y - 1);
    }
};

class Ackley : public Function {
public:
    explicit Ackley(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                           "Ackley",
                                                                                           -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            sum1 += pow(x[i], 2);
            sum2 += cos(2.0 * PI * x[i]);
        }
        sum1 = -0.2 * sqrt(sum1 / dimensions);
        sum2 /= dimensions;
        return -20 * exp(sum1) - exp(sum2) + 20.0 + E;
    }
};

class Schwefel : public Function {
public:
    explicit Schwefel(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                             "Schwefel",
                                                                                             -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 10.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double term;
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] += 4.209687462275036e+002;
            if (x[i] > 500) {
                y -= (500.0 - fmod(x[i], 500)) * sin(sqrt(500.0 - fmod(x[i], 500)));
                term = (x[i] - 500.0) / 100.0;
                y += pow(term, 2) / dimensions;
            } else if (x[i] < -500) {
                y -= (-500.0 + fmod(fabs(x[i]), 500)) * sin(sqrt(500.0 - fmod(fabs(x[i]),
                                                                              500)));
                term = (x[i] + 500.0) / 100.0;
                y += pow(term, 2) / dimensions;
            } else
                y -= x[i] * sin(sqrt(fabs(x[i])));
        }
        return y + 4.189828872724338e+002 * dimensions;
    }
};

class Hybrid2 : public Function {
private:
    const unsigned char functionNumber = 6;
    unsigned char newStarts[6];

    Function *functions[6];
public:
    explicit Hybrid2(const unsigned char dimensions) : Function(7, dimensions, "Hybrid 2 (6)",
                                                                2000.0) {
        double percentages[6] = {0.1, 0.2, 0.2, 0.2, 0.1, 0.2};
        unsigned char newDimensions[functionNumber];
        unsigned char sum = 0.0;
        for (unsigned char i = 0; i < functionNumber - 1; i++) {
            newDimensions[i] = ceil(percentages[i] * dimensions);
            sum += newDimensions[i];
        }
        newDimensions[functionNumber - 1] = dimensions - sum;
        newStarts[0] = 0;
        for (unsigned char i = 1; i < functionNumber; i++) {
            newStarts[i] = newStarts[i - 1] + newDimensions[i - 1];
        }
        functions[0] = new HGBat(number, newDimensions[0]);
        functions[1] = new Katsuura(number, newDimensions[1]);
        functions[2] = new Ackley(number, newDimensions[2]);
        functions[3] = new Rastrigin(number, newDimensions[3]);
        functions[4] = new Schwefel(number, newDimensions[4]);
        functions[5] = new Schaffer(number, newDimensions[5], -1.0);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        for (unsigned char i = 0; i < dimensions; i++) {
            firstCopiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            double z = functions[i]->apply(&firstCopiedX[newStarts[i]], false, false, 0, 0);
            y += z;
        }
        return y;
    }
};

class HappyCat : public Function {
public:
    explicit HappyCat(const unsigned char number, const unsigned char dimensions) : Function(number, dimensions,
                                                                                             "Happy cat",
                                                                                             -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] -= 1.0;
            sum1 += x[i];
            sum2 += pow(x[i], 2);
        }
        return pow(fabs(sum2 - dimensions), 0.25) + (0.5 * sum2 + sum1) / dimensions + 0.5;
    }
};

class GriewankRosenbrock : public Function {
public:
    explicit GriewankRosenbrock(const unsigned char number, const unsigned char dimensions) :
            Function(number, dimensions, "Griewank and Rosenbrock", -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 5e-2, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double term1;
        double term2;
        double term3;
        double y = 0.0;
        x[0] += 1.0;
        for (unsigned char i = 0; i < dimensions - 1; i++) {
            x[i + 1] += 1.0;
            term1 = pow(x[i], 2) - x[i + 1];
            term2 = x[i] - 1.0;
            term3 = 100.0 * pow(term1, 2) + pow(term2, 2);
            // TODO - correct formula has cos(x[i] / sqrt(i))
            y += pow(term3, 2) / 4000.0 - cos(term3) + 1.0;
        }
        term1 = pow(x[dimensions - 1], 2) - x[0];
        term2 = x[dimensions - 1] - 1.0;
        term3 = 100.0 * pow(term1, 2) + pow(term2, 2);
        y += pow(term3, 2) / 4000.0 - cos(term3) + 1.0;
        return y;
    }
};

class Hybrid3 : public Function {
private:
    const unsigned char functionNumber = 5;
    unsigned char newStarts[5];

    Function *functions[5];
public:
    explicit Hybrid3(const unsigned char dimensions) : Function(8, dimensions, "Hybrid 3 (5)",
                                                                2200.0) {
        double percentages[5] = {0.3, 0.2, 0.2, 0.1, 0.2};
        unsigned char newDimensions[functionNumber];
        unsigned char sum = 0.0;
        for (unsigned char i = 0; i < functionNumber - 1; i++) {
            newDimensions[i] = ceil(percentages[i] * dimensions);
            sum += newDimensions[i];
        }
        newDimensions[functionNumber - 1] = dimensions - sum;
        newStarts[0] = 0;
        for (unsigned char i = 1; i < functionNumber; i++) {
            newStarts[i] = newStarts[i - 1] + newDimensions[i - 1];
        }
        functions[0] = new Katsuura(number, newDimensions[0]);
        functions[1] = new HappyCat(number, newDimensions[1]);
        functions[2] = new GriewankRosenbrock(number, newDimensions[2]);
        functions[3] = new Schwefel(number, newDimensions[3]);
        functions[4] = new Ackley(number, newDimensions[4]);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        for (unsigned char i = 0; i < dimensions; i++) {
            firstCopiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            y += functions[i]->apply(&firstCopiedX[newStarts[i]], false, false, 0, 0);
        }
        return y;
    }
};

class Ellipsis : public Function {
public:
    explicit Ellipsis(const unsigned char number, const unsigned char dimensions) :
            Function(number, dimensions, "High Conditioned Elliptic", -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double y = 0;
        for (unsigned char i = 0; i < dimensions; i++) {
            y += pow(10.0, 6.0 * i / (dimensions - 1)) * pow(x[i], 2);
        }
        return y;
    }
};

class Discus : public Function {
public:
    explicit Discus(const unsigned char number, const unsigned char dimensions) :
            Function(number, dimensions, "Discus", -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double y = pow(10.0, 6.0) * pow(x[0], 2);
        for (unsigned char i = 1; i < dimensions; i++) {
            y += pow(x[i], 2);
        }
        return y;
    }
};

class Composition1 : public Function {
private:
    const unsigned char functionNumber = 5;
    const double delta[5] = {10, 20, 30, 40, 50};
    const double bias[5] = {0, 200, 300, 100, 400};

    Function *functions[5];
public:
    explicit Composition1(const unsigned char dimensions) : Function(9, dimensions, "Composition 1 (5)",
                                                                     2300.0) {
        functions[0] = new Rosenbrock(number, dimensions, -1.0);
        functions[1] = new Ellipsis(number, dimensions);
        functions[2] = new BentCigar(number, dimensions);
        functions[3] = new Discus(number, dimensions);
        functions[4] = new Ellipsis(number, dimensions);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        double y[functionNumber];
        bool shiftFlags[5] = {shiftFlag, shiftFlag, shiftFlag, shiftFlag, shiftFlag};
        bool rotateFlags[5] = {rotateFlag, rotateFlag, rotateFlag, rotateFlag, false};
        for (unsigned char i = 0; i < functionNumber; i++) {
            memcpy(secondCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(secondCopiedX, shiftFlags[i], rotateFlags[i],
                                       i * dimensions, i * dimensions * dimensions);
        }
        // TODO - the documentation contains other values
        y[1] *= 1e-6;
        y[2] *= 1e-26;
        y[3] *= 1e-6;
        y[4] *= 1e-6;
        return compose(functionNumber, x, y, delta, bias);
    }
};

class Composition2 : public Function {
private:
    const unsigned char functionNumber = 3;
    const double delta[3] = {20, 10, 10};
    const double bias[3] = {0, 200, 100};

    Function *functions[3];
public:
    explicit Composition2(const unsigned char dimensions) : Function(10, dimensions, "Composition 2 (3)",
                                                                     2400.0) {
        functions[0] = new Schwefel(number, dimensions);
        functions[1] = new Rastrigin(number, dimensions);
        functions[2] = new HGBat(number, dimensions);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        double y[functionNumber];
        bool shiftFlags[3] = {shiftFlag, shiftFlag, shiftFlag};
        bool rotateFlags[3] = {false, rotateFlag, rotateFlag};
        for (unsigned char i = 0; i < functionNumber; i++) {
            memcpy(secondCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(secondCopiedX, shiftFlags[i], rotateFlags[i],
                                       i * dimensions, i * dimensions * dimensions);
        }
        return compose(functionNumber, x, y, delta, bias);
    }
};

class ExpandedSchaffer : public Function {
public:
    explicit ExpandedSchaffer(const unsigned char number, const unsigned char dimensions) :
            Function(number, dimensions, "Expanded Schaffer", -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 1.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double term1;
        double term2;
        double term3;
        double y = 0.0;
        for (unsigned char i = 0; i < dimensions - 1; i++) {
            term1 = pow(x[i], 2) + pow(x[i + 1], 2);
            term2 = pow(sin(sqrt(term1)), 2);
            term3 = 1.0 + 0.001 * term1;
            y += 0.5 + (term2 - 0.5) / pow(term3, 2);
        }
        term1 = pow(x[dimensions - 1], 2) + pow(x[0], 2);
        term2 = pow(sin(sqrt(term1)), 2);
        term3 = 1.0 + 0.001 * term1;
        y += 0.5 + (term2 - 0.5) / pow(term3, 2);
        return y;
    }
};

class Griewank : public Function {
public:
    explicit Griewank(const unsigned char number, const unsigned char dimensions) :
            Function(number, dimensions, "Griewank", -1.0) {}

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        shiftRotate(x, 6.0, shiftFlag, rotateFlag, shiftStart, rotateStart);
        double sum = 0.0;
        double product = 1.0;
        for (unsigned char i = 0; i < dimensions; i++) {
            sum += pow(x[i], 2);
            product *= cos(x[i] / sqrt(i + 1.0));
        }
        return sum / 4000.0 - product + 1.0;
    }
};

class Composition3 : public Function {
private:
    const unsigned char functionNumber = 5;
    const double delta[5] = {20, 20, 30, 30, 20};
    const double bias[5] = {0, 200, 300, 400, 200};

    Function *functions[5];
public:
    explicit Composition3(const unsigned char dimensions) : Function(11, dimensions, "Composition 3 (5)",
                                                                     2600.0) {
        functions[0] = new ExpandedSchaffer(number, dimensions);
        functions[1] = new Schwefel(number, dimensions);
        functions[2] = new Griewank(number, dimensions);
        functions[3] = new Rosenbrock(number, dimensions, -1.0);
        functions[4] = new Rastrigin(number, dimensions);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        double y[functionNumber];
        for (unsigned char i = 0; i < functionNumber; i++) {
            memcpy(secondCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(secondCopiedX, shiftFlag, rotateFlag, i * dimensions,
                                       i * dimensions * dimensions);
        }
        // TODO - the documentation contains other values
        y[0] /= 2.0e+3;
        y[2] *= 10.0;
        y[4] *= 10.0;
        return compose(functionNumber, x, y, delta, bias);
    }
};

class Composition4 : public Function {
private:
    const unsigned char functionNumber = 6;
    const double delta[6] = {10, 20, 30, 40, 50, 60};
    const double bias[6] = {0, 300, 500, 100, 400, 200};

    Function *functions[6];
public:
    explicit Composition4(const unsigned char dimensions) : Function(12, dimensions, "Composition 4 (6)",
                                                                     2700.0) {
        functions[0] = new HGBat(number, dimensions);
        functions[1] = new Rastrigin(number, dimensions);
        functions[2] = new Schwefel(number, dimensions);
        functions[3] = new BentCigar(number, dimensions);
        functions[4] = new Ellipsis(number, dimensions);
        functions[5] = new ExpandedSchaffer(number, dimensions);
    }

    double apply(double x[], bool shiftFlag, bool rotateFlag, unsigned char shiftStart, unsigned rotateStart) override {
        double y[functionNumber];
        for (unsigned char i = 0; i < functionNumber; i++) {
            memcpy(secondCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(secondCopiedX, shiftFlag, rotateFlag, i * dimensions,
                                       i * dimensions * dimensions);
        }
        // TODO - the documentation contains other values
        y[0] *= 10.0;
        y[1] *= 10.0;
        y[2] *= 10.0 / 4.0;
        y[3] *= 1e-26;
        y[4] *= 1e-6;
        y[5] /= 2.0e+3;
        return compose(functionNumber, x, y, delta, bias);
    }
};

class Functions {
private:
    Function *functions[FUNCTIONS];

public:
    Functions(const unsigned char dimensions) {
        functions[0] = new Zakharov(dimensions);
        functions[1] = new Rosenbrock(2, dimensions, 400.0);
        functions[2] = new Schaffer(3, dimensions, 600.0);
        functions[3] = new StepRastrigin(dimensions);
        functions[4] = new Levy(dimensions);
        functions[5] = new Hybrid1(dimensions);
        functions[6] = new Hybrid2(dimensions);
        functions[7] = new Hybrid3(dimensions);
        functions[8] = new Composition1(dimensions);
        functions[9] = new Composition2(dimensions);
        functions[10] = new Composition3(dimensions);
        functions[11] = new Composition4(dimensions);
    }

    ~Functions() {
        for (auto &function: functions) {
            delete function;
        }
    }

    Function *getFunction(unsigned char index) {
        return functions[index];
    }
};

class EPRPSO {
private:
    const unsigned char dimensions{};
    const unsigned int maximumEvaluations{};
    const double left{};
    const double right{};
    const unsigned char populationSize;
    const double initialInertia;
    const double minimumInertia;
    double inertia;
    const double cognition;
    const double social;
    const double inertiaDecay;
    const double speedDecay;
    const double mutation;
    const unsigned char maximumStuckIndividuals;
    const double regenerationError;
    unsigned int stuckGenerations;
    unsigned int maximumStuckGenerations;
    const unsigned int minimumStuckGenerations;
    const double stuckGenerationsDecay;
    unsigned int currentGeneration;
    const double initialTemperature;
    const double minimumTemperature;
    double temperature;
    const double temperatureDecay;
    double checkpointBestEvaluation;
    double bestEvaluation;
    unsigned char checkpointIndex;
    unsigned int checkpoint;

    double pastPopulation[POPULATION][DIMENSIONS];
    double population[POPULATION][DIMENSIONS];
    double populationSpeed[POPULATION][DIMENSIONS];
    double pastPopulationEvaluation[POPULATION];
    double populationEvaluation[POPULATION];
    double bestGenerationIndividual[DIMENSIONS];
    double individualInertia[DIMENSIONS];
    double pastIndividualDifference[DIMENSIONS];
    double individualCognition[DIMENSIONS];
    double bestIndividualDifference[DIMENSIONS];
    double individualSocial[DIMENSIONS];
    double randomSpeed[DIMENSIONS];

    Function *function;

public:
    EPRPSO(const unsigned char dimensions, const unsigned int maximumEvaluations, const double left, const double right,
           const unsigned char populationSize, const double minimumInertia, const double inertia,
           const double cognition, const double social, const double inertiaDecay, const double speedDecay,
           const double mutation, const unsigned char maximumStuckIndividuals, const double regenerationError,
           const unsigned int maximumStuckGenerations, const unsigned int minimumStuckGenerations,
           const double stuckGenerationsDecay, const double minimumTemperature, const double temperature,
           const double temperatureDecay, Function *function) :
            dimensions(dimensions), maximumEvaluations(maximumEvaluations), left(left), right(right),
            populationSize(populationSize), initialInertia(inertia), minimumInertia(minimumInertia), inertia(inertia),
            cognition(cognition), social(social), inertiaDecay(inertiaDecay), speedDecay(speedDecay),
            mutation(mutation), maximumStuckIndividuals(maximumStuckIndividuals), regenerationError(regenerationError),
            stuckGenerations(0), maximumStuckGenerations(maximumStuckGenerations),
            minimumStuckGenerations(minimumStuckGenerations), stuckGenerationsDecay(stuckGenerationsDecay),
            currentGeneration(1), initialTemperature(temperature), minimumTemperature(minimumTemperature),
            temperature(temperature), temperatureDecay(temperatureDecay), checkpointBestEvaluation(INF),
            bestEvaluation(INF), checkpointIndex(0), function(function) {
        checkpoint = computeCheckpoint();
    }

    [[nodiscard]] unsigned int computeCheckpoint() const {
        return (unsigned int) (1 / pow(dimensions, fabs(checkpointIndex / 5.0 - 3.0)) *
                               maximumEvaluations);
    }

    void generatePopulation() {
        for (unsigned char i = 0; i < populationSize; i++) {
            for (unsigned char j = 0; j < dimensions; j++) {
                population[i][j] = random(left, right);
                populationSpeed[i][j] = speedDecay * random(left, right);
            }
        }
    }

    void regeneratePopulation() {
        generatePopulation();
        inertia = initialInertia;
        temperature = initialTemperature;
    }

    bool evaluatePopulation() {
        double bestGenerationEvaluation = INF;
        unsigned char stuckIndividuals = 0;
        for (unsigned char i = 0; i < populationSize; i++) {
            populationEvaluation[i] = function->apply(population[i]);
            if (populationEvaluation[i] < checkpointBestEvaluation) {
                checkpointBestEvaluation = populationEvaluation[i];
            }
            if (populationEvaluation[i] < bestEvaluation) {
                bestEvaluation = populationEvaluation[i];
            }
            if (function->getEvaluations() >= checkpoint) {
                ++checkpointIndex;
                checkpoint = computeCheckpoint();
                checkpointBestEvaluation = INF;
            }
            if (function->getEvaluations() >= maximumEvaluations) {
                return false;
            }
            if (fabs(populationEvaluation[i] - function->getMinimum()) <= STOP_ERROR) {
                return false;
            }
            if (populationEvaluation[i] < bestGenerationEvaluation) {
                bestGenerationEvaluation = populationEvaluation[i];
                memcpy(bestGenerationIndividual, population[i], dimensions * sizeof(double));
            }
            if (populationEvaluation[i] - bestEvaluation <= regenerationError) {
                ++stuckIndividuals;
            }
        }
        if (stuckIndividuals >= maximumStuckIndividuals) {
            ++stuckGenerations;
        } else {
            stuckGenerations = 0;
        }
        return true;
    }

    void updatePopulation() {
        for (unsigned char i = 0; i < populationSize; i++) {
            if (populationEvaluation[i] < pastPopulationEvaluation[i]) {
                pastPopulationEvaluation[i] = populationEvaluation[i];
                memcpy(pastPopulation[i], population[i], dimensions * sizeof(double));
            }
        }
    }

    static void multiply(const unsigned char length, double destination[], const double source[], double multiplier) {
        for (unsigned char i = 0; i < length; i++) {
            destination[i] = source[i] * multiplier;
        }
    }

    static void difference(const unsigned char length, double destination[], const double first[],
                           const double second[]) {
        for (unsigned char i = 0; i < length; i++) {
            destination[i] = first[i] - second[i];
        }
    }

    void generateRandomSpeed() {
        for (unsigned char i = 0; i < dimensions; i++) {
            randomSpeed[i] = inertia * speedDecay * mutation * random(left, right);
        }
    }

    void sumPopulationSpeed(unsigned char individual) {
        for (unsigned char j = 0; j < dimensions; j++) {
            populationSpeed[individual][j] = individualInertia[j] + individualCognition[j] + individualSocial[j] +
                                             randomSpeed[j];
        }
    }

    void simulatePopulation() {
        double newTemperature;
        for (unsigned char i = 0; i < populationSize; i++) {
            multiply(dimensions, individualInertia, populationSpeed[i], inertia);
            difference(dimensions, pastIndividualDifference, pastPopulation[i],
                       population[i]);
            multiply(dimensions, individualCognition, pastIndividualDifference,
                     cognition * realRandom01());
            difference(dimensions, bestIndividualDifference, bestGenerationIndividual,
                       population[i]);
            multiply(dimensions, individualSocial, bestIndividualDifference,
                     social * realRandom01());
            generateRandomSpeed();
            sumPopulationSpeed(i);
            for (unsigned char j = 0; j < dimensions; j++) {
                newTemperature = max(minimumTemperature, exp(-temperatureDecay / (temperature / currentGeneration)));
                if (realRandom01() < newTemperature) {
                    populationSpeed[i][j] = speedDecay * random(left, right);
                }
                population[i][j] += populationSpeed[i][j];
                if (population[i][j] < left) {
                    population[i][j] = left;
                    populationSpeed[i][j] = 0;
                } else {
                    if (population[i][j] > right) {
                        population[i][j] = right;
                        populationSpeed[i][j] = 0;
                    }
                }
            }
        }
        inertia = max(minimumInertia, inertia * inertiaDecay);
    }

    double run() {
        generatePopulation();
        if (!evaluatePopulation()) {
            return bestEvaluation;
        }
        if (stuckGenerations >= maximumStuckGenerations) {
            stuckGenerations = 0;
            maximumStuckGenerations = max(minimumStuckGenerations, (unsigned int) (maximumStuckGenerations *
                                                                                   stuckGenerationsDecay));
            regeneratePopulation();
            if (!evaluatePopulation()) {
                return bestEvaluation;
            }
        }
        memcpy(pastPopulation, population, populationSize * dimensions * sizeof(double));
        memcpy(pastPopulationEvaluation, populationEvaluation, populationSize * sizeof(double));
        while (true) {
            ++currentGeneration;
            simulatePopulation();
            if (!evaluatePopulation()) {
                return bestEvaluation;
            }
            if (stuckGenerations >= maximumStuckGenerations) {
                stuckGenerations = 0;
                maximumStuckGenerations = max(minimumStuckGenerations, (unsigned int) (maximumStuckGenerations *
                                                                                       stuckGenerationsDecay));
                regeneratePopulation();
                if (!evaluatePopulation()) {
                    return bestEvaluation;
                }
            }
            updatePopulation();
        }
    }
};

class Algorithm {
private:
    double randomSeeds[RANDOM_SEEDS];
    Functions *functions;

public:
    Algorithm() {
        FILE *file = fopen("../input_data/Rand_Seeds.txt", "r");
        for (double &randomSeed: randomSeeds) {
            fscanf(file, "%lf", &randomSeed);
        }
        fclose(file);
        functions = new Functions(DIMENSIONS);
    }

    void getResults(double (results)[][RUNS], const double minimumInertia, const double inertia, const double cognition,
                    const double social, const double inertiaDecay, const double speedDecay, const double speedMutation,
                    const unsigned char stuckIndividuals, const unsigned int maximumStuckGenerations,
                    const unsigned int minimumStuckGenerations, const double stuckGenerationsDecay,
                    const double minimumTemperature, const double temperature, const double temperatureDecay) {
        for (unsigned char i = 0; i < FUNCTIONS; i++) {
            for (unsigned char j = 0; j < RUNS; j++) {
                unsigned short randomIndex = (DIMENSIONS / 10 * (i + 1) * RUNS + j + 1) - RUNS;
                randomIndex = (randomIndex % 1000);
                srand(randomSeeds[randomIndex]);
                Function *function = functions->getFunction(i);
                function->resetEvaluations();
                EPRPSO eprpso(DIMENSIONS, EVALUATIONS, LEFT, RIGHT,
                              POPULATION, minimumInertia, inertia, cognition, social, inertiaDecay,
                              speedDecay, speedMutation, stuckIndividuals,
                              REGENERATION_ERROR, maximumStuckGenerations, minimumStuckGenerations,
                              stuckGenerationsDecay, minimumTemperature, temperature, temperatureDecay, function);
                results[i][j] = eprpso.run() - function->getMinimum();
            }
        }
    }
};

class AlgorithmMinimum {
    unsigned char dimensions;
    unsigned short chromosomeSize;
    unsigned int populationSize;
    double leftEnds[PARAMETERS];
    double rightEnds[PARAMETERS];
    unsigned char population[individuals * bits * PARAMETERS];
    unsigned char newPopulation[individuals * bits * PARAMETERS];
    long long generationNumbers[individuals * PARAMETERS];
    double generationArguments[individuals * PARAMETERS];
    double generationValues[individuals];
    double fitness[individuals];
    double selection[individuals + 1];
    double fitnessAverage[generations];
    double fitnessMaximum[generations];
    double minimumArguments[PARAMETERS];
    double minimumValue;

    FILE *file;
    Algorithm *algorithm;

    const unsigned int seed;
    default_random_engine *generator;
    uniform_int_distribution<int> *pointDistribution;
    uniform_real_distribution<double> *realDistribution01;

public:
    AlgorithmMinimum(unsigned char newDimensions, double newLeftEnds[], double newRightEnds[]) :
            seed(system_clock::now().time_since_epoch().count()) {
        minimumValue = DBL_MAX;
        dimensions = newDimensions;
        chromosomeSize = bits * dimensions;
        populationSize = individuals * chromosomeSize;
        memcpy(leftEnds, newLeftEnds, dimensions * sizeof(double));
        memcpy(rightEnds, newRightEnds, dimensions * sizeof(double));

        char fileName[64];
        strcpy(fileName, "../");
        strcat(fileName, ALGORITHM);
        strcat(fileName, "/");
        strcat(fileName, ALGORITHM);
        strcat(fileName, " parameters.txt");
        file = fopen(fileName, "a");
        algorithm = new Algorithm();
        generator = new default_random_engine(seed);
        pointDistribution = new uniform_int_distribution<int>(2, chromosomeSize - 2);
        realDistribution01 = new uniform_real_distribution<double>(0.0, 1.0);
    }

    ~AlgorithmMinimum() {
        delete realDistribution01;
        delete generator;
        delete algorithm;
        fclose(file);
    }

    double getResult(double arguments[]) {
        double results[FUNCTIONS][RUNS];
        algorithm->getResults(results, arguments[0], arguments[1], arguments[2],
                              arguments[3], arguments[4], arguments[5],
                              arguments[6], arguments[7], arguments[8],
                              arguments[9], arguments[10],
                              arguments[11], arguments[12], arguments[13]);
        double result = 0;
        for (unsigned char i = 0; i < FUNCTIONS; i++) {
            double sum = 0;
            for (unsigned char j = 0; j < RUNS; j++) {
                sum += results[i][j];
            }
            result += (sum / RUNS);
        }
        return result;
    }

    void setFirstValue() {
        for (unsigned char index = 0; index < dimensions; ++index) {
            minimumArguments[index] = leftEnds[index];
        }
        minimumValue = getResult(minimumArguments);
    }

    void generateRandomPopulation() {
        for (unsigned int index = 0; index < populationSize; ++index) {
            population[index] = naturalRandom(2);
        }
    }

    void mutate() {
        for (unsigned int index = 0; index < populationSize; ++index) {
            if (realDistribution01->operator()(*generator) < mutation) {
                population[index] = 1 - population[index];
            }
        }
    }

    unsigned short hammingDistance(unsigned short first, unsigned short second) {
        unsigned short count = 0;
        for (unsigned short index = 0; index < chromosomeSize; ++index) {
            count += population[chromosomeSize * first + index] ^ population[chromosomeSize * second + index];
        }
        return count;
    }

    void singlePoint(unsigned short first, unsigned short second) {
        if (hammingDistance(first, second) > dimensions * 2) {
            const unsigned short point = pointDistribution->operator()(*generator);
            for (unsigned short index = point; index < chromosomeSize; ++index) {
                swap(population[chromosomeSize * first + index], population[chromosomeSize * second + index]);
            }
        }
    }

    void multiplePoints(unsigned short first, unsigned short second) {
        if (hammingDistance(first, second) > 10) {
            vector<unsigned short> points;
            points.reserve(cuts);
            for (unsigned short iterator = 0; iterator < cuts; ++iterator) {
                points.emplace_back(pointDistribution->operator()(*generator));
            }
            sort(points.begin(), points.end());
            auto iterator = points.begin();
            for (unsigned short cut = 0; cut < cuts; cut += 2) {
                if (cut == cuts - 1) {
                    for (unsigned short index = *iterator; index < chromosomeSize; ++index) {
                        swap(population[chromosomeSize * first + index], population[chromosomeSize * second +
                                                                                    index]);
                    }
                } else {
                    for (unsigned short index = *iterator; index < *(iterator + 1); ++index) {
                        swap(population[chromosomeSize * first + index], population[chromosomeSize * second +
                                                                                    index]);
                    }
                    iterator += 2;
                }
            }
        }
    }

    void singlePointCrossover() {
        vector<pair<double, unsigned short>> probabilities;
        probabilities.reserve(individuals);
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            probabilities.emplace_back(make_pair(realDistribution01->operator()(*generator), iterator));
        }
        sort(probabilities.begin(), probabilities.end());
        auto individual = probabilities.begin();
        ++individual;
        for (; individual < probabilities.end() && individual->first < crossover; individual += 2) {
            singlePoint((individual - 1)->second, individual->second);
        }
        if (individual < probabilities.end()) {
            if ((individual - 1)->first < crossover) {
                singlePoint((individual - 1)->second, individual->second);
            }
        } else {
            if (individual == probabilities.end()) {
                singlePoint((individual - 2)->second, (individual - 1)->second);
            }
        }
    }

    void multiplePointsCrossover() {
        vector<pair<double, unsigned short>> probabilities;
        probabilities.reserve(individuals);
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            probabilities.emplace_back(make_pair(realDistribution01->operator()(*generator), iterator));
        }
        sort(probabilities.begin(), probabilities.end());
        auto individual = probabilities.begin();
        ++individual;
        for (; individual < probabilities.end() && individual->first < crossover; individual += 2) {
            multiplePoints((individual - 1)->second, individual->second);
        }
        if (individual < probabilities.end()) {
            if ((individual - 1)->first < crossover) {
                multiplePoints((individual - 1)->second, individual->second);
            }
        } else {
            if (individual == probabilities.end()) {
                multiplePoints((individual - 2)->second, (individual - 1)->second);
            }
        }
    }

    void computeArguments(const unsigned short chromosomeStart, const unsigned int geneStart) {
        for (unsigned char index = 0; index < dimensions; ++index) {
            generationNumbers[chromosomeStart + index] = 0;
            const unsigned short argument = bits * index;
            for (unsigned char position = 0; position < bits; ++position) {
                generationNumbers[chromosomeStart + index] = generationNumbers[chromosomeStart + index] * 2 +
                                                             population[geneStart + argument + position];
                // convert base 2 to base 10
            }
            generationArguments[chromosomeStart + index] =
                    static_cast<double>(generationNumbers[chromosomeStart + index]) / (pow(2, bits) - 1);
            // translate to closed interval [0, 1]
            generationArguments[chromosomeStart + index] *= rightEnds[index] - leftEnds[index];
            // translate to closed interval [0, rightEnds - leftEnds]
            generationArguments[chromosomeStart + index] += leftEnds[index];
            // translate to closed interval [leftEnds, rightEnds]
        }
    }

    void evaluateValue(const unsigned short iterator, const unsigned short chromosomeStart) {
        generationValues[iterator] = getResult(generationArguments + chromosomeStart);
        if (minimum(generationValues[iterator], minimumValue)) {
            memcpy(minimumArguments, generationArguments + chromosomeStart, dimensions * sizeof(double));
            minimumValue = generationValues[iterator];
        }
    }

    void computeFitness(unsigned short generation) {
        double maximumValue = DBL_MIN;
        double fitnessSum = 0;
        fitnessMaximum[generation] = DBL_MIN;
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            const unsigned short chromosomeStart = dimensions * iterator;
            const unsigned int geneStart = chromosomeSize * iterator;
            computeArguments(chromosomeStart, geneStart);
            generationValues[iterator] = getResult(generationArguments + chromosomeStart);
            if (generationValues[iterator] > maximumValue) {
                maximumValue = generationValues[iterator];
            }
        }
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            fitness[iterator] = 1.01 * maximumValue - generationValues[iterator];
            if (fitness[iterator] > fitnessMaximum[generation]) {
                fitnessMaximum[generation] = fitness[iterator];
            }
            fitnessSum += fitness[iterator];
        }
        fitnessAverage[generation] = fitnessSum / individuals;
        selection[0] = 0;
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            selection[iterator + 1] = selection[iterator] + fitness[iterator] / fitnessSum;
        }
    }

    unsigned short getRouletteWheelIndividual() {
        const double position = realDistribution01->operator()(*generator);
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            if (selection[iterator] <= position && position <= selection[iterator + 1]) {
                return iterator;
            }
        }
    }

    unsigned short getTournamentIndividual() {
        unsigned char participants[tournament];
        for (unsigned char index = 0; index < tournament; ++index) {
            participants[index] = naturalRandom(individuals);
        }
        double bestValue = DBL_MAX;
        unsigned char bestIndex = 0;
        for (unsigned char index = 0; index < tournament; ++index) {
            if (minimum(generationValues[participants[index]], bestValue)) {
                bestValue = generationValues[participants[index]];
                bestIndex = participants[index];
            }
        }
        return bestIndex;
    }

    void select(unsigned short generation) {
        computeFitness(generation);
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            memcpy(newPopulation + (unsigned long long) chromosomeSize * iterator,
                   population + (unsigned long long) chromosomeSize * getTournamentIndividual(),
                   chromosomeSize * sizeof(unsigned char));
        }
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            memcpy(population + (unsigned long long) chromosomeSize * iterator,
                   newPopulation + (unsigned long long) chromosomeSize * getTournamentIndividual(),
                   chromosomeSize * sizeof(unsigned char));
        }
    }

    void computeMinimum() {
        for (unsigned short iterator = 0; iterator < individuals; ++iterator) {
            const unsigned short chromosomeStart = dimensions * iterator;
            const unsigned int geneStart = chromosomeSize * iterator;
            computeArguments(chromosomeStart, geneStart);
            evaluateValue(iterator, chromosomeStart);
        }
    }

    void run() {
        fprintf(file, "\n");
        setFirstValue();
        generateRandomPopulation();
        computeMinimum();
        for (unsigned short generation = 0; generation < generations; ++generation) {
            printf("Generation %d:\n", generation + 1);
            select(generation);
            double ratio = fitnessAverage[generation] / fitnessMaximum[generation];
            if (ratio > 0.9) {
                singlePointCrossover();
                mutation += 0.0002;
            } else {
                multiplePointsCrossover();
                if (mutation > 0.001 && ratio < 0.7) {
                    mutation -= 0.0001;
                }
            }
            mutate();
            computeMinimum();
            for (unsigned char index = 0; index < dimensions; index++) {
                printf("%.3f ", minimumArguments[index]);
                fprintf(file, "%.3f ", minimumArguments[index]);
            }
            printf("%f\n", minimumValue);
            fprintf(file, "%f\n", minimumValue);
            fflush(file);
        }
        printf("\n");
        fprintf(file, "\n");
    }
};

int main() {
    double leftEnds[PARAMETERS];
    double rightEnds[PARAMETERS];

    // minimum inertia
    leftEnds[0] = 0.001;
    rightEnds[0] = 0.1;

    // inertia
    leftEnds[1] = 0.5;
    rightEnds[1] = 1.05;

    // cognition
    leftEnds[2] = 0.8;
    rightEnds[2] = 3.05;

    // social
    leftEnds[3] = 0.8;
    rightEnds[3] = 3.05;

    // inertia decay
    leftEnds[4] = 0.5;
    rightEnds[4] = 0.999;

    // speed decay
    leftEnds[5] = 0.05;
    rightEnds[5] = 0.2;

    // speed mutation
    leftEnds[6] = 0.005;
    rightEnds[6] = 0.02;

    // stuck individuals
    leftEnds[7] = 10;
    rightEnds[7] = 50;

    // maximum stuck generations
    leftEnds[8] = 100;
    rightEnds[8] = 500;

    // minimum stuck generations
    leftEnds[9] = 75;
    rightEnds[9] = 400;

    // stuck generations decay
    leftEnds[10] = 0.5;
    rightEnds[10] = 0.999;

    // minimum temperature
    leftEnds[11] = 0.005;
    rightEnds[11] = 0.2;

    // temperature
    leftEnds[12] = 5;
    rightEnds[12] = 100;

    // temperature decay
    leftEnds[13] = 0.5;
    rightEnds[13] = 0.999;

    AlgorithmMinimum algorithmMinimum(PARAMETERS, leftEnds, rightEnds);

    for (unsigned char index = 1; index <= runs; ++index) {
        srand(static_cast<unsigned int>(time(nullptr) * clock()));
        printf("Run %d:\n", index);
        algorithmMinimum.run();
    }

    return 0;
}