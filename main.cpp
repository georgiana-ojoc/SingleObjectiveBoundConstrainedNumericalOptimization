#include <cmath>
#include <string>

#define CHECKPOINTS     15
#define COGNITION       1.0
#define DIMENSIONS      10
#define E               2.7182818284590452353602874713526625
#define ERROR           10E-8
#define EVALUATIONS     200000
#define FUNCTIONS       1
#define INERTIA         0.75
#define INERTIA_DECAY   0.99
#define INF             1.0e99
#define LEFT            -100.0
#define MUTATION        0.01
#define PI              3.1415926535897932384626433832795029
#define POPULATION      100
#define RANDOM_SEEDS    1000
#define RIGHT           100.0
#define RUNS            30
#define SOCIAL          2.05
#define SPEED_DECAY     0.1

using namespace std;

double random01() {
    return (double) rand() / RAND_MAX;
}

double random(const double left, const double right) {
    return random01() * (right - left) + left;
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
    double copiedX[DIMENSIONS];
    double anotherCopiedX[DIMENSIONS];

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
        memcpy(copiedX, x, dimensions * sizeof(double));
        for (unsigned char i = 0; i < dimensions; i++) {
            x[i] = 0;
            for (unsigned char j = 0; j < dimensions; j++) {
                x[i] += copiedX[j] * rotationData[start + i * dimensions + j];
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
        return apply(x, true, true, 0, 0) + minimum;
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
            copiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            y += functions[i]->apply(&copiedX[newStarts[i]], false, false, 0, 0);
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
            copiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            double z = functions[i]->apply(&copiedX[newStarts[i]], false, false, 0, 0);
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
            copiedX[i] = x[shuffleData[i] - 1];
        }
        double y = 0.0;
        for (unsigned char i = 0; i < functionNumber; i++) {
            y += functions[i]->apply(&copiedX[newStarts[i]], false, false, 0, 0);
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
            memcpy(anotherCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(anotherCopiedX, shiftFlags[i], rotateFlags[i],
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
            memcpy(anotherCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(anotherCopiedX, shiftFlags[i], rotateFlags[i],
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
            memcpy(anotherCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(anotherCopiedX, shiftFlag, rotateFlag, i * dimensions,
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
            memcpy(anotherCopiedX, x, dimensions * sizeof(double));
            y[i] = functions[i]->apply(anotherCopiedX, shiftFlag, rotateFlag, i * dimensions,
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

class PSO {
private:
    const unsigned char functionIndex;
    const unsigned char dimensions{};
    const unsigned int maximumEvaluations{};
    const double left{};
    const double right{};
    const unsigned char populationSize;
    double inertia;
    const double cognition;
    const double social;
    const double inertiaDecay;
    const double speedDecay;
    const double mutation;
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
    FILE *file;

public:
    PSO(const unsigned char functionIndex, const unsigned char dimensions, const unsigned int maximumEvaluations,
        const double left, const double right, const unsigned char populationSize, const double inertia,
        const double cognition, const double social, const double inertiaDecay, const double speedDecay,
        const double mutation, Function *function, FILE *file) : functionIndex(functionIndex), dimensions(dimensions),
                                                                 maximumEvaluations(maximumEvaluations), left(left),
                                                                 right(right), populationSize(populationSize),
                                                                 inertia(inertia), cognition(cognition), social(social),
                                                                 inertiaDecay(inertiaDecay), speedDecay(speedDecay),
                                                                 mutation(mutation), bestEvaluation(INF),
                                                                 checkpointIndex(0), function(function), file(file) {
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

    bool evaluatePopulation() {
        double bestGenerationEvaluation = INF;
        for (unsigned char i = 0; i < populationSize; i++) {
            populationEvaluation[i] = function->apply(population[i]);
            if (populationEvaluation[i] < bestEvaluation) {
                bestEvaluation = populationEvaluation[i];
            }
            if (function->getEvaluations() >= checkpoint) {
                ++checkpointIndex;
                checkpoint = computeCheckpoint();
                printf("Evaluations: %d, best: %.6f\n", function->getEvaluations(), bestEvaluation);
                fprintf(file, "%.6E ", bestEvaluation);
            }
            if (function->getEvaluations() >= maximumEvaluations) {
                printf("Finished after %d evaluations\n", function->getEvaluations());
                fprintf(file, "\n");
                return false;
            }
            if (fabs(populationEvaluation[i] - function->getMinimum()) <= ERROR) {
                printf("~~~ Finished after %d evaluations\n", function->getEvaluations());
                while (checkpointIndex < CHECKPOINTS) {
                    ++checkpointIndex;
                    fprintf(file, "%f ", ERROR);
                }
                fprintf(file, "\n");
                return false;
            }
            if (populationEvaluation[i] < bestGenerationEvaluation) {
                bestGenerationEvaluation = populationEvaluation[i];
                memcpy(bestGenerationIndividual, population[i], dimensions * sizeof(double));
            }
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
        for (unsigned char i = 0; i < populationSize; i++) {
            multiply(dimensions, individualInertia, populationSpeed[i], inertia);
            difference(dimensions, pastIndividualDifference, pastPopulation[i],
                       population[i]);
            multiply(dimensions, individualCognition, pastIndividualDifference,
                     cognition * random01());
            difference(dimensions, bestIndividualDifference, bestGenerationIndividual,
                       population[i]);
            multiply(dimensions, individualSocial, bestIndividualDifference,
                     social * random01());
            generateRandomSpeed();
            sumPopulationSpeed(i);
            for (unsigned char j = 0; j < dimensions; j++) {
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
        inertia *= inertiaDecay;
    }

    void run(unsigned char runIndex) {
        generatePopulation();
        if (!evaluatePopulation()) {
            return;
        }
        memcpy(pastPopulation, population, populationSize * dimensions * sizeof(double));
        memcpy(pastPopulationEvaluation, populationEvaluation, populationSize * sizeof(double));
        while (true) {
            simulatePopulation();
            if (!evaluatePopulation()) {
                break;
            }
            updatePopulation();
        }
    }
};

void printParameters() {
    printf("Dimensions: %d\n", DIMENSIONS);
    printf("Evaluations: %d\n", EVALUATIONS);
    printf("Runs: %d\n", RUNS);
    printf("Population: %d\n", POPULATION);
    printf("Inertia: %.2f\n", INERTIA);
    printf("Cognition: %.2f\n", COGNITION);
    printf("Social: %.2f\n", SOCIAL);
    printf("Inertia decay: %.2f\n", INERTIA_DECAY);
    printf("Speed decay: %.2f\n", SPEED_DECAY);
    printf("Mutation: %.2f\n", MUTATION);
    printf("\n");
}

string getFileName(const char *algorithm, unsigned char function, unsigned char dimensions) {
    string fileName = "../";
    fileName += algorithm;
    fileName += '/';
    fileName += algorithm;
    fileName += '_';
    fileName += to_string(function + 1);
    fileName += '_';
    fileName += to_string(dimensions);
    fileName += ".txt";
    return fileName;
}

int main() {
    printParameters();
    FILE *file = fopen("../input_data/Rand_Seeds.txt", "r");
    double randomSeeds[RANDOM_SEEDS];
    for (double &randomSeed: randomSeeds) {
        fscanf(file, "%lf", &randomSeed);
    }
    fclose(file);
    Functions functions(DIMENSIONS);
    for (unsigned char i = 0; i < FUNCTIONS; i++) {
        file = fopen(getFileName("PSO", i, DIMENSIONS).c_str(), "a");
        for (unsigned char j = 0; j < RUNS; j++) {
            unsigned short randomIndex = (DIMENSIONS / 10 * (i + 1) * RUNS + j + 1) - RUNS;
            randomIndex = (randomIndex % 1000);
            srand(randomSeeds[randomIndex]);
            printf("Am resetat seed-ul\n");
            Function *function = functions.getFunction(i);
            function->resetEvaluations();
            printf("Function %d, run %d, minimum = %.1f:\n", i + 1, j + 1, function->getMinimum());
            PSO pso(i, DIMENSIONS, EVALUATIONS, LEFT, RIGHT,
                    POPULATION, INERTIA, COGNITION, SOCIAL, INERTIA_DECAY,
                    SPEED_DECAY, MUTATION, function, file);
            pso.run(j);
            printf("\n");
        }
        fclose(file);
    }
    return 0;
}