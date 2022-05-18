import csv
import statistics

import numpy


def get_errors(file_name):
    errors = []
    with open(file_name) as file:
        for line in file:
            errors += [[float(error) for error in line.split()]]
    return errors


def get_error_statistics(file_name, function):
    errors = get_errors(file_name)
    last_errors = [run[-1] for run in errors]
    return function, f"{min(last_errors):.3E}", f"{max(last_errors):.3E}", f"{statistics.median(last_errors):.3E}", \
           f"{statistics.mean(last_errors):.3E}", f"{statistics.stdev(last_errors):.3E}"


def main():
    algorithm = "EPRPSO"
    dimensions = 10
    functions = 12
    with open(f"../{algorithm}/errors_{dimensions}.csv", 'w') as file:
        csv_writer = csv.writer(file)
        total_steps = numpy.empty(shape=(0, 6))
        for function in range(1, functions + 1):
            steps = get_error_statistics(f"../{algorithm}/{algorithm}_{function}_{dimensions}.txt", function)
            csv_writer.writerow(steps)
            total_steps = numpy.vstack((total_steps, numpy.array([steps], dtype=float)))
        csv_writer.writerow(numpy.sum(total_steps, axis=0))


if __name__ == '__main__':
    main()
