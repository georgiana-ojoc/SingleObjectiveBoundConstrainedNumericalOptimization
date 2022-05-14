import matplotlib.pyplot as pyplot

from process_results import get_errors


def plot(algorithm, dimensions, functions, errors, evaluations):
    runs = len(errors[0])
    rows = 4
    columns = functions // 4
    figure, axes = pyplot.subplots(rows, columns, figsize=(21, 28))
    for row in range(rows):
        for column in range(columns):
            function = row * columns + column
            for run in errors[function]:
                axes[row, column].plot(evaluations, run)
                axes[row, column].set_title(f"Errors of function {function + 1} \n "
                                            f"on {dimensions} dimensions for {runs} runs")
    for axis in axes.flat:
        axis.set(xlabel="k", ylabel="error")
    pyplot.savefig(f"../{algorithm}/plots_{dimensions}.png")


def main():
    algorithm = "EPRPSO"
    dimensions = 20
    functions = 12
    errors = []
    evaluations = list(range(1, 17))
    for function in range(1, functions + 1):
        errors += [get_errors(f"../{algorithm}/{algorithm}_{function}_{dimensions}.txt")]
    plot(algorithm, dimensions, functions, errors, evaluations)


if __name__ == '__main__':
    main()
