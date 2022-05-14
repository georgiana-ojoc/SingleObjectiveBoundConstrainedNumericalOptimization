import matplotlib.pyplot as pyplot

from process_results import get_errors


def plot(first_algorithm, second_algorithm, dimensions, function, errors, evaluations):
    runs = len(errors[0])
    rows = 1
    columns = 2
    figure, axes = pyplot.subplots(rows, columns, figsize=(10, 6))
    text = f"using {first_algorithm}"
    for column in range(columns):
        for run in errors[column]:
            axes[column].plot(evaluations, run)
            axes[column].set_title(f"Errors of function {function} \n "
                                   f"on {dimensions} dimensions for {runs} runs \n {text}")
        text = f"using {second_algorithm}"
    for axis in axes.flat:
        axis.set(xlabel="k", ylabel="error")
    pyplot.savefig(f"../plots_{dimensions}_{function}_{first_algorithm}_{second_algorithm}.png")


def main():
    first_algorithm = "PRPSO"
    second_algorithm = "EPRPSO"
    dimensions = 20
    function = 7
    errors = []
    evaluations = list(range(1, 17))
    errors += [get_errors(f"../{first_algorithm}/{first_algorithm}_{function}_{dimensions}.txt")]
    errors += [get_errors(f"../{second_algorithm}/{second_algorithm}_{function}_{dimensions}.txt")]
    plot(first_algorithm, second_algorithm, dimensions, function, errors, evaluations)


if __name__ == '__main__':
    main()
