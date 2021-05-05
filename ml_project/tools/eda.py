import matplotlib.pyplot as plt
import pandas as pd

from clfit.data import read_data


def markdown(df):
    return pd.DataFrame(df).to_markdown()


def main():
    path_to_data = "./data/raw/heart.csv"
    data = read_data(path_to_data)
    binary_columns = ["sex", "fbs", "exang"]
    categorical_columns = ["cp", "restecg", "slope", "ca", "thal"]
    target_column = "target"
    numeric_columns = list(
        set(data.columns.tolist()) - set(binary_columns) - set(categorical_columns)
    )
    numeric_columns.remove(target_column)

    data[binary_columns + categorical_columns] = data[
        binary_columns + categorical_columns
    ].astype("category")

    print("## Data example\n")
    print(data.head().to_markdown())
    print()

    print("## General info\n")
    print("```")
    print(data.info())
    print("```\n")

    print("## Statistics\n")
    print("### Numeric\n")
    print(markdown(data.describe()))
    print()

    print("### Categorical\n")
    print(markdown(data.describe(include="category")))
    print()

    print("## Target distribution\n")
    print(data[target_column].value_counts(normalize=True).to_markdown())
    print()

    data[binary_columns + categorical_columns] = data[
        binary_columns + categorical_columns
    ].astype("int")

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(1, 1, 1)
    data.hist(ax=ax)

    fig.tight_layout()
    fname = "histogram.png"
    hist_path = f"report/{fname}"
    fig.savefig(hist_path)

    print("## Histogram\n")
    print(f"![histogram]({fname})")
    print()

    print("## Correlation\n")
    print("### Numeric\n")
    print(data[numeric_columns + [target_column]].corr().to_markdown())
    print()

    print("### Categorical\n")
    print(data[categorical_columns + [target_column]].corr().to_markdown())


if __name__ == "__main__":
    main()
