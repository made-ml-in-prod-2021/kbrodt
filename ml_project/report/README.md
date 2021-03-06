## Data example

|    |   age |   sex |   cp |   trestbps |   chol |   fbs |   restecg |   thalach |   exang |   oldpeak |   slope |   ca |   thal |   target |
|---:|------:|------:|-----:|-----------:|-------:|------:|----------:|----------:|--------:|----------:|--------:|-----:|-------:|---------:|
|  0 |    63 |     1 |    3 |        145 |    233 |     1 |         0 |       150 |       0 |       2.3 |       0 |    0 |      1 |        1 |
|  1 |    37 |     1 |    2 |        130 |    250 |     0 |         1 |       187 |       0 |       3.5 |       0 |    0 |      2 |        1 |
|  2 |    41 |     0 |    1 |        130 |    204 |     0 |         0 |       172 |       0 |       1.4 |       2 |    0 |      2 |        1 |
|  3 |    56 |     1 |    1 |        120 |    236 |     0 |         1 |       178 |       0 |       0.8 |       2 |    0 |      2 |        1 |
|  4 |    57 |     0 |    0 |        120 |    354 |     0 |         1 |       163 |       1 |       0.6 |       2 |    0 |      2 |        1 |

## General info

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   age       303 non-null    int64
 1   sex       303 non-null    category
 2   cp        303 non-null    category
 3   trestbps  303 non-null    int64
 4   chol      303 non-null    int64
 5   fbs       303 non-null    category
 6   restecg   303 non-null    category
 7   thalach   303 non-null    int64
 8   exang     303 non-null    category
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    category
 11  ca        303 non-null    category
 12  thal      303 non-null    category
 13  target    303 non-null    int64
dtypes: category(8), float64(1), int64(5)
memory usage: 17.9 KB
None
```

## Statistics

### Numeric

|       |      age |   trestbps |     chol |   thalach |   oldpeak |     target |
|:------|---------:|-----------:|---------:|----------:|----------:|-----------:|
| count | 303      |   303      | 303      |  303      | 303       | 303        |
| mean  |  54.3663 |   131.624  | 246.264  |  149.647  |   1.0396  |   0.544554 |
| std   |   9.0821 |    17.5381 |  51.8308 |   22.9052 |   1.16108 |   0.498835 |
| min   |  29      |    94      | 126      |   71      |   0       |   0        |
| 25%   |  47.5    |   120      | 211      |  133.5    |   0       |   0        |
| 50%   |  55      |   130      | 240      |  153      |   0.8     |   1        |
| 75%   |  61      |   140      | 274.5    |  166      |   1.6     |   1        |
| max   |  77      |   200      | 564      |  202      |   6.2     |   1        |

### Categorical

|        |   sex |   cp |   fbs |   restecg |   exang |   slope |   ca |   thal |
|:-------|------:|-----:|------:|----------:|--------:|--------:|-----:|-------:|
| count  |   303 |  303 |   303 |       303 |     303 |     303 |  303 |    303 |
| unique |     2 |    4 |     2 |         3 |       2 |       3 |    5 |      4 |
| top    |     1 |    0 |     0 |         1 |       0 |       2 |    0 |      2 |
| freq   |   207 |  143 |   258 |       152 |     204 |     142 |  175 |    166 |

## Target distribution

|    |   target |
|---:|---------:|
|  1 | 0.544554 |
|  0 | 0.455446 |

## Histogram

![histogram](histogram.png)

## Correlation

### Numeric

|          |   trestbps |    oldpeak |     thalach |        chol |       age |     target |
|:---------|-----------:|-----------:|------------:|------------:|----------:|-----------:|
| trestbps |  1         |  0.193216  | -0.0466977  |  0.123174   |  0.279351 | -0.144931  |
| oldpeak  |  0.193216  |  1         | -0.344187   |  0.0539519  |  0.210013 | -0.430696  |
| thalach  | -0.0466977 | -0.344187  |  1          | -0.00993984 | -0.398522 |  0.421741  |
| chol     |  0.123174  |  0.0539519 | -0.00993984 |  1          |  0.213678 | -0.0852391 |
| age      |  0.279351  |  0.210013  | -0.398522   |  0.213678   |  1        | -0.225439  |
| target   | -0.144931  | -0.430696  |  0.421741   | -0.0852391  | -0.225439 |  1         |

### Categorical

|         |         cp |    restecg |      slope |         ca |       thal |    target |
|:--------|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|
| cp      |  1         |  0.0444206 |  0.119717  | -0.181053  | -0.161736  |  0.433798 |
| restecg |  0.0444206 |  1         |  0.0930448 | -0.0720424 | -0.0119814 |  0.13723  |
| slope   |  0.119717  |  0.0930448 |  1         | -0.0801552 | -0.104764  |  0.345877 |
| ca      | -0.181053  | -0.0720424 | -0.0801552 |  1         |  0.151832  | -0.391724 |
| thal    | -0.161736  | -0.0119814 | -0.104764  |  0.151832  |  1         | -0.344029 |
| target  |  0.433798  |  0.13723   |  0.345877  | -0.391724  | -0.344029  |  1        |
