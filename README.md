# From Raw Dataset to Machine-Learning-Ready Data

> A detailed, practical guide that explains every concept in three steps: **What is it? → Why do we use it? → How does it work? (with a hands-on example)**.

**Prerequisites:** Python basics (variables, lists, conditions, loops, functions)
**Tools:** Jupyter Notebook or Google Colab, NumPy, Pandas, Matplotlib/Seaborn, Scikit-learn

---

## The Big Picture (Pipeline)

```
Raw dataset  →  Exploration (EDA)  →  Cleaning  →  Preprocessing  →  ML model
    CSV          head/info/plots      NaN/dupes     split/encode/scale
```

| Block | Topic | Goal |
|---|---|---|
| **A** | NumPy | Foundation for fast numerical computing |
| **B** | Pandas | Working with tables and data |
| **C** | Cleaning + EDA | Understanding and cleaning the data |
| **D** | Preprocessing | Preparing data for the model without leakage |

### Standard imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# Block A — NumPy

**NumPy** is Python's numerical computing library. Pandas, Scikit-learn and many other libraries are built on top of it.

---

## A1. `ndarray`

### What is it?
`ndarray` (N-dimensional array) is NumPy's core object. It looks like a Python list, but stores elements of **a single type** in contiguous memory.

### Why?
- Much faster than lists for computation (implemented in C).
- Uses less memory.
- Supports math on whole arrays without loops.
- ML data (images, tables, weights) is all represented as arrays.

### How?

```python
import numpy as np

# Create from a list
a = np.array([1, 2, 3, 4])
print(a)              # [1 2 3 4]

# 2D array (2 rows × 3 columns)
m = np.array([[1, 2, 3],
              [4, 5, 6]])

# Built-in constructors
np.zeros((2, 3))      # array of zeros
np.ones((2, 2))       # array of ones
np.arange(0, 10, 2)   # [0 2 4 6 8]
np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]
np.random.rand(3, 3)  # random numbers between 0 and 1
```

**Speed comparison with a list:**

```python
import time
lst = list(range(1_000_000))
arr = np.arange(1_000_000)

t = time.time(); [x * 2 for x in lst];  print("list :", time.time() - t)
t = time.time(); arr * 2;               print("numpy:", time.time() - t)
# NumPy is typically tens of times faster
```

---

## A2. `shape`

### What is it?
An attribute returning the **dimensions** of the array as a tuple: `(rows, columns, ...)`.

### Why?
Most ML bugs come from shape mismatches. Checking the shape is the first thing you should do.

### How?

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])

print(m.shape)    # (2, 3)  → 2 rows, 3 columns
print(m.size)     # 6       → total number of elements
print(len(m))     # 2       → length of the first axis
```

---

## A3. `ndim`

### What is it?
The number of **dimensions** (axes) of the array.

### Why?
To distinguish scalar (0D), vector (1D), matrix (2D) and tensor (3D+). For example, a color image is 3D (height × width × channels).

### How?

```python
a = np.array(5)                     # scalar
b = np.array([1, 2, 3])             # vector
c = np.array([[1, 2], [3, 4]])      # matrix
d = np.zeros((28, 28, 3))           # RGB image

print(a.ndim, b.ndim, c.ndim, d.ndim)   # 0 1 2 3
```

> Difference between `shape` and `ndim`: `ndim = len(shape)`.

---

## A4. `dtype`

### What is it?
The **data type** of the array's elements (`int64`, `float64`, `bool`, `<U5` for strings, ...).

### Why?
- Affects **precision** and **memory usage**.
- Some models require `float` input.
- Wrong conversions can lose information (e.g. 3.7 → 3).

### How?

```python
a = np.array([1, 2, 3])
print(a.dtype)                  # int64

b = np.array([1.5, 2.5])
print(b.dtype)                  # float64

# Specify the type at creation
c = np.array([1, 2, 3], dtype=np.float32)

# Convert the type
d = b.astype(int)
print(d)                        # [1 2]  ← decimals were dropped!

# Save memory
big = np.ones(1_000_000, dtype=np.float64)
small = big.astype(np.float32)
print(big.nbytes, small.nbytes) # 8000000  4000000
```

---

## A5. `reshape`

### What is it?
Changing the **shape** of an array without changing its data (the total number of elements must stay the same).

### Why?
Models expect specific shapes. For example, scikit-learn expects `X` as `(n_samples, n_features)`. Images are flattened before being fed to some models.

### How?

```python
a = np.arange(12)          # [0 1 2 ... 11]  shape = (12,)

b = a.reshape(3, 4)        # 3 rows × 4 columns
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1 = "figure it out for me"
c = a.reshape(2, -1)       # (2, 6)
d = a.reshape(-1, 1)       # (12, 1) ← a single column (common with sklearn)

# Flatten
b.flatten()                # back to (12,)
```

---

## A6. Indexing

### What is it?
Accessing a specific element by its position. Indexing starts at **0**; negative indices count from the end.

### Why?
To read or modify a specific value.

### How?

```python
a = np.array([10, 20, 30, 40])
print(a[0])       # 10
print(a[-1])      # 40  (last element)

m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(m[1, 2])    # 6  (row 1, column 2)
m[0, 0] = 100     # modify a value

# Fancy indexing: several positions at once
print(a[[0, 2]])  # [10 30]
```

---

## A7. Slicing

### What is it?
Extracting a **part** of an array using `start:stop:step` (the stop index is **excluded**).

### Why?
To select specific rows/columns, separate features from the target, or take a sample.

### How?

```python
a = np.arange(10)        # [0 1 2 3 4 5 6 7 8 9]
print(a[2:5])            # [2 3 4]
print(a[:3])             # [0 1 2]
print(a[7:])             # [7 8 9]
print(a[::2])            # [0 2 4 6 8]
print(a[::-1])           # reversed array

m = np.arange(12).reshape(3, 4)
print(m[:, 1])           # entire column 1 → [1 5 9]
print(m[0, :])           # entire row 0
print(m[:2, 1:3])        # 2 rows × 2 columns

# ML example: split X and y from an array
data = np.array([[5, 3, 0],
                 [6, 2, 1],
                 [7, 1, 0]])
X = data[:, :-1]         # all columns except the last
y = data[:, -1]          # last column
```

> **Warning:** a slice is a **view** (a window onto the same data); modifying it modifies the original. Use `.copy()` for an independent copy.

```python
b = a[2:5].copy()
```

---

## A8. Boolean Masks

### What is it?
An array of `True/False` values used to select the elements that satisfy a condition.

### Why?
To filter data without loops (e.g. keep all values greater than 50, or replace outliers).

### How?

```python
a = np.array([5, 12, 7, 20, 3, 15])

mask = a > 10
print(mask)         # [False  True False  True False  True]
print(a[mask])      # [12 20 15]

# Combined conditions (use & | ~ with parentheses)
print(a[(a > 5) & (a < 20)])    # [12  7 15]
print(a[~(a > 10)])             # [5 7 3]

# Modify using a mask
a[a > 10] = 10
print(a)            # [ 5 10  7 10  3 10]

# Count elements satisfying a condition
print((a > 5).sum())

# np.where: condition → two values
b = np.array([1, -2, 3, -4])
print(np.where(b > 0, "positive", "negative"))
```

---

## A9. Vectorized Operations

### What is it?
Applying an operation to **every element of the array at once**, without writing a `for` loop.

### Why?
- Shorter, clearer code.
- Much faster (runs in C).
- It's the natural way of thinking in ML.

### How?

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)         # [11 22 33 44]
print(a * b)         # [10 40 90 160]  (element-wise)
print(a ** 2)        # [ 1  4  9 16]
print(np.sqrt(a))    # square root of each element
print(np.exp(a))

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)         # or np.dot(A, B)

# Loop vs vectorized
# ❌ slow
result = []
for x in a:
    result.append(x * 2)
# ✅ fast
result = a * 2
```

**ML example — computing MSE:**

```python
y_true = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.5, 8.0])
mse = np.mean((y_true - y_pred) ** 2)
print(mse)           # 0.5
```

---

## A10. Statistics

### What is it?
Functions that summarize an array with one or more numbers: mean, median, standard deviation, sum, min/max, ...

### Why?
To quickly understand the data distribution. They are the basis of standardization and outlier detection.

### How?

```python
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])

print(np.mean(a))        # 5.0   mean
print(np.median(a))      # 4.5   median
print(np.std(a))         # 2.0   standard deviation
print(np.var(a))         # 4.0   variance
print(a.min(), a.max())  # 2 9
print(a.sum())           # 40
print(np.percentile(a, 75))

# On a 2D array: axis sets the direction
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.mean(axis=0))    # [2.5 3.5 4.5]  ← mean of each column
print(m.mean(axis=1))    # [2. 5.]        ← mean of each row
print(m.argmax())        # index of the largest value

# Handling NaN
x = np.array([1, np.nan, 3])
print(np.mean(x))        # nan
print(np.nanmean(x))     # 2.0
```

> **Axis rule:** `axis=0` = collapse the **rows** (one result per column); `axis=1` = collapse the **columns** (one result per row).

---

## A11. Broadcasting

### What is it?
A rule that lets NumPy perform operations between arrays of **different shapes** by automatically "stretching" the smaller one (without actually copying data).

### Why?
To apply an operation to every row/column easily, such as subtracting each column's mean (the basis of standardization).

### How?

**The rule:** dimensions are compared from right to left; they're compatible if they are **equal** or one of them is **1**.

```python
# scalar × array
a = np.array([1, 2, 3])
print(a + 10)            # [11 12 13]

# 2D array + vector
m = np.array([[1, 2, 3],
              [4, 5, 6]])     # (2, 3)
v = np.array([10, 20, 30])    # (3,)  → stretched to (2, 3)
print(m + v)
# [[11 22 33]
#  [14 25 36]]

# column + row → full matrix
col = np.array([[1], [2], [3]])   # (3, 1)
row = np.array([10, 20, 30])      # (3,)
print(col + row)                  # (3, 3)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

**ML example — manual standardization:**

```python
X = np.array([[1., 200.],
              [2., 300.],
              [3., 400.]])

mean = X.mean(axis=0)      # (2,)
std  = X.std(axis=0)       # (2,)
X_scaled = (X - mean) / std   # broadcasting: (3,2) - (2,) → (3,2)
print(X_scaled.mean(axis=0))  # ≈ [0 0]
```

---

# Block B — Pandas

**Pandas** is a library for working with tabular data. Its two core structures are:
- **Series**: a single column (1D) with an index.
- **DataFrame**: a full table (2D) with rows and columns.

We'll use this DataFrame in the examples below:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Ali", "Sara", "Omar", "Lina", "Youssef", "Nada"],
    "age":    [25, 30, 22, 35, 28, 40],
    "city":   ["Rabat", "Casa", "Rabat", "Fes", "Casa", "Rabat"],
    "salary": [5000, 7000, 4000, 9000, 6000, 10000],
    "dept":   ["IT", "HR", "IT", "Finance", "HR", "IT"],
})
```

---

## B1. `read_csv`

### What is it?
A function that reads a CSV file into a DataFrame.

### Why?
Most datasets (Kaggle, UCI, ...) come as CSV. This is the starting point of nearly every project.

### How?

```python
df = pd.read_csv("data.csv")

# Useful options
df = pd.read_csv(
    "data.csv",
    sep=";",                  # separator (default ",")
    encoding="utf-8",         # or "latin-1" for encoding problems
    usecols=["age", "city"],  # read only specific columns
    na_values=["?", "N/A"],   # treat these values as NaN
    parse_dates=["date"],     # convert a column to datetime
    nrows=1000,               # read only the first 1000 rows
)

# Directly from a URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
titanic = pd.read_csv(url)

# In Google Colab: upload a file
# from google.colab import files
# files.upload()
```

---

## B2. `head` and `tail`

### What is it?
Displays the **first** (`head`) or **last** (`tail`) n rows (default 5).

### Why?
The first visual check: was the data read correctly? Do the columns make sense? Does the end of the file contain total rows or corrupted data?

### How?

```python
df.head()       # first 5 rows
df.head(3)      # first 3
df.tail(2)      # last 2 rows
df.sample(3)    # 3 random rows (very useful if the data is sorted)
```

---

## B3. `shape`

### What is it?
A tuple: `(number of rows, number of columns)`.

### Why?
To know the size of your data and track changes after each cleaning step (how many rows did we drop?).

### How?

```python
print(df.shape)       # (6, 5)
n_rows, n_cols = df.shape

before = df.shape[0]
df2 = df.drop_duplicates()
print("Removed:", before - df2.shape[0])
```

---

## B4. `columns`

### What is it?
The list of column names.

### Why?
To see what's available, rename columns, and standardize names (spaces, capital letters).

### How?

```python
print(df.columns)             # Index(['name','age','city','salary','dept'])
print(df.columns.tolist())

# Rename
df = df.rename(columns={"dept": "department"})

# Clean names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select columns
df[["name", "age"]]           # several columns (list inside brackets)
df["age"]                     # a single column (Series)
```

---

## B5. `dtypes`

### What is it?
The data type of each column (`int64`, `float64`, `object`, `bool`, `datetime64`, `category`).

### Why?
To catch errors early: a numeric column read as text (`object`) because of a symbol like `"$100"`, or a date read as a string.

### How?

```python
print(df.dtypes)
# name       object
# age         int64
# city       object
# salary      int64

# Select columns by type
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns
```

---

## B6. `info`

### What is it?
A technical summary: number of rows, columns, type, number of **non-null** values per column, and memory usage.

### Why?
The fastest way to spot **missing values** (non-null count lower than the total row count) and wrong types.

### How?

```python
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6 entries, 0 to 5
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    6 non-null      object
#  1   age     6 non-null      int64
#  ...
```

> If you see `Age 714 non-null` out of 891 rows → 177 values are missing.

---

## B7. `describe`

### What is it?
A **statistical** summary of numeric columns: count, mean, std, min, 25%, 50%, 75%, max.

### Why?
To understand scale and distribution and spot outliers (e.g. max much larger than the 75th percentile → possible outlier).

### How?

```python
df.describe()
#              age        salary
# count   6.000000      6.000000
# mean   30.000000   6833.333333
# std     6.531973   2316.607...
# min    22.000000   4000.000000
# 50%    29.000000   6500.000000
# max    40.000000  10000.000000

df.describe(include="object")   # text columns: count, unique, top, freq
df.describe(include="all")      # everything
```

---

## B8. `loc` and `iloc`

### What is it?
Tools for selecting rows/columns:
- **`loc`**: by **label** — the end is **included**.
- **`iloc`**: by integer **position** — the end is **excluded**.

### Why?
To select precise parts of a table: specific rows, specific columns, or both.

### How?

```python
# iloc: by position
df.iloc[0]               # first row
df.iloc[0:3]             # rows 0, 1, 2
df.iloc[0, 1]            # row 0, column 1
df.iloc[:, 0:2]          # all rows, first two columns
df.iloc[[0, 2], [1, 3]]  # specific rows and columns

# loc: by label
df.loc[0, "name"]                  # 'Ali'
df.loc[0:2, "name":"city"]         # ⚠️ end is included (0, 1, 2)
df.loc[df["age"] > 28, ["name", "age"]]   # condition + columns

# Modify
df.loc[df["name"] == "Ali", "salary"] = 5500
```

| | `loc` | `iloc` |
|---|---|---|
| Based on | Labels/index | Integer position |
| `a:b` | Includes b | Excludes b |
| Accepts a boolean condition | Yes | Yes (values only) |

---

## B9. Filtering

### What is it?
Selecting the rows that satisfy a condition (Boolean Indexing), the same idea as masks in NumPy.

### Why?
To analyze a subset (IT employees) or exclude bad data.

### How?

```python
df[df["age"] > 28]                         # single condition
df[(df["age"] > 25) & (df["city"] == "Rabat")]   # AND
df[(df["dept"] == "IT") | (df["dept"] == "HR")]  # OR
df[~(df["city"] == "Casa")]                # NOT

df[df["city"].isin(["Rabat", "Fes"])]      # within a list
df[df["name"].str.startswith("A")]         # strings
df[df["salary"].between(5000, 8000)]       # range

# query: cleaner syntax
df.query("age > 25 and city == 'Rabat'")
```

> Use `&` and `|` (not `and`/`or`) with parentheses around each condition.

---

## B10. Creating New Columns (Feature Creation)

### What is it?
Adding a new column derived from existing columns.

### Why?
**Feature Engineering**: new features often improve model performance more than changing the algorithm.

### How?

```python
# Arithmetic
df["salary_k"] = df["salary"] / 1000
df["annual_salary"] = df["salary"] * 12

# Conditions
df["is_senior"] = df["age"] > 30
df["level"] = np.where(df["salary"] > 6000, "high", "low")

# Binning into categories
df["age_group"] = pd.cut(df["age"],
                         bins=[0, 25, 35, 100],
                         labels=["young", "adult", "senior"])

# apply: custom function
df["name_len"] = df["name"].apply(len)
df["city_upper"] = df["city"].str.upper()

# Drop a column
df = df.drop(columns=["name_len"])
```

---

## B11. `sort_values`

### What is it?
Sorting rows by one or more columns.

### Why?
To extract the top/bottom values (Top 10) and organize time series.

### How?

```python
df.sort_values("salary")                          # ascending
df.sort_values("salary", ascending=False)         # descending
df.sort_values(["dept", "salary"], ascending=[True, False])  # several columns

df.nlargest(3, "salary")                          # top 3
df.sort_values("age").reset_index(drop=True)      # renumber the index
```

---

## B12. `value_counts`

### What is it?
Counts how many times each unique value appears in a column.

### Why?
- Check the distribution of categorical variables.
- Detect **class imbalance** in the target y.
- Spot spelling errors ("Rabat" vs "rabat").

### How?

```python
df["city"].value_counts()
# Rabat    3
# Casa     2
# Fes      1

df["city"].value_counts(normalize=True)   # proportions
# Rabat    0.50 ...

df["city"].value_counts(dropna=False)     # also shows NaN
df["city"].nunique()                      # number of unique values
df["city"].unique()                       # the unique values

# Direct plot
df["city"].value_counts().plot(kind="bar")
```

---

## B13. `groupby`

### What is it?
Splitting the data into groups by a column's value, then applying an aggregation to each group (Split → Apply → Combine).

### Why?
To answer questions like: average salary per department? Number of employees per city?

### How?

```python
df.groupby("dept")["salary"].mean()
# Finance    9000.0
# HR         6500.0
# IT         6333.3

# Several aggregations
df.groupby("dept")["salary"].agg(["mean", "min", "max", "count"])

# Several grouping columns
df.groupby(["dept", "city"])["salary"].sum()

# Different aggregations for different columns
df.groupby("dept").agg(
    avg_salary=("salary", "mean"),
    max_age=("age", "max"),
    n=("name", "count"),
).reset_index()

# transform: keeps the original shape (useful for filling NaN with the group mean)
df["dept_avg"] = df.groupby("dept")["salary"].transform("mean")
```

---

## B14. `merge`

### What is it?
Combining two tables based on a common column (like SQL JOIN).

### Why?
Real-world data is spread across several tables (customers + orders + products). To build a single training dataset you must join them.

### How?

```python
employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name":   ["Ali", "Sara", "Omar", "Lina"],
    "dept_id":[10, 20, 10, 30],
})
departments = pd.DataFrame({
    "dept_id":  [10, 20, 40],
    "dept_name":["IT", "HR", "Legal"],
})

# inner: only matches
pd.merge(employees, departments, on="dept_id", how="inner")

# left: all rows of the left table
pd.merge(employees, departments, on="dept_id", how="left")
#    emp_id  name  dept_id dept_name
# 0       1   Ali       10        IT
# 1       2  Sara       20        HR
# 2       3  Omar       10        IT
# 3       4  Lina       30       NaN   ← no dept 30

# right / outer
pd.merge(employees, departments, on="dept_id", how="outer")

# Different column names
pd.merge(a, b, left_on="id", right_on="user_id")

# Stacking vertically
pd.concat([df1, df2], axis=0, ignore_index=True)
```

| `how` | Result |
|---|---|
| `inner` | Matching rows only |
| `left` | All left rows + matches |
| `right` | All right rows + matches |
| `outer` | Everything |

---

# Block C — Cleaning & Exploration (Cleaning & EDA)

> **EDA = Exploratory Data Analysis**: understanding the data before any modeling.
> **Cleaning**: fixing problems such as missing values, duplicates and wrong types.

We'll use this "dirty" example:

```python
dirty = pd.DataFrame({
    "id":     [1, 2, 3, 3, 4, 5],
    "age":    [25, np.nan, 30, 30, 200, 28],
    "city":   ["Rabat", "Casa", None, None, "Fes", "rabat "],
    "date":   ["2024-01-05", "2024-02-10", "2024-03-15",
               "2024-03-15", "2024-04-20", "2024-05-25"],
    "salary": ["5000", "7000", "6000", "6000", "9000", "4000"],
})
```

---

## C1. `isna`

### What is it?
Returns a DataFrame of True/False flags marking missing values (`NaN`/`None`). Its alias is `isnull()`.

### Why?
Most ML algorithms **don't accept NaN** and raise an error. You also need to know the size of the problem before choosing a fix.

### How?

```python
dirty.isna()                       # True/False table
dirty.isna().sum()                 # missing count per column
# age     1
# city    2

(dirty.isna().mean() * 100).round(1)   # percentage

dirty[dirty["age"].isna()]         # rows where age is missing
dirty.dropna()                     # drop rows containing NaN
dirty.dropna(subset=["age"])       # only based on a specific column

# Visualize missing values
sns.heatmap(dirty.isna(), cbar=False)
plt.show()
```

> Rule of thumb: if a column is **>60%** missing, consider dropping it. If only a little is missing, fill it (impute).

---

## C2. `duplicated` and `drop_duplicates`

### What is it?
- `duplicated()`: flags duplicate rows (True for the second copy onward).
- `drop_duplicates()`: removes them.

### Why?
Duplicates bias the model and cause **data leakage** (the same row in both train and test makes performance look better than it is).

### How?

```python
dirty.duplicated()                 # Series of True/False
dirty.duplicated().sum()           # number of duplicates → 1
dirty[dirty.duplicated(keep=False)]# show all copies

clean = dirty.drop_duplicates()    # drop fully identical rows

# Based on specific columns
clean = dirty.drop_duplicates(subset=["id"], keep="first")
# keep: "first" | "last" | False (drop all)
```

---

## C3. `fillna`

### What is it?
Fills missing values with a chosen value.

### Why?
To keep rows instead of dropping them (dropping too many reduces the data and weakens the model).

### How?

```python
# Numeric → mean or median (median is better with outliers)
dirty["age"] = dirty["age"].fillna(dirty["age"].median())

# Categorical → mode (most frequent value) or "Unknown"
dirty["city"] = dirty["city"].fillna(dirty["city"].mode()[0])
dirty["city"] = dirty["city"].fillna("Unknown")

# Constant
dirty["age"].fillna(0)

# Forward/backward fill (time series)
dirty["age"].ffill()
dirty["age"].bfill()

# Fill with the group mean
dirty["age"] = dirty["age"].fillna(
    dirty.groupby("city")["age"].transform("median")
)
```

> **Important warning:** in an ML project, compute the mean/median from **train only** (see Data Leakage in Block D). Using `fillna` on the whole dataset before splitting leaks information from the test set.

---

## C4. `astype`

### What is it?
Converting a column to another type.

### Why?
Numeric columns read as text can't be used in calculations, and categories stored as `category` save memory.

### How?

```python
print(dirty["salary"].dtype)            # object (text!)
dirty["salary"] = dirty["salary"].astype(int)
dirty["id"]     = dirty["id"].astype("category")
dirty["age"]    = dirty["age"].astype("float32")

# When some values can't be converted (like "N/A" or "$100")
s = pd.Series(["10", "20", "abc", "40"])
pd.to_numeric(s, errors="coerce")       # "abc" → NaN

# Clean symbols before converting
prices = pd.Series(["$1,200", "$850"])
prices.str.replace("[$,]", "", regex=True).astype(float)
```

---

## C5. `to_datetime`

### What is it?
Converting date strings into the `datetime64` type.

### Why?
- Enables time-based sorting and date arithmetic.
- Lets you extract useful features for the model (month, weekday, year).

### How?

```python
dirty["date"] = pd.to_datetime(dirty["date"])

# With an explicit format
pd.to_datetime("25/12/2024", format="%d/%m/%Y")

# Ignore errors
pd.to_datetime(dirty["date"], errors="coerce")

# Extract components (.dt accessor)
dirty["year"]       = dirty["date"].dt.year
dirty["month"]      = dirty["date"].dt.month
dirty["dayofweek"]  = dirty["date"].dt.dayofweek    # 0 = Monday
dirty["is_weekend"] = dirty["date"].dt.dayofweek >= 5

# Time difference
today = pd.Timestamp("2024-12-31")
dirty["days_since"] = (today - dirty["date"]).dt.days
```

---

## C6. Cleaning Text and Outliers

### What is it?
Fixing inconsistencies in text (spaces, upper/lower case) and illogical values (outliers).

### Why?
`"Rabat"` and `"rabat "` would be treated as two different categories, and `age = 200` distorts the mean.

### How?

```python
# Standardize text
dirty["city"] = dirty["city"].str.strip().str.lower().str.title()

# Detect outliers with the IQR method
Q1 = dirty["age"].quantile(0.25)
Q3 = dirty["age"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = dirty[(dirty["age"] < lower) | (dirty["age"] > upper)]

# Handle: drop or clip
dirty = dirty[(dirty["age"] >= lower) & (dirty["age"] <= upper)]
# or
dirty["age"] = dirty["age"].clip(lower, upper)

# Logical rules
dirty = dirty[dirty["age"].between(0, 120)]
```

---

## C7. Distributions

### What is it?
The shape of how a variable's values are spread: where are they concentrated? Are they symmetric (normal)? Skewed?

### Why?
- Determines the right treatment (log transform for skewed data, median instead of mean).
- Reveals outliers.
- Some models assume a normal distribution.

### How?

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram (numeric variable)
sns.histplot(df["salary"], kde=True, bins=10)
plt.title("Salary Distribution")
plt.show()

# Boxplot: shows median and outliers
sns.boxplot(x=df["salary"])
plt.show()

# Skewness
print(df["salary"].skew())      # > 0 : long right tail

# Log transform to reduce skew
df["log_salary"] = np.log1p(df["salary"])

# Categorical variable: countplot
sns.countplot(data=df, x="city")
plt.show()

# All histograms at once
df.hist(figsize=(10, 6), bins=15)
plt.show()
```

---

## C8. Comparing Variables

### What is it?
Analyzing the **relationship** between two or more variables: numeric–numeric, categorical–numeric, categorical–categorical.

### Why?
To discover features useful for predicting the target, and to spot redundant (highly correlated) features.

### How?

```python
# 1) numeric ↔ numeric: scatter + correlation
sns.scatterplot(data=df, x="age", y="salary")
plt.show()

corr = df[["age", "salary"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
# +1 strong positive | 0 no relation | -1 negative

# 2) categorical ↔ numeric: boxplot / barplot / groupby
sns.boxplot(data=df, x="dept", y="salary")
plt.show()
df.groupby("dept")["salary"].mean().plot(kind="bar")

# 3) categorical ↔ categorical: crosstab
pd.crosstab(df["city"], df["dept"])
sns.heatmap(pd.crosstab(df["city"], df["dept"]), annot=True)

# 4) overall view
sns.pairplot(df[["age", "salary"]])
plt.show()

# 5) effect on the target (Titanic example)
# titanic.groupby("sex")["survived"].mean()
```

> Correlation **does not imply** causation.

---

## C9. Simple Visualizations (Matplotlib / Seaborn)

### What is it?
Plots that help you understand data visually.

### Why?
The eye catches patterns and anomalies far faster than tables.

### How? (choosing the right plot)

| Goal | Plot | Code |
|---|---|---|
| Numeric distribution | Histogram | `sns.histplot(df["x"])` |
| Outliers | Boxplot | `sns.boxplot(x=df["x"])` |
| Counting categories | Countplot | `sns.countplot(data=df, x="c")` |
| Relationship of two numerics | Scatter | `sns.scatterplot(data=df, x="a", y="b")` |
| Correlations | Heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True)` |
| Change over time | Line | `df.plot(x="date", y="v")` |

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df["salary"], kde=True, ax=axes[0])
axes[0].set_title("Distribution")

sns.boxplot(data=df, x="dept", y="salary", ax=axes[1])
axes[1].set_title("Salary by Dept")

sns.scatterplot(data=df, x="age", y="salary", hue="dept", ax=axes[2])
axes[2].set_title("Age vs Salary")

plt.tight_layout()
plt.show()
```

---

## Complete Cleaning Example

```python
df = pd.read_csv("data.csv")

# 1. Explore
df.info(); print(df.describe()); print(df.isna().sum())

# 2. Duplicates
df = df.drop_duplicates()

# 3. Types
df["date"]   = pd.to_datetime(df["date"], errors="coerce")
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

# 4. Text
df["city"] = df["city"].str.strip().str.title()

# 5. Logical outliers
df = df[df["age"].between(0, 120) | df["age"].isna()]

# 6. Final check
print(df.shape); print(df.isna().sum())
```

> Note: **statistical** NaN filling (mean/median) is postponed until after `train_test_split` in ML projects, as we'll see next.

---

# Block D — Preprocessing for AI

We'll use this example throughout this section:

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    "age":    [25, 32, np.nan, 41, 29, 55, 38, np.nan, 47, 23],
    "salary": [3000, 5200, 4100, np.nan, 3600, 9000, 6100, 4800, 7500, 2900],
    "city":   ["Rabat","Casa","Fes","Casa","Rabat","Fes","Casa","Rabat","Fes","Rabat"],
    "edu":    ["BAC","Master","Licence","Master","BAC","PhD","Master","Licence","PhD","BAC"],
    "bought": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],   # the target
})
```

---

## D1. Separating X and y

### What is it?
Separating the **features** (X, the inputs) from the **target** (y, what we want to predict).

### Why?
The model learns the relationship `y = f(X)`. If the target stays inside X, that's **direct leakage** (the model "cheats").

### How?

```python
X = data.drop(columns=["bought"])   # all columns except the target
y = data["bought"]

print(X.shape, y.shape)             # (10, 4) (10,)
```

> Also make sure to drop columns like `id`, or columns derived from the target.

---

## D2. `train_test_split`

### What is it?
Splitting the data into:
- **Train**: to train the model (and learn any transformation parameters).
- **Test**: to evaluate the model on data it has **never seen**.

### Why?
To measure the model's ability to **generalize**. Evaluating on the training data itself gives a misleading picture (overfitting).

### How?

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # reproducibility
    stratify=y,          # keep class proportions in y (for classification)
)

print(X_train.shape, X_test.shape)   # (8, 4) (2, 4)
print(y_train.value_counts(normalize=True))
```

| Parameter | Purpose |
|---|---|
| `test_size` | Test proportion (usually 0.2 – 0.3) |
| `random_state` | Reproducible results |
| `stratify=y` | Preserves relative class balance |
| `shuffle=False` | For time series (don't shuffle the order) |

---

## D3. Categorical Encoding

### What is it?
Converting text/categorical columns into **numbers**, because models only understand numbers.

### Why?
`"Rabat"` can't be multiplied by a weight. The chosen method affects performance.

### How?

**1) One-Hot Encoding** — for categories **without order** (city, color):

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[["city"]])                       # learn categories from train only
train_city = ohe.transform(X_train[["city"]])
test_city  = ohe.transform(X_test[["city"]])

print(ohe.get_feature_names_out())
# ['city_Casa' 'city_Fes' 'city_Rabat']
# Rabat → [0, 0, 1]
```

**2) Ordinal Encoding** — for categories **with an order** (education level):

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[["BAC", "Licence", "Master", "PhD"]])
oe.fit(X_train[["edu"]])
oe.transform(X_test[["edu"]])      # BAC→0, Licence→1, Master→2, PhD→3
```

**3) Label Encoding** — for the target y only:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(["cat", "dog", "cat"])   # [0 1 0]
```

> Don't use sequential numbers (Rabat=0, Casa=1, Fes=2) for an unordered variable; the model would assume Fes > Casa > Rabat.
> `handle_unknown="ignore"` prevents a crash if a new category appears in test.

---

## D4. Imputation (Filling Missing Values)

### What is it?
Filling missing values with a statistic **learned from train**, then applied to both train and test.

### Why?
- Models don't accept NaN.
- Using `SimpleImputer` stores the learned value (from train) and reuses it on new data (e.g. in production).

### How?

```python
from sklearn.impute import SimpleImputer

num_cols = ["age", "salary"]

imputer = SimpleImputer(strategy="median")    # mean | median | most_frequent | constant
imputer.fit(X_train[num_cols])                # ← learn the median from train only

print(imputer.statistics_)                    # the stored medians

X_train_num = imputer.transform(X_train[num_cols])
X_test_num  = imputer.transform(X_test[num_cols])   # same values as train!

# For categorical
cat_imputer = SimpleImputer(strategy="most_frequent")

# Advanced
from sklearn.impute import KNNImputer
KNNImputer(n_neighbors=3)
```

| Strategy | When? |
|---|---|
| `mean` | Symmetric distribution without outliers |
| `median` | Skewed distribution / outliers present |
| `most_frequent` | Categorical |
| `constant` | A fixed value such as `"Unknown"` |

---

## D5. Standardization and Normalization (Scaling)

### What is it?
Putting numeric features on a **similar scale**:
- **Standardization (StandardScaler):** `z = (x - mean) / std` → mean 0, std 1.
- **Normalization (MinMaxScaler):** `x' = (x - min) / (max - min)` → squeezes values into [0, 1].

### Why?
`age` (20–60) and `salary` (3000–9000): algorithms based on **distances or gradient descent** (KNN, SVM, Logistic/Linear Regression, Neural Networks, PCA) will be dominated by the feature with larger values. **Decision trees and Random Forests** don't need scaling.

### How?

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
scaler.fit(X_train_num)                  # learns mean and std from train
print(scaler.mean_, scaler.scale_)

X_train_scaled = scaler.transform(X_train_num)
X_test_scaled  = scaler.transform(X_test_num)   # with the training mean/std

# Normalization
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train_num)      # fit + transform on train
X_test_mm  = mm.transform(X_test_num)           # transform only!
```

| | StandardScaler | MinMaxScaler |
|---|---|---|
| Formula | (x − μ) / σ | (x − min) / (max − min) |
| Range | Unbounded | [0, 1] |
| Sensitive to outliers | Moderate | Very high |
| Usage | Most common | Neural networks / images |

---

## D6. Preventing Data Leakage — The Most Important Topic

### What is it?
**Information from the test data (or from the future/target)** reaching the training stage, making the model look better than it really is.

### Why?
The model is deployed on completely **new data**. If test information leaks in during preparation, your experimental performance (say 98%) is **fake**, then collapses in reality (70%).

### The golden rule

> **`fit` on Train only → `transform` on Train and Test.**
> That is: all parameters (mean, std, min, max, median, categories) are learned from train **only**.

### How?

**❌ The common mistake (leakage):**

```python
# Transforming the whole dataset before splitting!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # mean/std were computed including test data
X_train, X_test = train_test_split(X_scaled, ...)
```

**✅ The correct way:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit + transform on train
X_test_s  = scaler.transform(X_test)        # transform only ✅
```

### Other types of leakage
- **Target leakage**: a feature computed from the target or only available after it happens (e.g. `refund_issued` when predicting fraud).
- **Duplicates** present in both train and test.
- **Time series**: training on the future and testing on the past (use a chronological split).
- **Feature selection / PCA** on the whole data before splitting.
- **Cross-validation** with a scaler already fitted on all the data.

### Best solution: `Pipeline` + `ColumnTransformer`

Bundles all steps and automatically guarantees no leakage (even inside cross-validation):

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

X = data.drop(columns=["bought"])
y = data["bought"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

nominal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore")),
])

ordinal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord",     OrdinalEncoder(categories=[["BAC", "Licence", "Master", "PhD"]])),
])

preprocessor = ColumnTransformer([
    ("num",     num_pipe,     ["age", "salary"]),
    ("nominal", nominal_pipe, ["city"]),
    ("ordinal", ordinal_pipe, ["edu"]),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression()),
])

model.fit(X_train, y_train)               # every fit happens on train only, automatically
print("Test accuracy:", model.score(X_test, y_test))

# Safe cross-validation: fitting is redone inside each fold
scores = cross_val_score(model, X_train, y_train, cv=3)
print(scores.mean())
```

---

# Suggested Mini-Project (End-to-End Practice)

Use the Titanic dataset:

```python
import seaborn as sns
titanic = sns.load_dataset("titanic")
```

1. **Explore:** `head, shape, info, describe, isna().sum()`
2. **EDA:** distribution of `age`, `survived` by `sex` and `pclass` (groupby + countplot), correlation heatmap.
3. **Clean:** `drop_duplicates`, drop redundant columns (`deck` has many missing values), standardize types.
4. **Feature Engineering:** `family_size = sibsp + parch + 1`, `is_alone`.
5. **Separate** `X` and `y = survived`.
6. **`train_test_split`** with `stratify=y`.
7. **ColumnTransformer:** impute + scale for numeric, impute + OneHot for categorical.
8. **Model** (LogisticRegression) inside a Pipeline, and measure accuracy.

---

# Quick Reference (Cheat Sheet)

### NumPy
```python
a.shape  a.ndim  a.dtype  a.reshape(r, -1)
a[1:3]   a[a > 5]   a.mean(axis=0)   (a - a.mean(0)) / a.std(0)
```

### Pandas
```python
pd.read_csv()  df.head()  df.info()  df.describe()
df.loc[rows, cols]  df.iloc[r, c]  df[df.x > 5]
df["new"] = ...   df.sort_values()   df["c"].value_counts()
df.groupby("c")["v"].mean()   pd.merge(a, b, on="k", how="left")
```

### Cleaning / EDA
```python
df.isna().sum()   df.duplicated().sum()   df.drop_duplicates()
df.fillna(v)   df.astype(t)   pd.to_datetime(s)
sns.histplot / boxplot / countplot / scatterplot / heatmap
```

### Preprocessing
```python
X, y = df.drop(columns="t"), df["t"]
train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
imputer / scaler / encoder .fit(X_train)  →  .transform(X_train / X_test)
Pipeline + ColumnTransformer  ← the safe solution against Data Leakage
```

---

# Further Reading

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/basics.html)
- [Pandas — Introductory tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Kaggle Learn — Pandas](https://www.kaggle.com/learn/pandas)
- [Kaggle Learn — Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
- [Scikit-learn — Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn — Common pitfalls / Data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

---

**Good luck on your data science journey!**
