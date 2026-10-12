# DataCo Supply Chain Dataset — Exploratory Data Analysis & Series Recommendations

## 1. Dataset Overview

- **Total Records:** 180,519 order line items
- **Total Features:** 63 columns
- **Date Range:** 2015-01-01 to 2018-01-31 (1127 continuous days, 3 full years)
- **Null Values:**
  - `Product Description`: 100% missing (not needed for simulation)
  - `Order Zipcode`: 86.24% missing
  - `Customer Lname`: 8 rows (<0.01%)
  - `Customer Zipcode`: 3 rows (<0.01%)
  - All other 49 columns have **zero missing values**.

---

## 2. Key Categorical Distributions

### Shipping Modes
| Shipping Mode   |   Orders | Share   |
|:----------------|---------:|:--------|
| Standard Class  |   107752 | 59.69%  |
| Second Class    |    35216 | 19.51%  |
| First Class     |    27814 | 15.41%  |
| Same Day        |     9737 | 5.39%   |

### Delivery Statuses
| Delivery Status   |   Orders | Share   |
|:------------------|---------:|:--------|
| Late delivery     |    98977 | 54.83%  |
| Advance shipping  |    41592 | 23.04%  |
| Shipping on time  |    32196 | 17.84%  |
| Shipping canceled |     7754 | 4.3%    |

### Markets
| Market       |   Orders | Share   |
|:-------------|---------:|:--------|
| LATAM        |    51594 | 28.58%  |
| Europe       |    50252 | 27.84%  |
| Pacific Asia |    41260 | 22.86%  |
| USCA         |    25799 | 14.29%  |
| Africa       |    11614 | 6.43%   |

---

## 3. Lead-Time & Scheduled Gap Analysis

Lead times vary substantially by shipping mode, revealing clear operational realities in the dataset:

| shipping_mode   |   order_count |   actual_mean |   actual_std |   scheduled_mean |   gap_mean |   gap_std | actual_days_dist                                 |
|:----------------|--------------:|--------------:|-------------:|-----------------:|-----------:|----------:|:-------------------------------------------------|
| First Class     |         27814 |         2     |        0     |                1 |      1     |     0     | 2d:27814                                         |
| Same Day        |          9737 |         0.478 |        0.5   |                0 |      0.478 |     0.5   | 0d:5080, 1d:4657                                 |
| Second Class    |         35216 |         3.991 |        1.416 |                2 |      1.991 |     1.416 | 2d:7138, 3d:7065, 4d:6978, 5d:7052, 6d:6983      |
| Standard Class  |        107752 |         3.996 |        1.417 |                4 |     -0.004 |     1.417 | 2d:21666, 3d:21700, 4d:21535, 5d:21111, 6d:21740 |

### Key Lead-Time Insights:
1. **First Class:** Exactly **2.0 days** with **0.0 standard deviation** across all 27,814 orders. However, scheduled shipment is often set to 1 day, causing a high late rate.
2. **Same Day:** Ships either same day (0 days: 5,080 orders) or next day (1 day: 4,657 orders), with a mean of **0.48 days** (std 0.50).
3. **Second Class & Standard Class:** Exhibit identical empirical actual transit time distributions uniformly spread across **2, 3, 4, 5, and 6 days**, with a mean of **4.00 days** and standard deviation of **1.42 days**.

---

## 4. Late-Delivery Rate Analysis

- **Overall Late Delivery Rate:** **54.83%** (98,977 out of 180,519 orders).
- The column `Late_delivery_risk` directly corresponds to `Delivery Status == 'Late delivery'`.

### Late Delivery Rate by Shipping Mode:
| Shipping Mode   |   total_orders |   late_orders |   late_rate |
|:----------------|---------------:|--------------:|------------:|
| Standard Class  |         107752 |         41023 |      0.3807 |
| Second Class    |          35216 |         26987 |      0.7663 |
| First Class     |          27814 |         26513 |      0.9532 |
| Same Day        |           9737 |          4454 |      0.4574 |

- **First Class:** Has the highest late delivery rate (**95.32%**) because actual transit is fixed at 2 days while scheduled transit is set to 1 day.
- **Second Class:** Has a **76.63%** late delivery rate.
- **Same Day:** **45.74%** late delivery rate.
- **Standard Class:** Most reliable against scheduled expectations, with a **38.07%** late delivery rate.

---

## 5. Top 10 Categories Demand Time-Series Analysis

The table below summarizes demand continuity, sparsity, and stability metrics for the top 10 categories across the full 1,127-day history:

| category             |   total_volume |   daily_mean |   daily_std |   cv |   zero_days_pct |   weekly_ac_lag7 |
|:---------------------|---------------:|-------------:|------------:|-----:|----------------:|-----------------:|
| Cleats               |          73734 |        65.43 |       27.65 | 0.42 |            10.7 |            0.629 |
| Women's Apparel      |          62956 |        55.86 |       24.57 | 0.44 |            10.7 |            0.601 |
| Indoor/Outdoor Games |          57803 |        51.29 |       22.64 | 0.44 |            10.7 |            0.604 |
| Cardio Equipment     |          37587 |        33.35 |       16.36 | 0.49 |            10.7 |            0.495 |
| Shop By Sport        |          32726 |        29.04 |       14.42 | 0.5  |            10.7 |            0.452 |
| Men's Footwear       |          22246 |        19.74 |        8.25 | 0.42 |            10.7 |            0.671 |
| Fishing              |          17325 |        15.37 |        6.68 | 0.43 |            10.7 |            0.601 |
| Water Sports         |          15540 |        13.79 |        6.15 | 0.45 |            10.7 |            0.556 |
| Camping & Hiking     |          13729 |        12.18 |        5.51 | 0.45 |            10.7 |            0.551 |
| Electronics          |           9436 |         8.37 |        7.45 | 0.89 |            27.6 |            0.441 |

### Top Category x Market Regional Slices:
| category             | market       |   total_volume |   daily_mean |   cv |   zero_days_pct |   weekly_ac_lag7 |
|:---------------------|:-------------|---------------:|-------------:|-----:|----------------:|-----------------:|
| Cleats               | LATAM        |          21881 |        19.42 | 1.71 |            73.2 |            0.894 |
| Cleats               | Europe       |          20343 |        18.05 | 1.72 |            65.3 |            0.858 |
| Women's Apparel      | LATAM        |          18908 |        16.78 | 1.72 |            73.2 |            0.88  |
| Women's Apparel      | Europe       |          17676 |        15.68 | 1.73 |            65.4 |            0.856 |
| Indoor/Outdoor Games | LATAM        |          17636 |        15.65 | 1.73 |            73.2 |            0.89  |
| Indoor/Outdoor Games | Europe       |          15692 |        13.92 | 1.72 |            65.3 |            0.829 |
| Cleats               | Pacific Asia |          15289 |        13.57 | 1.97 |            72.8 |            0.848 |
| Women's Apparel      | Pacific Asia |          12630 |        11.21 | 1.99 |            72.8 |            0.847 |
| Indoor/Outdoor Games | Pacific Asia |          12108 |        10.74 | 1.99 |            72.8 |            0.86  |
| Cardio Equipment     | LATAM        |          11346 |        10.07 | 1.77 |            73.2 |            0.83  |

---

## 6. Series Recommendations for Simulation & Forecasting

Based on volume density, length of unbroken history, low sparsity (zero-demand days), low coefficient of variation ($CV = \sigma / \mu$), and weekly periodicity:

### Primary Category Recommendation: **Cleats**
- **Total Demand Volume:** 73,734 units (ranked #1 in dataset).
- **Daily Average:** 65.4 units/day.
- **Sparsity:** Zero-demand on only 10.7% of days globally, and **0.0% zero-demand weeks** (145 continuous weeks).
- **Stability ($CV$):** 0.42 at daily level, and **0.12 at weekly level** — exceptionally stable with low noise.
- **Seasonality:** Strong lag-7 weekly autocorrelation ($r = 0.629$).

### Secondary Category Recommendation: **Women's Apparel**
- **Total Demand Volume:** 62,956 units (ranked #2).
- **Daily Average:** 55.9 units/day.
- **Stability ($CV$):** 0.44 daily, 0.13 weekly.
- **Weekly Autocorrelation:** $r = 0.601$.

### Recommended Market Slices (if regional granularity is desired):
1. **Cleats in Europe:** 20,343 units, daily mean 18.1, lag-7 autocorrelation of **0.858**.
2. **Cleats in LATAM:** 21,881 units, daily mean 19.4, lag-7 autocorrelation of **0.894**.
3. **Women's Apparel in Europe:** 17,676 units, daily mean 15.7, lag-7 autocorrelation of **0.856**.

---

## 7. Configuration Exposure & Flexibility Notice

> **Important Architecture Constraint:**
> As specified in the design guidelines, **the final series is NOT hardcoded**. 
> Downstream components (simulation environment, forecasting models, and multi-agent systems) read the chosen series from [`configs/data.yaml`](file:///c:/Users/saran/smartsupply/configs/data.yaml):
> - `granularity`: `"daily"` or `"weekly"`
> - `category`: e.g., `"Cleats"`, `"Women's Apparel"`, or `null` for all categories
> - `market`: e.g., `"Europe"`, `"LATAM"`, or `null` for all markets
> - `rolling_window`: `7` (default)
