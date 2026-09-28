# DSA 4060 – Recommender Systems

## Week 2 Practical: Recommendation Data and Popularity Baselines

This repository contains my **DSA 4060 Recommender Systems Week 2 Practical Task** on recommendation data and popularity baselines.

The purpose of this practical is to work with user-movie rating data, clean and explore the interactions, construct a user-item interaction matrix, measure the sparsity of the interaction data, and develop a simple **popularity-based movie recommender system**.

The project also demonstrates some of the basic challenges encountered when working with recommender-system data, including missing values, duplicate records, invalid ratings, repeated user-item interactions, sparse interaction matrices, and popularity bias.

---

# 1. Practical Objectives

The main objectives of this practical were to:

1. Load and inspect the movie ratings dataset.
2. Identify missing values and duplicate records.
3. Clean the interaction data.
4. Confirm that ratings fall within the expected rating scale.
5. Explore user and movie interaction patterns.
6. Determine the most active user.
7. Determine the most-rated movie.
8. Calculate the overall average rating.
9. Construct a user-item interaction matrix.
10. Calculate the sparsity of the interaction matrix.
11. Build a simple popularity-based recommender system.
12. Recommend the top 10 movies with at least 10 ratings.
13. Export the recommendations to a CSV file.
14. Interpret the strengths and limitations of the popularity-based approach.

---

# 2. Dataset

The dataset used in this practical was provided for the DSA 4060 Week 2 exercise.

The original dataset is stored in:

```text
data/ratings.csv
```

The dataset initially contains:

* **509 records**
* **5 columns**

Although the assignment description identifies four main interaction fields, the provided dataset also contains the movie title.

## Dataset Attributes

| Column        | Description                                |
| ------------- | ------------------------------------------ |
| `user_id`     | Unique identifier representing a user      |
| `movie_id`    | Unique identifier representing a movie     |
| `movie_title` | Name/title of the movie                    |
| `rating`      | Rating given by a user to a movie          |
| `timestamp`   | Date and time when the rating was recorded |

The rating field is expected to contain values between **1 and 5**.

---

# 3. Tools and Technologies

The practical was completed using:

* Python
* Pandas
* NumPy
* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

Pandas was mainly used for data loading, cleaning, aggregation, construction of the user-item matrix, and generation of the final recommendations.

---

# 4. Loading and Inspecting the Dataset

The dataset was loaded into a Pandas DataFrame.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("../data/ratings.csv")

df.head(10)
```

The first 10 records were displayed to understand the structure and contents of the dataset.

The dataset dimensions were checked using:

```python
df.shape
```

The original dataset dimensions were:

```text
(509, 5)
```

This means that the original dataset contains **509 rows and 5 columns**.

The column data types were also inspected to confirm how Pandas interpreted each variable.

The main columns include string identifiers and titles, an integer rating field, and a timestamp field.

---

# 5. Initial Data Quality Inspection

Before performing the recommendation analysis, the dataset was inspected for common data-quality problems.

The inspection focused on:

* Missing values
* Duplicate records
* Invalid ratings

## Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

The initial inspection identified:

| Column        | Missing Values |
| ------------- | -------------: |
| `user_id`     |              2 |
| `movie_id`    |              1 |
| `movie_title` |              1 |
| `rating`      |              0 |
| `timestamp`   |              0 |

User and movie IDs are particularly important because they are required to identify interactions between users and movies.

Records without these identifiers cannot be reliably used when constructing the user-item interaction matrix.

---

# 6. Duplicate Records

Duplicate records were checked using:

```python
df.duplicated().sum()
```

A total of:

**3 duplicate records**

were found in the original dataset.

Duplicate records can distort the analysis because the same interaction may be counted more than once.

The duplicates were therefore removed before continuing with the analysis.

```python
df_clean = df.drop_duplicates().copy()
```

After removing duplicates, the dataset decreased from **509 to 506 records**.

---

# 7. Handling Missing User and Movie IDs

Records containing missing `user_id` or `movie_id` values were removed.

```python
df_clean = df_clean.dropna(
    subset=["user_id", "movie_id"]
).copy()
```

This was necessary because both identifiers are required when creating the user-item interaction matrix.

A rating without a valid user or movie identifier cannot represent a usable user-movie interaction.

---

# 8. Rating Scale Validation

The assignment uses an expected rating scale of:

**1 to 5**

The minimum and maximum values were inspected.

The dataset initially produced:

```text
Minimum rating: -1
Maximum rating: 6
```

Three invalid ratings were discovered.

| User | Movie | Movie Title        | Invalid Rating |
| ---- | ----- | ------------------ | -------------: |
| U006 | M010  | Beyond the Horizon |              0 |
| U028 | M015  | The Green Valley   |              6 |
| U041 | M023  | The Startup        |             -1 |

All three values fall outside the valid **1–5 rating scale**.

Instead of changing these values to another rating, the affected records were removed.

```python
df_clean = df_clean[
    df_clean["rating"].between(1, 5)
].copy()
```

After this step:

```text
Minimum rating: 1
Maximum rating: 5
```

---

# 9. Final Data Cleaning Results

After completing all cleaning operations, the dataset was checked again.

The final cleaned dataset contains:

```text
Cleaned dataset dimensions: (500, 5)
Duplicate records: 0
Missing user IDs: 0
Missing movie IDs: 0
Invalid ratings: 0
```

Therefore, the analysis continues using **500 valid rating records**.

## Cleaning Summary

| Data Quality Measure         | Result |
| ---------------------------- | -----: |
| Original records             |    509 |
| Duplicate records identified |      3 |
| Missing user IDs identified  |      2 |
| Missing movie IDs identified |      1 |
| Invalid ratings identified   |      3 |
| Final valid rating records   |    500 |
| Remaining duplicates         |      0 |
| Remaining missing user IDs   |      0 |
| Remaining missing movie IDs  |      0 |
| Remaining invalid ratings    |      0 |

The counts of individual issues should not simply be subtracted from 509 because some data-quality issues can occur in overlapping records. The final cleaned DataFrame contains 500 usable rating events.

---

# 10. Exploratory Analysis

After cleaning, exploratory analysis was performed to understand the interaction behaviour of users and movies.

The following statistics were calculated:

* Total number of rating records
* Number of unique users
* Number of unique movies
* Most active user
* Most-rated movie
* Overall average rating

## Main Results

| Metric                             |         Result |
| ---------------------------------- | -------------: |
| Total valid rating records         |            500 |
| Unique users                       |             50 |
| Unique movies                      |             24 |
| Average rating                     |           3.45 |
| Most active user                   |           U010 |
| Ratings submitted by U010          |             17 |
| Most-rated movie                   | Nairobi Nights |
| Movie ID                           |           M001 |
| Ratings received by Nairobi Nights |             37 |

---

# 11. Total Number of Ratings

The cleaned dataset contains:

**500 rating records**

These records represent the valid rating events remaining after removing the identified data-quality problems.

---

# 12. Unique Users

The number of unique users was calculated using:

```python
df_clean["user_id"].nunique()
```

The dataset contains:

**50 unique users**

These users form the rows of the user-item interaction matrix.

---

# 13. Unique Movies

The number of unique movies was calculated using:

```python
df_clean["movie_id"].nunique()
```

The cleaned dataset contains:

**24 unique movies**

These movies form the columns of the user-item interaction matrix.

---

# 14. Most Active User

The most active user was determined by counting the number of rating records associated with each user.

```python
user_rating_counts = df_clean["user_id"].value_counts()
```

The result was:

```text
Most active user: U010
Number of ratings: 17
```

Therefore, **U010** generated the largest number of rating events in the cleaned dataset.

---

# 15. Most-Rated Movie

Movie rating counts were calculated by grouping the data by movie ID and movie title.

The most-rated movie was:

**Nairobi Nights (M001)**

It received:

**37 ratings**

The most-rated movie represents the movie with the largest number of recorded rating events rather than necessarily the movie with the highest average rating.

This distinction is important because **popularity and rating quality are not exactly the same concept**.

---

# 16. Average Dataset Rating

The average rating across the cleaned dataset is approximately:

**3.45 out of 5**

This provides a general indication of the rating behaviour within the dataset.

---

# 17. User-Item Interaction Matrix

A user-item interaction matrix was constructed using the cleaned data.

The matrix was created using:

```python
user_item_matrix = df_clean.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    aggfunc="mean"
)
```

In this matrix:

* Each **row** represents a user.
* Each **column** represents a movie.
* Each populated cell represents the rating associated with a user-movie pair.
* `NaN` indicates that no rating is available for that particular user-movie combination.

The dimensions of the matrix are:

```text
(50, 24)
```

Therefore, the interaction matrix contains:

* **50 users**
* **24 movies**

This gives:

```text
50 × 24 = 1,200
```

possible user-movie interactions.

---

# 18. Repeated User-Movie Ratings

An important observation was made when constructing the interaction matrix.

Although the cleaned dataset contains:

**500 rating records**

the matrix contains only:

**430 observed user-movie cells**

This occurs because some users rated the same movie more than once.

When `pivot_table()` creates the matrix, repeated ratings belonging to the same user-movie combination are aggregated using the mean.

For example, if the same user rated the same movie more than once, these are multiple rating events in the raw interaction data but represent only **one user-movie pair** in the matrix.

This explains why:

```text
500 rating records ≠ 500 unique user-movie interactions
```

The distinction is important when calculating sparsity.

---

# 19. Sparsity Calculation

Sparsity measures how much of the possible user-item interaction space contains no observed interaction.

The formula used is:

```text
Sparsity = 1 - (Observed Interactions / Possible Interactions)
```

The dataset contains:

```text
Number of users = 50
Number of movies = 24
```

Therefore:

```text
Possible interactions = 50 × 24
                      = 1,200
```

The user-item matrix contains:

```text
Observed interactions = 430
```

Therefore:

```text
Sparsity = 1 - (430 / 1,200)

         = 1 - 0.3583

         = 0.6417
```

As a percentage:

**Sparsity = 64.17%**

---

# 20. Sparsity Interpretation

The user-item interaction matrix is approximately:

**64.17% sparse**

This means that approximately **64.17% of all possible user-movie combinations do not contain an observed rating**.

The observed proportion is approximately:

```text
100% - 64.17% = 35.83%
```

Therefore, approximately **35.83% of the possible user-movie pairs contain an observed interaction**.

Sparsity is common in recommender systems because most users interact with only a subset of the available items.

As the number of users and items grows, the number of possible interactions can become very large while each user continues to interact with only a relatively small number of items.

---

# 21. Popularity-Based Recommender System

The final part of the practical involved building a simple popularity-based recommender.

A popularity-based recommender does not attempt to model the preferences of individual users.

Instead, it recommends movies that perform well across the user population according to a chosen popularity measure.

For this practical, two statistics were calculated for each movie:

1. **Average rating**
2. **Number of ratings**

The statistics were generated using:

```python
movie_stats = (
    df_clean.groupby(["movie_id", "movie_title"])
    .agg(
        average_rating=("rating", "mean"),
        number_of_ratings=("rating", "count")
    )
    .reset_index()
)
```

---

# 22. Minimum Rating Threshold

Ranking movies using average rating alone can produce misleading recommendations.

For example, a movie with only one rating of 5 could receive an average rating of 5.0 and appear better than a movie that has consistently received high ratings from many users.

To reduce this problem, the practical requires movies to have at least:

**10 ratings**

before they can be considered for recommendation.

The filter used was:

```python
popular_movies = movie_stats[
    movie_stats["number_of_ratings"] >= 10
].copy()
```

The eligible movies were then ranked primarily by average rating.

When movies had the same average rating, the number of ratings was used as the secondary sorting criterion.

---

# 23. Top 10 Movie Recommendations

The resulting top 10 popularity-based recommendations are:

| Rank | Movie ID | Movie Title         | Average Rating | Number of Ratings |
| ---: | -------- | ------------------- | -------------: | ----------------: |
|    1 | M021     | Market Day          |           4.47 |                15 |
|    2 | M013     | Midnight in Mombasa |           4.35 |                17 |
|    3 | M014     | Code Breakers       |           4.25 |                16 |
|    4 | M019     | Future Africa       |           4.13 |                15 |
|    5 | M020     | Home Again          |           4.13 |                15 |
|    6 | M007     | Hidden Lake         |           4.12 |                16 |
|    7 | M006     | City of Tomorrow    |           3.91 |                34 |
|    8 | M005     | Mountain Echoes     |           3.86 |                37 |
|    9 | M012     | The Final Goal      |           3.76 |                17 |
|   10 | M004     | Coastal Dreams      |           3.67 |                27 |

---

# 24. Top Recommendation

The highest-ranked movie is:

## Market Day (M021)

It received:

* **Average rating: 4.47**
* **Number of ratings: 15**

Among movies meeting the minimum requirement of 10 ratings, *Market Day* has the highest average rating.

It therefore appears at the top of the popularity-based recommendation list.

---

# 25. Why These Movies Appear at the Top

The recommended movies appear at the top because they have relatively high average ratings while also satisfying the requirement of receiving at least 10 ratings.

The threshold is important because it prevents movies with extremely small numbers of ratings from dominating the ranking.

For example, a movie that receives a single 5-star rating could technically have an average rating of 5.0. However, there is not enough evidence from one rating to confidently describe the movie as broadly popular.

By requiring at least 10 ratings, the recommender considers movies with a larger amount of interaction evidence.

---

# 26. Popularity Versus Number of Ratings

An interesting result can be observed by comparing the most-rated movie with the highest-ranked recommendation.

The most-rated movie is:

**Nairobi Nights (M001)** with **37 ratings**.

However, the highest-ranked recommendation is:

**Market Day (M021)** with an average rating of **4.47 from 15 ratings**.

This happens because the recommendation system does not rank movies using rating count alone.

Instead:

1. Movies must first receive at least 10 ratings.
2. Eligible movies are primarily ranked by their average rating.
3. Rating count acts as a secondary sorting criterion when needed.

Therefore, receiving the most ratings does not automatically make a movie the highest-ranked recommendation.

---

# 27. Advantage of the Popularity-Based Approach

One major advantage of a popularity-based recommender is its **simplicity**.

The system does not need to learn a detailed preference profile for each individual user.

This also makes it useful for the **new-user or cold-start problem**.

When a new user joins the system, there may be no previous ratings available to determine their preferences.

A popularity-based system can still immediately recommend generally well-rated movies.

The approach is also easy to understand and explain because the recommendation criteria are transparent.

---

# 28. Limitations

Despite its simplicity, the popularity-based approach has important limitations.

## Limitation 1: Lack of Personalization

Every user receives essentially the same recommendation list.

The system does not consider:

* A user's previous ratings
* Favourite genres
* Individual interests
* Similar users
* Viewing history
* Personal preferences

For example, two users with completely different movie preferences would still receive the same top 10 recommendations.

More advanced recommender systems attempt to solve this problem by generating personalized recommendations.

---

## Limitation 2: Popularity Bias

The approach can favour movies that already receive significant user attention.

Less popular or newer movies may struggle to appear in the recommendation list because they have not accumulated enough ratings.

This can create a feedback loop where popular movies receive more exposure and therefore continue to collect more interactions.

The minimum requirement of 10 ratings can also prevent newly introduced movies from appearing in the recommendation list until they collect enough ratings.

---

# 29. Output File

## Top 10 Movie Recommendations

After calculating the average rating and number of ratings for each movie, only movies with **at least 10 ratings** were considered eligible for recommendation. The eligible movies were ranked primarily by their average rating, with the number of ratings used as a secondary sorting criterion.

The final top 10 recommendations are:

| Rank | Movie ID | Movie Title         | Average Rating | Number of Ratings |
| ---: | -------- | ------------------- | -------------: | ----------------: |
|    1 | M021     | Market Day          |           4.47 |                15 |
|    2 | M013     | Midnight in Mombasa |           4.35 |                17 |
|    3 | M014     | Code Breakers       |           4.25 |                16 |
|    4 | M019     | Future Africa       |           4.13 |                15 |
|    5 | M020     | Home Again          |           4.13 |                15 |
|    6 | M007     | Hidden Lake         |           4.12 |                16 |
|    7 | M006     | City of Tomorrow    |           3.91 |                34 |
|    8 | M005     | Mountain Echoes     |           3.86 |                37 |
|    9 | M012     | The Final Goal      |           3.76 |                17 |
|   10 | M004     | Coastal Dreams      |           3.67 |                27 |

### Top Recommendation

The highest-ranked movie is **Market Day (M021)**, which has an average rating of **4.47 from 15 ratings**. It ranks first because it has the highest average rating among the movies that satisfy the minimum requirement of 10 ratings.

An interesting observation is that the most-rated movie is not necessarily the highest-ranked recommendation. For example, **Mountain Echoes (M005)** received 37 ratings but has an average rating of 3.86, placing it eighth in the recommendation list. Similarly, **City of Tomorrow (M006)** received 34 ratings but ranks seventh with an average rating of 3.91.

This demonstrates that the recommender is not simply selecting movies with the largest number of ratings. The rating-count threshold determines whether a movie is eligible, while the average rating is the main factor used to rank the eligible movies.

The complete recommendation results are also available in:

`output/top_10_recommendations.csv`


The file contains:

* Movie ID
* Movie title
* Average rating
* Number of ratings

This satisfies the requirement to submit a separate CSV containing the top 10 recommended movies.

---

# 30. Project Structure

The repository is organised as follows:

```text
dsa4060-week2-popularity-recommender/
│
├── data/
│   └── ratings.csv
│
├── notebooks/
│   └── week2_popularity_recommender.ipynb
│
├── output/
│   └── top_10_recommendations.csv
│
├── README.md
│
└── requirements.txt
```

### `data/`

Contains the original movie-rating dataset used for the practical.

### `notebooks/`

Contains the completed Jupyter Notebook with the code, outputs, Markdown explanations, analysis, and interpretation.

### `output/`

Contains the CSV file generated by the popularity-based recommender.

### `README.md`

Provides documentation explaining the practical, methodology, findings, recommendations, and project structure.

### `requirements.txt`

Contains the Python packages required to run the project.

---

# 31. How to Run the Project

## Step 1: Clone the Repository

```bash
git clone <repository-url>
```

## Step 2: Enter the Project Directory

```bash
cd dsa4060-week2-popularity-recommender
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Open the Notebook

Open:

```text
notebooks/week2_popularity_recommender.ipynb
```

using Jupyter Notebook or Visual Studio Code.

## Step 5: Run the Notebook

Run all notebook cells from top to bottom.

The notebook will:

1. Load the dataset.
2. Inspect the data.
3. Identify missing values.
4. Identify duplicate records.
5. Remove invalid records.
6. Perform exploratory analysis.
7. Construct the user-item interaction matrix.
8. Calculate sparsity.
9. Calculate movie popularity statistics.
10. Generate the top 10 recommendations.
11. Export the recommendations to CSV.

---

# 32. Key Findings

The main findings from the practical are:

| Finding                            |         Result |
| ---------------------------------- | -------------: |
| Original records                   |            509 |
| Clean rating records               |            500 |
| Unique users                       |             50 |
| Unique movies                      |             24 |
| Average dataset rating             |           3.45 |
| Most active user                   |           U010 |
| Ratings by most active user        |             17 |
| Most-rated movie                   | Nairobi Nights |
| Ratings for most-rated movie       |             37 |
| Possible user-movie interactions   |          1,200 |
| Observed user-movie interactions   |            430 |
| Matrix sparsity                    |         64.17% |
| Minimum ratings for recommendation |             10 |
| Highest-ranked recommendation      |     Market Day |
| Market Day average rating          |           4.47 |
| Market Day rating count            |             15 |

---

# 33. Conclusion

This practical demonstrated the basic workflow involved in preparing interaction data and building a baseline recommender system.

The original dataset contained 509 records and required cleaning before analysis. Duplicate records, missing identifiers, and ratings outside the valid 1–5 scale were identified and handled. After cleaning, 500 valid rating records remained.

Exploratory analysis showed that the dataset contains 50 unique users and 24 unique movies, with an overall average rating of approximately 3.45. User U010 was the most active user with 17 rating events, while *Nairobi Nights* was the most-rated movie with 37 ratings.

A user-item interaction matrix was then constructed. Although there were 500 cleaned rating records, the matrix contained 430 observed user-movie interactions because some users had rated the same movie more than once. With 1,200 possible interactions, the resulting sparsity was approximately 64.17%.

Finally, a popularity-based recommender was developed by calculating the average rating and number of ratings for each movie. Movies were required to have at least 10 ratings before being eligible for recommendation.

The highest-ranked movie was *Market Day*, with an average rating of approximately 4.47 from 15 ratings.

The popularity-based recommender provides a useful and interpretable baseline, particularly when information about individual users is unavailable. However, its recommendations are not personalized and can be affected by popularity bias.

Future work could extend this baseline using more advanced recommendation approaches such as **user-based collaborative filtering, item-based collaborative filtering, content-based filtering, or matrix factorization** to provide recommendations that better reflect individual user preferences.
