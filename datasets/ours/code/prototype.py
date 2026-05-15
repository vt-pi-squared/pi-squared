import os, re, glob, json
import pandas as pd
import mdpd

from sqlite3 import connect
conn = connect(":memory:")
# final_answer_python = None


def demo():
    md_table = """
    |    | club         | sport      |   founded | league                | venue               |
    |---:|:-------------|:-----------|----------:|:----------------------|:--------------------|
    |  0 | fk vojvodina | football   |      1914 | jelen superliga       | kara\u0111or\u0111e stadium   |
    |  1 | fk proleter  | football   |      1951 | first league          | stadion slana bara  |
    |  2 | fk novi sad  | football   |      1921 | first league          | detelinara stadium  |
    |  3 | kk vojvodina | basketball |      2000 | sinalco superleague   | spens sports center |
    |  4 | kk novi sad  | basketball |      1985 | sinalco superleague   | spens sports center |
    |  5 | ok vojvodina | volleyball |      1946 | serbian volley league | spens sports center |
    |  6 | hk vojvodina | hockey     |      1957 | serbian hockey league | spens sports center |
    |  7 | hk novi sad  | hockey     |      1998 | serbian hockey league | spens sports center |
    """

    df = mdpd.from_md(md_table)
    del df['']
    print(df.dtypes, df.info())

    # convert columns to appropriate/best data types
    df = df.convert_dtypes()
    for col in df.columns:
        try:
            # convert to numeric
            df[col] = pd.to_numeric(df[col])
        except:
            try:
                pass
                # convert to datetime
                # df[col] = pd.to_datetime(df[col])
                # convert to list/dict if possible
                # df[col] = df[col].apply(eval)
            except:
                pass

    print(df.dtypes)
    df.to_sql(name="df", con=conn)


    python_code = """
    import pandas as pd

    # overall average founding year
    overall_avg_year = df['founded'].mean()

    # average founding year per sport
    sport_avg = df.groupby('sport')['founded'].mean().reset_index(name='avg_founded')

    # sport with the oldest (smallest) average founding year
    oldest_sport = sport_avg.loc[sport_avg['avg_founded'].idxmin(), 'sport']
    # clubs of that sport older than the overall average
    filtered = df[
        (df['sport'] == oldest_sport) &
        (df['founded'] < overall_avg_year)
    ]

    # most common venue among these clubs
    venue_counts = filtered['venue'].value_counts()
    most_common_venue = venue_counts.idxmax()
    most_common_count = venue_counts.max()

    final_answer_python = {
        'sport': oldest_sport,
        'most_common_venue': most_common_venue,
        'venue_occurrences': int(most_common_count)
    }
    print(final_answer_python)
    """.replace("    ", "")  # remove leading spaces for cleaner exec

    # exec("global final_answer_python\n" + python_code)
    # this command is tricky. If it's global env, then everything is good.
    # But if it's local env, then final_answer_python is not kept after exec.
    # My idea is to declare it as global inside the code.
    # This is a bit hacky but it works for the demo purpose.
    # In real use, we can use a more elegant way to get the final answer from the local env of exec.

    # or we can do this way
    local_vars = locals()
    exec(python_code, globals(), local_vars)
    print(local_vars.keys())
    final_answer_python = local_vars.get('final_answer_python', None)

    print(final_answer_python)

    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 5 columns):
    #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
    0   club     8 non-null      object
    1   sport    8 non-null      object
    2   founded  8 non-null      object
    3   league   8 non-null      object
    4   venue    8 non-null      object
    dtypes: object(5)
    memory usage: 452.0+ bytes
    club       object
    sport      object
    founded    object
    league     object
    venue      object
    dtype: object None
    club       string[python]
    sport      string[python]
    founded             Int64
    league     string[python]
    venue      string[python]
    dtype: object
    {'sport': 'football', 'most_common_venue': 'karađorđe stadium', 'venue_occurrences': 1}
    """


if __name__ == "__main__":
    demo()


"""
{
  "brainstorm": "The dataset lists various clubs from Vojvodina and Novi Sad, across four sports (football, basketball, volleyball, hockey). Each club has a founding year, participates in a specific league, and shares venues (some clubs share the same venue). Interesting analyses could involve:
- Comparing the average founding year of clubs per sport to see which sport has the oldest clubs on average.
- Determining which venue hosts the most diverse set of sports.
- Finding the sport‑venue combination with the earliest‑founded club.
- Calculating the median founding year for each league and identifying leagues with the oldest median.
- Identifying clubs whose founding year is earlier than the average founding year of all clubs in the same sport **and** that share a venue with at least one club from a different sport.

A multi‑hop, aggregative question that ties several columns together would be insightful and suitably complex.",
  "question": "For each venue, list the sport that has the earliest‑founded club hosted at that venue, together with that club’s name and its founding year. Sort the results by venue name alphabetically.",
  "sql_query": "```sql
SELECT 
    venue,
    sport,
    club,
    founded
FROM df AS d1
WHERE founded = (
    SELECT MIN(founded)
    FROM df AS d2
    WHERE d2.venue = d1.venue
)
ORDER BY venue ASC;
```",
  "python_code": "```python
# Assuming df is the pandas DataFrame containing the table
# Find the minimum founded year per venue
min_founded_per_venue = df.groupby('venue')['founded'].min().reset_index(name='min_founded')

# Merge back to get the rows that match the minimum founded year for each venue
merged = pd.merge(df, min_founded_per_venue,
                  left_on=['venue', 'founded'], right_on=['venue', 'min_founded'],
                  how='inner')

# Select required columns and sort by venue alphabetically
final_answer_python = merged[['venue', 'sport', 'club', 'founded']].sort_values('venue')
print(final_answer_python)
```"
}

#####
expected_reasoning_traces = '''Let's break down the solution step by step.


**Step 1: Find the minimum founding year for each venue**

First, we want to know, for each venue, what is the earliest year that a club was founded. We do this by grouping the table by the 'venue' column and finding the minimum value in the 'founded' column for each group.

```python
min_founded_per_venue = df.groupby('venue')['founded'].min().reset_index(name='min_founded')
```
The result of this step is
>>> min_founded_per_venue


**Step 2: Merge to get the full club info for each minimum**

Now, we want to get not just the year, but also the club and sport associated with that minimum founding year at each venue. We do this by merging the original table with the table we just created, matching on both 'venue' and 'founded' (which should equal 'min_founded').

```python
merged = pd.merge(
    df,
    min_founded_per_venue,
    left_on=['venue', 'founded'],
    right_on=['venue', 'min_founded'],
    how='inner'
)
```

The merged table is as follows:
>>> merged

The table contain now contains, for each venue, the club(s) that were founded in the earliest year at that venue, along with all their details.


**Step 3: Select and sort the final columns**

Finally, we select only the columns we care about: 'venue', 'sport', 'club', and 'founded'. We also sort the results alphabetically by venue name.

```python
final_answer_python = merged[['venue', 'sport', 'club', 'founded']].sort_values('venue')
```

The final output is:
>>> final_answer_python
'''

"""