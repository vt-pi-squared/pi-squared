import dspy
DSPyChatAdapter = dspy.ChatAdapter()


# https://dspy.ai/tutorials/cache/#disablingenabling-dspy-cache
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


# convert DSPy program to generic prompt. the magic is here
# https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py#L34
def get_messages_templates(dspy_signature, demos=[]):
    generic_messages = DSPyChatAdapter.format(
        dspy_signature,
        demos=demos,
        inputs={k: f"{{{k}}}" for k in dspy_signature.input_fields},
    )
    return generic_messages

# get the output parsing function
# DSPyChatAdapter.parse(dspy_signature, completion)
# messages = get_messages_templates(signature)
# messages[-1]["content"] = messages[-1]["content"].format(**inputs)


# ------------------------------------------------------------------
# Wikipedia Table Expansion
# ------------------------------------------------------------------

# Identify common attributes from Wikipedia summaries and enrich the table with new columns
wiki_table_multicol_expansion_description = f"""
You are a creative and grounded brainstorming partner for enriching tables in Wikipedia pages.

You will be given:
- a table in HTML format,
- some metadata about the table,
- summaries of the Wikipedia pages referenced by a column of the table.

Your task is to enrich the table by adding one or several (up to 3) new columns that capture common attributes from the Wikipedia summaries which are not already semantically reflected in the existing columns of the table.

**Action plan**:
- Identify one or several common attributes from the Wikipedia summaries. Each attribute must be:
    - relevant to the data in the table and provide additional insights to the readers,
    - **not already present or partially reflected in the existing columns of the table**,
    - not in conflict with the existing data in the table (e.g., statistics from different years compared to similar existing attributes).
- Extract the relevant data/values from the summaries to populate the new columns.
- Values must be short pieces of content (e.g., a number, a phrase, a several-word string), directly retrieved from the summaries, no extra notes, and should not require additional interpretation or analysis.
- Return the enriched table in HTML format, ensuring that the new columns are seamlessly integrated with the existing columns.
- **Provide a list of notes on the enriched table** about any cautions or assumptions made during the enrichment process, for examples, in the follow cases:
    - Statistics in a new column are from different years, so they may not be directly comparable. You should provide such a note with specific year information, e.g., "- Attribute X: row 1's value is from 2020, while row 2's value is from 2022."
- In case there are typos in the original table, you are allowed to modify these typos. Don't need to report the incident. Typos cases include but are not limited to:
    - Misspelled words -> Fix the typos (e.g., "Spanyol" -> "Espanyol")
    - Inconsistent (and not sensical) statistics -> Resolve the inconsistent or the typos (e.g., in a numeric column, values are in different units, or one uses `,` as thousand separator while others use `.`, etc.)

NOTE:
- Existing column names may not fully reflect the content of the column, so you should also consider the actual data in the columns when determining whether an attribute is already present.
- There are some common taxonomies of attributes that are often present in Wikipedia summaries, such as:
    - Person: teams/companies/organizations, awards, career milestones, etc.
    - Organization: industry, headquarters location, revenue, number of employees, etc.
    - Event: date, location, participants, outcome, etc.
    - Geographical entity: population, area, GDP, neighboring entities, etc.
    - Knowledge: year, field, key contributions, inventors, etc.
"""

wiki_table_singlecol_expansion_description = f"""
You are a creative and grounded brainstorming partner for enriching tables in Wikipedia pages.

You will be given:
- a table in HTML format,
- some metadata about the table,
- summaries of the Wikipedia pages referenced by a column of the table.

Your task is to enrich the table by adding one new column that captures a common attribute from the Wikipedia summaries which is not already semantically reflected in the existing columns of the table.

**Action plan**:
- Identify one common attribute from the Wikipedia summaries. The attribute must be:
    - relevant to the data in the table and provide additional insights to the readers,
    - **not already present or partially reflected in the existing columns of the table**,
    - not in conflict with the existing data in the table (e.g., statistics from different years compared to similar existing attributes).
- Extract the relevant data/values from the summaries to populate the new column.
- Values must be short pieces of content (e.g., a number, a phrase, a several-word string), directly retrieved from the summaries, no extra notes, and should not require additional interpretation or analysis.
- Return the enriched table in HTML format, ensuring that the new column is seamlessly integrated with the existing columns.
- **Provide a list of notes on the enriched table** about any cautions or assumptions made during the enrichment process, for example, in the following cases:
    - Statistics in the new column are from different years, so they may not be directly comparable. You should provide such a note with specific year information, e.g., "- Attribute X: row 1's value is from 2020, while row 2's value is from 2022."
- In case there are typos in the original table, you are allowed to modify these typos. Don't need to report the incident. Typos cases include but are not limited to:
    - Misspelled words -> Fix the typos (e.g., "Spanyol" -> "Espanyol")
    - Inconsistent (and not sensical) statistics -> Resolve the inconsistent or the typos (e.g., in a numeric column, values are in different units, or one uses `,` as thousand separator while others use `.`, etc.)

NOTE:
- Existing column names may not fully reflect the content of the column, so you should also consider the actual data in the columns when determining whether an attribute is already present.
- There are some common taxonomies of attributes that are often present in Wikipedia summaries, such as:
    - Person: teams/companies/organizations, awards, career milestones, etc.
    - Organization: industry, headquarters location, revenue, number of employees, etc.
    - Event: date, location, participants, outcome, etc.
    - Geographical entity: population, area, GDP, neighboring entities, etc.
    - Knowledge: year, field, key contributions, inventors, etc.
"""

class TableExpansion(dspy.Signature):
    __doc__ = wiki_table_multicol_expansion_description
    table: str = dspy.InputField(desc='Table in HTML format')
    metadata: str = dspy.InputField(desc='Metadata of the table, such as data source, important notes, etc.')
    column_name: str = dspy.InputField(desc='The name of the column that references Wikipedia pages')
    wiki_summaries: str = dspy.InputField(desc='Summaries of the Wikipedia pages referenced by a column of the table')
    enriched_table: str = dspy.OutputField(desc='The enriched table in HTML format with new columns added. Wrapped with ```html ...``` for better readability.')
    notes: str = dspy.OutputField(desc='A list of notes about any cautions or assumptions made during the enrichment process')

TableExpansion_parser = lambda text: DSPyChatAdapter.parse(TableExpansion, text)


# ------------------------------------------------------------------
# SQL and Python-guided QA and Reasoning Traces Generation
# ------------------------------------------------------------------
# Utilize SQL closeness to natural language, Python's capability to express reasoning logics, and both for X-check

# 1st LLM call: Generate Q and SQL based on the table
sqa_gen_description = f"""
You are an expert designed for advanced data analysis.

You are given a table in Markdown format, a column name, and some context about the table, which reflect real information from the Internet.
Your task is to generate **a complex, self-contained, and natural question** about the data along with a SQL implementation to answer it.

**Strict Requirements:**

1. **Question Design:**

1.1. Generate ONE single-focus and concise question.
    - DON'T concatenate multiple sub-questions with "and" or "which... and which".
    - Split multi-part questions into separate questions. For example, instead of "which X and which Y", ask only "which X". 
    - Instead of "which X and which Y when X", just ask "When X, then which Y?" to maintain the complexity and multi-hop reasoning, but avoid asking two separate questions in one.
        - DON'T: "Which team has the highest average score and which team has the lowest average score?", "What is the year when the Lansing fare per mile reached its highest recorded value, and what was the difference in fare per mile between Lansing and Grand Rapids in the year?"  
        - DO: "Which team has the highest average score?", "Which team has the lowest average score?", "In the year when the Lansing fare per mile reached its highest recorded value, what was the difference in fare per mile between Lansing and Grand Rapids?"

1.2. **Complexity**: Ensure the question probes deeply into the data's nuances.
    - It should involve multi-hop reasoning and information aggregation, by embedding multiple constraints or conditions in the question.
        - Example: "in the year when X reached its maximum", "for the team with the lowest average score in X", "among the teams with average score above 80", etc.
    - It can involving (conditional) **counting or information aggregation (e.g., sum, max, min, average, etc.)** to increase the complexity.
    - Nonetheless, the question is still as meaningful and natural as being asked by real data analysts.

1.3. **Self-containedness**: The question should be fully self-contained, providing all necessary context and details within the question itself, without requiring external information or references to the table or metadata.
    - Utilize specific information in the table and metadata to design the question. Assuming that the question will be given without any context, and one can still fully understand it.
        - For example, given a table with a section name "Number of athletes by National Olympic Committee". phrase "ranked 31 to 45" should be more specific to explain which ranking is referred to, such as "ranked 31 to 45 in the number of athletes sent to the Olympics".
    - Must have phrases about time, location, or taxonomy etc. to narrow down the scope of the question.
        - Example: "In the year when X reached its highest value", "In competition X, among the teams with average score above 80", "for the team with the lowest average score in X", etc.
    - The wording **should NOT refer to the table or any metadata items**, as real users do NOT have access to these information when being asked.
        - AVOID phrases like "which column", "in the column of", "the value in column X", or, "as listed in (somewhere), "section", etc.
        - AVOID words like 'listed', 'list', 'table', 'column', 'row', 'field', 'section', 'caption', 'metadata', etc., which implicitly or explicitly refer to the table or metadata.

1.4. **Naturalness**: The question should be phrased in a natural and conversational manner, as if it were being asked by a real data analyst or researcher.
    - Use simple, clear terminology. Avoid jargon, use layman-term descriptions instead.
    - The wording must be fluent and coherent, with proper grammar and syntax. It should read like a question that a human would naturally ask when analyzing data.

1.5. If the input `condition_col` is not "No condition column", the generated question MUST include a clear condition or constraint that references that column.
    - The condition should be derived from the values present in that column (inspect the column contents) and may be:
        - a filter on categorical values (e.g., "for [the object of question] with [condition_col] is 'Value'"),
        - a numeric comparison (e.g., "consider [the object of question] with [condition_col] > 1000", "regarding ... among in the top 10% of [condition_col]"),
        - a temporal or extrema-based condition (e.g., "in the year when [condition_col] reached its maximum", "for the month with the lowest [condition_col]"),
        - a ranking-based constraint (e.g., "among the top 3 entries by [condition_col]").

1.6. Answer:
    - The question must have a deterministic, single answer.
    - Ensure the question is answerable using only data present in the table.
    - Keywords of the question that will be used to determine the answer should be present in the table and not cover multiple values in the corresponding column.
        - E.g., "Existing" vs "Existing with temporary stands" should be treated as different keywords, and the question should be specific to one of them, such as "for the existing stadiums" or "for the stadiums with temporary stands".
    - For the ease of post evaluation, the answer must be a specific value (a **short** string, a number) or a small set of information (a list or an entity-value dictionary with no more than 3 elements or key-value pairs).
        - Example: {{'team_home': 'Poli Ejido', 'team_away': 'Espanyol'}}, "Espanyol", 83.6, ["Espanyol", "Poli Ejido"], etc.

1.7. **Excellent, Good and Bad Examples**:
    - EXCELLENT questions:
        - Based on the 2014 Pew Research Center survey data comparing the ethnic composition of Latter-day Saints in the U.S. to the general U.S. population in 2020, which ethnicity had the largest positive percentage point difference between its representation among Latter-day Saints and its representation in the general U.S. population?
        - Among the regional groupings for which both the International Monetary Fund and the World Bank provided data, which region showed the largest absolute difference between its 2025 IMF forecast and its 2024 World Bank estimate?
        - In the 2025-26 NCAA Division I women's basketball season, among the recorded upsets where a non-Division I team defeated a Division I opponent, which game featured the smallest point differential between the winner and the loser?
        - In the study of standard reduction potentials for aqueous iodine species, which chemical couple exhibits a potential of exactly 0 volts in basic conditions while simultaneously having the highest recorded potential among all such couples in acidic conditions?
        - Based on the 2022 cobalt production and reserves data reported by the USGS, what is the ratio of reserves to annual production of the group of all explicitly reported Asian countries (excluding aggregate categories like 'Other countries' or 'World total')?
    - GOOD questions:
        - What was the amount spent in millions of nominal dollars by the highest spending U.S Federal Department in the fiscal year of 1955?
        - Who holds the all-time record at the Grammys for the most wins in the album of the year category?
        - What is the current age of the oldest person to sail solo across the Pacific Ocean?
        - How many NBA players have scored 60 or more points in a regular season game since 2023?
        - Of the countries with a head of state assuming office prior to 2000, which are the five with the largest GDP per capita?
        - How many current and former Real Madrid players are ranked in the top 10 of the 2025 Forbes list of the world's highest-paid athletes?
    - BAD questions (and reason why each is bad):
        - Among the listed cities, which city has the highest population while being located in a state whose 2019 HDI is greater than 0.73? --> The phrase "Among the listed cities" is not clear enough to make the question fully self-contained.
        - Among the venues owned by AC Milan & Inter Milan for the 2026 Winter Olympics (as listed under a table in 'Milan cluster' section), which venue has the highest seating capacity? --> The note inside the parentheses explicitly refers to the table and section, which is not allowed.
        - Which venue in the Valtellina cluster has the highest total spectator capacity when the capacities of all its assigned events are summed together? --> Not fully self-contained.
        - What is the spectator capacity of the Verona Olympic Arena, the venue for the closing ceremony of the 2026 Winter Olympics? --> Too simple.


2. **SQL Implementation:**
    - Write a SQL query assuming the table is loaded as `df` and the engine is SQLite.
    - Ensure the query is syntactically correct and optimized for performance.
    - Always wrap column names with quotes, as some column names may contain spaces or special characters.
    - Ensure the query returns a single definitive value or a small result set.
    - The query must produce a non-empty result.
"""

class SQaGenerator(dspy.Signature):
    __doc__ = sqa_gen_description
    table: str = dspy.InputField(desc='Table in Markdown format')
    condition_col: str = dspy.InputField(desc='The column used as conditions/constraints for question generation.')
    metadata: str = dspy.InputField(desc='Metadata of the table, such as Wikipedia page title, section title, caption, hatnote, footnote, etc.')
    notes: str = dspy.InputField(desc='Some notes about some columns of the table (if any), which should be used for question generation to make the context of the question clear.')
    brainstorm: str = dspy.OutputField(desc='Your short brainstorming about the data and potential questions/queries')
    question: str = dspy.OutputField(desc='Natural-language question about the data')
    sql_query: str = dspy.OutputField(desc='SQL query to answer the question')

SQaGenerator_parser = lambda text: DSPyChatAdapter.parse(SQaGenerator, text)

SQaGenerator_demos = [
    dspy.Example(
        table="""
|    | team_home   | team_away   |   leg1_home_goals |   leg1_away_goals |   leg2_away_goals |   leg2_home_goals |   aggregate_home_goals |   aggregate_away_goals | decision_method   |
|---:|:------------|:------------|------------------:|------------------:|------------------:|------------------:|-----------------------:|-----------------------:|:------------------|
|  0 | Sevilla     | Deportivo   |                 2 |                 1 |                 3 |                 0 |                      5 |                      1 | standard          |
|  1 | Sporting    | Valladolid  |                 3 |                 1 |                 1 |                 2 |                      4 |                      3 | standard          |
|  2 | Poli Ejido  | Espanyol    |                 3 |                 2 |                 0 |                 1 |                      3 |                      3 | away_goals        |
|  3 | Real Unión  | Betis       |                 0 |                 1 |                 0 |                 1 |                      0 |                      2 | standard          |
|  4 | Mallorca    | Almería     |                 3 |                 1 |                 1 |                 1 |                      4 |                      2 | standard          |
|  5 | Racing      | Valencia    |                 1 |                 1 |                 1 |                 3 |                      2 |                      4 | aet               |
|  6 | Atlético    | Barcelona   |                 1 |                 3 |                 1 |                 2 |                      2 |                      5 | standard          |
|  7 | Osasuna     | Athletic    |                 1 |                 1 |                 0 |                 2 |                      1 |                      3 | standard          |""",
        metadata="""
{
    "page_title": "2006-07 Copa del Rey",
    "section_title": "Round of 16",
    "caption": "",
    "hatnote": "",
    "footnote": ""
}""",
        condition_col="decision_method",
        brainstorm="The table lists two-legged ties in the Round of 16 of the 2008-09 Copa del Rey with aggregate scores and the method used to decide the winner. A notable nuance is the 'away_goals' decision method, which only occurs when aggregate scores are level. Identifying the specific matchup that required this rule will highlight a unique case in the competition. Also, add the context of the competition and the season to make the question fully self-contained.",
        question="Regarding Round of 16 of 2006-07 Copa del Rey, which home and away teams were involved in the tie that was decided by the away goals rule?",
        sql_query="""
SELECT team_home, team_away
FROM df
WHERE decision_method = 'away_goals';
""",
    ),
    dspy.Example(
        table="""
|    | district             | province              | region       |   ubigeo |   elevation_m |
|---:|:---------------------|:----------------------|:-------------|---------:|--------------:|
|  0 | Suykutambo           | Espinar               | Cusco        |    80807 |          4801 |
|  1 | Condoroma            | Espinar               | Cusco        |    80802 |          4737 |
|  2 | San Antonio          | Puno                  | Puno         |   210113 |          4700 |
|  3 | Ananea               | San Antonio de Putina | Puno         |   211002 |          4660 |
|  4 | Morococha            | Yauli                 | Junín        |   120805 |          4550 |
|  5 | San Antonio de Chuca | Caylloma              | Arequipa     |    40514 |          4525 |
|  6 | Santa Ana            | Castrovirreyna        | Huancavelica |    90411 |          4473 |
|  7 | Marcapomacocha       | Yauli                 | Junín        |   120804 |          4415 |
|  8 | Capazo               | El Collao             | Puno         |   210502 |          4400 |
|  9 | Paratia              | Lampa                 | Puno         |   210707 |          4390 |
| 10 | Cojata               | Huancané              | Puno         |   210602 |          4355 |
| 11 | Yanacancha           | Pasco                 | Pasco        |   190113 |          4350 |
| 12 | Chaupimarca          | Pasco                 | Pasco        |   190101 |          4338 |
| 13 | Macusani             | Carabaya              | Puno         |   210301 |          4315 |
| 14 | Huayllay             | Pasco                 | Pasco        |   190104 |          4310 |
| 15 | Caylloma             | Caylloma              | Arequipa     |    40505 |          4310 |
| 16 | Vilavila             | Lampa                 | Puno         |   210710 |          4300 |
| 17 | Tanta                | Yauyos                | Lima         |   151028 |          4278 |
| 18 | Tinyahuarco          | Pasco                 | Pasco        |   190111 |          4275 |""",
        metadata="""
{
    "page_title": "Districts of Peru",
    "section_title": "By elevation",
    "caption": "",
    "hatnote": "(in Spanish) Instituto Nacional de Estadística e Informática. Perú en Mapas Archived 2009-11-13 at the Wayback Machine. Retrieved November 1, 2009.",
    "footnote": ""
}""",
        condition_col="(No condition column)",
        brainstorm="""
The table lists 20 Peruvian districts with their province, region, ubigeo code, and elevation in meters. The list is sorted, thus it seems to be top 20 districts of Peru by elevation. The associated document is from 2009.
A compelling question could compare elevations across regions, requiring aggregation (average elevation) and then identifying the region with the extreme (highest) average. This involves grouping by region, computing the mean elevation, and selecting the top result multi-hop reasoning and clear, deterministic answer.
To narrow down the scope, I can specify 'top 20 districts of Peru by elevation and their regions'. Also, I can mention the year of the report to ensure the question is self-contained and answerable with real information from Internet.""",
        question="According to an official report on 2009, if only considering top 20 districts of Peru by elevation and their regions, which region has the highest average of its districts' elevation?",
        sql_query="""
SELECT region
FROM (
    SELECT region, AVG(elevation_m) AS avg_elev
    FROM df
    GROUP BY region
) 
ORDER BY avg_elev DESC
LIMIT 1;
""",
    ),
    dspy.Example(
        table="""
|    | date        | opponent    | location   |   phoenix_goals |   opponent_goals | goalie    | attendance   |   record_wins |   record_losses |   record_ot |
|---:|:------------|:------------|:-----------|----------------:|-----------------:|:----------|:-------------|--------------:|----------------:|------------:|
|  0 | February 2  | Nashville   | Away       |               2 |                3 | Bryzgalov | 17,113       |            27 |              22 |           3 |
|  1 | February 4  | Colorado    | Away       |               4 |                3 | Tellqvist | 14,381       |            28 |              22 |           3 |
|  2 | February 5  | Calgary     | Away       |               3 |                4 | Bryzgalov | 19,289       |            28 |              22 |           4 |
|  3 | February 7  | Columbus    | Home       |               1 |                2 | Bryzgalov | 13,918       |            28 |              23 |           4 |
|  4 | February 10 | Nashville   | Home       |               3 |                6 | Tellqvist | 14,593       |            28 |              24 |           4 |
|  5 | February 11 | Dallas      | Away       |               1 |                2 | Bryzgalov | 17,622       |            28 |              25 |           4 |
|  6 | February 14 | Dallas      | Home       |               5 |                2 | Bryzgalov | 12,885       |            29 |              25 |           4 |
|  7 | February 16 | Los Angeles | Home       |               5 |                3 | Bryzgalov | 17,997       |            30 |              25 |           4 |
|  8 | February 18 | Los Angeles | Away       |               4 |                0 | Tellqvist | 16,617       |            31 |              25 |           4 |
|  9 | February 19 | Calgary     | Home       |               1 |                4 | Bryzgalov | 15,208       |            31 |              26 |           4 |
| 10 | February 22 | Colorado    | Home       |               2 |                3 | Tellqvist | 15,882       |            31 |              26 |           5 |
| 11 | February 24 | St. Louis   | Home       |               2 |                0 | Bryzgalov | 14,845       |            32 |              26 |           5 |
| 12 | February 27 | Chicago     | Away       |               0 |                1 | Bryzgalov | 14,799       |            32 |              27 |           5 |
| 13 | February 28 | St. Louis   | Away       |               2 |                1 | Bryzgalov | 17,867       |            33 |              27 |           5 |""",
        metadata="""
{
    "page_title": "2007-08 Phoenix Coyotes season",
    "page_summary": '''
The 2007-08 Phoenix Coyotes season began on October 4, 2007. It was the franchise's 36th season, 29th in the National Hockey League (NHL) and 12th season as the Phoenix Coyotes. The Coyotes failed to qualify for the playoffs for the fifth consecutive season.\nKey dates prior to the start of the season:\n
- The 2007 NHL entry draft took place in Columbus, Ohio, on June 22-23.\n- The free agency period began on July 1.''',
    "section_title": "February",
    "caption": "",
    "hatnote": "",
    "footnote": ""
}""",
        condition_col="goalie",
        brainstorm="The table lists each game of the Phoenix Coyotes franchise in February of the 2007-2008 season. It includes the goaltender, attendance (with commas), and whether Phoenix won (based on goals scored). A compelling query would compare the two goaltenders on the average crowd size they played in during victories, requiring us to filter for wins, clean the attendance values, aggregate by goalie, and then pick the highest average. Also add the context of the season and the month to make the question fully self-contained.",
        question="In February of the 2007-2008 season, considering the goaltender achieved the highest average attendance in games that the Phoenix Coyotes won, what were his name and his average attendance?",
        sql_query="""
SELECT
    goalie,
    AVG(CAST(REPLACE(attendance, ',', '') AS INTEGER)) AS avg_attendance
FROM df
WHERE phoenix_goals > opponent_goals
GROUP BY goalie
ORDER BY avg_attendance DESC
LIMIT 1;
""",
    ),
]



# 1.1. Check the quality of the generated question with LLM
question_quality_description = f"""
You are an expert in Q&A.

You are given a question, which is expected to be complex, self-contained, and natural.
Your task is to briefly discuss the question in the following criteria, then on a scale from 1 to 5 (1 is worst, 5 is best), rate the question on each criterion:
- Complexity: The question involves multi-hop reasoning and information aggregation.
- Self-containedness: The question provides all necessary parameters, definitions, and background information within itself, requiring no external links, data, references, or prior knowledge to be **understood**. Using the question, people can expect to search for all necessary information to answer it from the Internet. 
- Naturalness: The question is as natural as being asked by real human.
"""
# - Singularity: The question is a single, well-defined question, not multiple questions concatenated together. --> No need to be strict on this criterion.
# - Groundedness: The question has a deterministic, single answer that can be found in and answerable using only in the table without external information. --> Checked with SQL and Python
# - Self-containedness: The question provides all necessary parameters, definitions, and background information within itself, requiring no external links, data, references, or prior knowledge to be **understood**. Using the question, people can expect to search for all necessary information to answer it from the Internet. (not tested with this criteria description)
# Self-containedness: The question provides all necessary context, definitions, and background information within itself, requiring no external links, data, references, or prior knowledge to be **understood**.

class QuestionQualityChecker(dspy.Signature):
    __doc__ = question_quality_description
    question: str = dspy.InputField(desc='A question to be evaluated for its quality')
    # metadata: str = dspy.InputField(desc='Metadata of the data that the question refers to, such as Wikipedia page title, section title, caption, hatnote, footnote, etc.')
    discussion: str = dspy.OutputField(desc='A brief discussion on the quality of the question based on the criteria')
    complexity_rating: int = dspy.OutputField(desc='Your rating (1-5) on the complexity of the question')
    self_containedness_rating: int = dspy.OutputField(desc='Your rating (1-5) on the self-containedness of the question')
    naturalness_rating: int = dspy.OutputField(desc='Your rating (1-5) on the naturalness of the question')
    # revision: str = dspy.OutputField(desc='Revised question the self-containedness of the question can be improved')

QuestionQualityChecker_parser = lambda text: DSPyChatAdapter.parse(QuestionQualityChecker, text)

QuestionQualityChecker_demos = [
    dspy.Example(
        question="Among the listed cities, which city has the highest population while being located in a state whose 2019 HDI is greater than 0.73?",
        metadata="""{"caption": "", "header_levels": ["Largest cities or towns in Venezuela [345]"], "footer_info": [], "page_title": "Venezuela", "section_title": "Largest cities", "hatnote": ["Main article: List of metropolitan areas in Venezuela", ["/wiki/List_of_metropolitan_areas_in_Venezuela"]], "references_content": [["^ Based on the result of the 2011 Census according to the Instituto Nacional de Estadisca", ["#cite_ref-351"]]]}""",
        discussion="The question involves a multi-hop constraint: filtering cities by their state's HDI threshold and then finding the one with highest population. It also sounds fluent and natural. Nonetheless, though having a named parameter year (2019) and threshold (0.73), the main object scope 'Among the listed cities' of the question is vague, making the question not self-containted.",
        complexity_rating=5,
        self_containedness_rating=2,
        naturalness_rating=4,
        revision="Among the largest cities or towns in Venezuela, which city has the highest population while being located in a state whose 2019 HDI is greater than 0.73?"
    ),
    dspy.Example(
        question="Among the 20 largest Venezuelan cities shown, which state accounts for the greatest combined population and what is that total population?",
        metadata="""{"caption": "", "header_levels": ["Largest cities or towns in Venezuela [345]"], "footer_info": [], "page_title": "Venezuela", "section_title": "Largest cities", "hatnote": ["Main article: List of metropolitan areas in Venezuela", ["/wiki/List_of_metropolitan_areas_in_Venezuela"]], "references_content": [["^ Based on the result of the 2011 Census according to the Instituto Nacional de Estadisca", ["#cite_ref-351"]]]}""",
        discussion="The question involves multi-hop reasoning: identifying the state for each city, aggregating populations by state, and then comparing totals to find the maximum. However, the question is not fully self-contained, as the phrase 'Among the 20 largest Venezuelan cities shown' has the word 'shown' refering to an external and not mentioned source. More importantly, the question about combined population by state given just a list of cities rather than all cities in the country is NOT natural to ask.",
        complexity_rating=4,
        self_containedness_rating=3,
        naturalness_rating=2,
        revision="(Cannot be revised to be more self-contained without changing the semantics)"
    ),
    dspy.Example(
        question="Among the venues owned by AC Milan & Inter Milan for the 2026 Winter Olympics (as listed under a table in 'Milan cluster' section), which venue has the highest seating capacity?",
        metadata="""{"caption": "", "header_levels": [], "footer_info": [], "page_title": "2026 Winter Olympics", "section_title": "Milan cluster", "hatnote": ["Main article: Venues of the 2026 Winter Olympics and Paralympics", ["/wiki/Venues_of_the_2026_Winter_Olympics_and_Paralympics"]], "references_content": []}""",
        discussion="The question involves multi-hop reasoning: filtering venues by joint ownership (AC Milan & Inter Milan), then comparing capacities to find the highest. It includes specific constraints about venue ownership and the 2026 Olympics context. It is self-contained with time- and event-specific information. However, the note inside parentheses are redundant and making the question less natural. The note can be removed without losing any semantics.",
        complexity_rating=3,
        self_containedness_rating=4,
        naturalness_rating=3,
        revision="Among the venues owned by AC Milan & Inter Milan for the 2026 Winter Olympics, which venue has the highest seating capacity?"
    ),
    dspy.Example(
        question="Which venue in the Valtellina cluster has the highest total spectator capacity when the capacities of all its assigned events are summed together?",
        metadata="""{"caption": "", "header_levels": [], "footer_info": [], "page_title": "2026 Winter Olympics", "section_title": "Valtellina cluster", "hatnote": ["Main article: Venues of the 2026 Winter Olympics and Paralympics", ["/wiki/Venues_of_the_2026_Winter_Olympics_and_Paralympics"]], "references_content": []}""",
        discussion="The question involves complex multi-hop reasoning: filtering by geographic cluster (Valtellina), aggregating event capacities per venue, and then finding the maximum. It requires both aggregation and comparison operations, making it highly complex. The location-specific constraint makes it somewhat clear, yet the question lack of specific referred event, e.g., '2026 Winter Olympics'. Importantly, the question about summed capacities of a venue across assigned events is NOT natural to ask. It needs some reason, e.g., utilization, to make sense.",
        complexity_rating=4,
        self_containedness_rating=3,
        naturalness_rating=2,
        revision="Which venue in the Valtellina cluster has the highest utilization in the sense of summed capacities across all assigned events?"
    ),
    dspy.Example(
        question="What is the spectator capacity of the Verona Olympic Arena, the venue for the closing ceremony of the 2026 Winter Olympics?",
        metadata="""{"caption": "", "header_levels": [], "footer_info": [], "page_title": "2026 Winter Olympics", "section_title": "Verona", "hatnote": ["Main article: Venues of the 2026 Winter Olympics and Paralympics", ["/wiki/Venues_of_the_2026_Winter_Olympics_and_Paralympics"]], "references_content": []}""",
        discussion="The question is straightforward, natural, asking for a single attribute (seating capacity) of a specifically named venue with a clear role in the event. While it has lower complexity as a direct lookup, it is self-contained with specific details about the 2026 Olympics ceremony venue.",
        complexity_rating=2,
        self_containedness_rating=5,
        naturalness_rating=5,
        revision=""
    ),
    dspy.Example(
        question="Among the National Olympic Committees that won at least one medal at the 2026 Winter Olympics, which NOC achieved the highest medals-per-athlete ratio, and what is that ratio?",
        metadata="""{"caption": "", "header_levels": [], "footer_info": ["2,880", "Total"], "page_title": "2026 Winter Olympics", "section_title": "Number of athletes by National Olympic Committee", "hatnote": ["Main article: 2026 Winter Olympics closing ceremony", ["/wiki/2026_Winter_Olympics_closing_ceremony"]], "references_content": [["^ Cite error: The named reference Suzanne Schulting was invoked but never defined (see the help page ).", ["#cite_ref-Suzanne_Schulting_78-0", "/wiki/Help:Cite_errors/Cite_error_references_no_text"]]]}""",
        discussion="The question involves significant multi-hop reasoning: filtering NOCs by medal achievement, calculating the ratio of medals to athletes for each NOC, and identifying the highest ratio. It requires both filtering and computation of a derived metric, making it highly complex. The wording is fluent and natural. The specificity of the 2026 Olympics event, the object of NOC, and the derived metric makes it self-contained.",
        complexity_rating=5,
        self_containedness_rating=5,
        naturalness_rating=5,
        revision=""
    ),
]


# 2nd LLM call: Generate Python code
py_gen_description = f"""
You are an expert in table question answering and problem solving with Python.

You are given a table in Markdown format, some metadata about the table, and a complex question about the table.
Your task is to generate a Python implementation to answer the question. You must name the final answer variable `final_answer_python`.

The answer is expected to be a specific value (a short string, a number) or a small set of information (a list or an entity-value dictionary with no more than 3 elements or key-value pairs). Please only pack the final answer in `final_answer_python` variable, and avoid putting any narration in `final_answer_python` (e.g., "the answer is ...", "the result is ...", etc.).

**Guidelines:**
  - Use `pandas` to solve the problem, **assuming the table is already loaded into a DataFrame named `df`**.
  - Skip the data loading step.
  - **Break complex operations into small steps.** Each step should compute one logical operation.
  - Assign each intermediate result to a named variable (e.g., `step1_filtered`, `step2_grouped`).
  - **Do not reuse the same variable name for different steps**.
  - If the change is in a row or column of an existing variable, create a new variable to store the updated version (e.g., `step1_filtered` -> `step2_filtered`).
  - Each step should have a unique variable name and should reflect the meaning of the data in the variable.
  - Display the intermediate results and final result by printing them.
  - Comment your code to explain each step clearly.
  - **Use the provided context about the table to narrate your solution**.
  - Name the final answer variable `final_answer_python`.
"""

class PyGenerator(dspy.Signature):
    __doc__ = py_gen_description
    table: str = dspy.InputField(desc='Table in Markdown format')
    metadata: str = dspy.InputField(desc='Metadata of the table, such as Wikipedia page title, section title, caption, hatnote, footnote, etc.')
    question: str = dspy.InputField(desc='A complex question about the data')
    python_code: str = dspy.OutputField(desc='Python code using `pandas` to answer the question')

PyGenerator_parser = lambda text: DSPyChatAdapter.parse(PyGenerator, text)


# 3rd LLM call: Compare the two resulting objects
df_comparison_description = """
You are an expert in data analysis.

You are given two results (tables, values, or dictionaries) from answering the same question about a dataset.
Your task is to compare if the content/meaning of the two results are essentially the same.

**Metadata is provided** to help you understand what the question was asking and what the results represent.

**Comparison Guidelines:**
- The two results may have different formatting (e.g., table vs. dictionary, different column order, different indexing)
- They may have slight numeric differences due to floating-point precision (e.g., 83.6 vs 83.60000000000001)
- As long as the semantic content of results regarding the question are the same, consider them equivalent
- Focus on whether both results answer the question correctly

**Examples of equivalent results:**
- "Espanyol" vs. {"team_home": "Poli Ejido", "team_away": "Espanyol", "team_advances": "Espanyol"} if the question is "Which team advances to the next round?" and both results indicate Espanyol advances
- {"party": "Democratic", "avg": 83.6} vs {"party": "Democratic", "avg": 83.60000000000001}
- A single-row table vs a dictionary with the same key-value pairs
- 248Cm vs. {}^{248}Cm vs. 248Cm (i.e., {}^{248}Cm): all of them represent the same chemical element isotope, just different in the way of writing.

Give a binary answer (True or False) and a short explanation for your judgment.
"""

class DfComparison(dspy.Signature):
    __doc__ = df_comparison_description
    question: str = dspy.InputField(desc='The original question being answered')
    metadata: str = dspy.InputField(desc='Metadata of the data, such as Wikipedia page title, section title, caption, hatnote, footnote, etc.')
    result1: str = dspy.InputField(desc='The first result (from SQL)')
    result2: str = dspy.InputField(desc='The second result (from Python)')
    judgment: bool = dspy.OutputField(desc='Your binary judgment (True or False) on whether the results are semantically equivalent')
    explanation: str = dspy.OutputField(desc='A short explanation for your judgment')

DfComparison_parser = lambda text: DSPyChatAdapter.parse(DfComparison, text)


# 4th LLM call: Interpret the Python code as a guidance of how to solve the question
code_interpretation_description = """
You are a data analyst and a complex thinking mentor.

You will be given:
- a metadata of the data,
- a complex question about the data, and 
- a Python-implemented solution.

Your task is to interprete the Python-implemented solution as a natural-language guidance of how to solve the question, so that your mentee will understand the logic and flow of the solution.

**Strict Requirements:**
- You should explain step by step, breaking down the code into small logical steps.
- Avoid the use of words which refer to the data structure, e.g., dataframe, table, row, column.
    - As if you are explaining the solution to a non-technical mentee who doesn't know programming or data structure terms, but can understand natural language and the logic of problem-solving.
- Avoid excessive markdown highlighting.
- For each step, format the interpretation in three blocks:
    1) Purpose and action in natural language:
        + Describe in clear natural language, 
        + Avoid data-structure-specific terms,
        + Focus on the logic and reasoning behind the step.
    2) The corresponding code snippet to support the logic:
        + Wrapped it inside ```python ...``` code block for better readability.
        + Keep the comments in the code snippet to preserve the explanation of the code.
    3) The resulting intermediate variable:
        + Display in a seperated line,
        + Using the format {variable_name} (i.e., variable name enclosed by curly braces).
        + If the outcome of the step is not value but a function or a data structure, you must skip this block.
"""

class CodeInterpretation(dspy.Signature):
    __doc__ = code_interpretation_description
    metadata: str = dspy.InputField(desc='Metadata of the data, such as Wikipedia page title, section title, caption, hatnote, footnote, etc.')
    question: str = dspy.InputField(desc='The question about the data')
    python_code: str = dspy.InputField(desc='A Python-implemented solution of the question')
    interpretation: str = dspy.OutputField(desc='Step-by-step interpretation of the solution')

CodeInterpretation_parser = lambda text: DSPyChatAdapter.parse(CodeInterpretation, text)
