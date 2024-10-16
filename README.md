# TweetNERD - End to End Entity Linking Benchmark for Tweets

This is the dataset described in the paper **TweetNERD - End to End Entity Linking Benchmark for Tweets** (to be released soon).

> Named Entity Recognition and Disambiguation (NERD) systems are foundational for information retrieval, question answering, event detection, and other natural language processing (NLP) applications. We introduce TweetNERD, a dataset of 340K+ Tweets across 2010-2021, for benchmarking NERD systems on Tweets. This is the largest and most temporally diverse open sourced dataset benchmark for NERD on Tweets and can be used to facilitate research in this area.


TweetNERD dataset is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) LICENSE.

The license only applies to the data files present in this dataset. See **Data usage policy** below. 


## Usage

We provide the dataset split across the following tab seperated files:

* **OOD.public.tsv**: OOD split of the data in the paper. 
* **Academic.public.tsv**: Academic split of the data described in the paper. 
* `part_*.public.tsv`: Remaining data split into parts in no particular order.

Each file is tab seperated and has has the following format:

| tweet_id            | phrase    | start | end | entityId  | score |
|---------------------|-----------|-------|-----|-----------|-------|
| 22                  | twttr     | 20    | 25  | Q918      | 3     |
| 21                  | twttr     | 20    | 25  | Q918      | 3     |
| 1457198399032287235 | Diwali    | 30    | 38  | Q10244    | 3     |
| 1232456079247736833 | NO_PHRASE | -1    | -1  | NO_ENTITY | -1    |

For tweets which don't have any entity, their column values for `phrase, start, end, entityId, score` are set `NO_PHRASE, -1, -1, NO_ENTITY, -1` respectively. 

Description of file columns is as follows:


| Column   | Type   | Missing Value | Description                                                                                           |
|----------|--------|---------------|-------------------------------------------------------------------------------------------------------|
| tweet_id | string |               | ID of the Tweet                                                                                       |
| phrase   | string | NO_PHRASE     | entity phrase                                                                                         |
| start    | int    | -1            | start offset of the phrase in text using `UTF-16BE` encoding                                          |
| end      | int    | -1            | end offset of the phrase in the text using `UTF-16BE` encoding                                        |
| entityId | string | NO_ENTITY     | Entity ID. If not missing can be NOT FOUND, AMBIGUOUS, or Wikidata ID of format Q{numbers}, e.g. Q918 |
| score    | int    | -1            | Number of annotators who agreed on the phrase, start, end, entityId information                       |

In order to use the dataset you need to utilize the `tweet_id` column and get the Tweet text using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api) (See **Data usage policy** section below).



## Data stats

| Split    |   Number of Rows |   Number unique tweets |
|:---------|-----------------:|-----------------------:|
| OOD      |            34102 |                  25000 |
| Academic |            51685 |                  30119 |
| part_0   |            11830 |                  10000 |
| part_1   |            35681 |                  25799 |
| part_2   |            34256 |                  25000 |
| part_3   |            36478 |                  25000 |
| part_4   |            37518 |                  24999 |
| part_5   |            36626 |                  25000 |
| part_6   |            34001 |                  24984 |
| part_7   |            34125 |                  24981 |
| part_8   |            32556 |                  25000 |
| part_9   |            32657 |                  25000 |
| part_10  |            32442 |                  25000 |
| part_11  |            32033 |                  24972 |


## Data usage policy

Use of this dataset is subject to you obtaining lawful access to the [Twitter API](https://developer.twitter.com/en/docs/twitter-api), which requires you to agree to the [Developer Terms Policies and Agreements](https://developer.twitter.com/en/developer-terms/).