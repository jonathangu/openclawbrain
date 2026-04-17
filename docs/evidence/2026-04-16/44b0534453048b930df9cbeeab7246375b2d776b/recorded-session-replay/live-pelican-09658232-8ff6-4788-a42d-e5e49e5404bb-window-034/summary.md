# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-88c322f4d12a10fc7fd9587ce37571bf5cf495196ecf9676587806d45fb2c729`
- bundle hash: `sha256-c63c2a861b8096b4fea8a5fa5b4b4bcf487545bf339c40da1a54d17e8f7cd935`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd552a1020dae7a8b929bbcdeb6c903b23f283ae3a9cd2bb5fbca79abfcd14ea |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5b6c1e61b6d0ad28559e5896a82c00b8a4a234f5df78d987cb05feec29bb0cc3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-fdb6b6f7e72f3eeb4049cbeddf81226eafb722eda899cd14b98b8ce941916140 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4218750f | sha256-30f8dc73f12ba045ad6c13df1d1b5a786c7c973e458cf742a55565d84632a17a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4218750f | sha256-21730a3bcd6b62ea6464ed687c345c6b423526bf7a76af2d83c1ad205eef56a2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0d22dc0e | sha256-0be69f1b99973c9ab6878dd45ee12baec5142fa1a7e828ef660b1fd491073193 |
