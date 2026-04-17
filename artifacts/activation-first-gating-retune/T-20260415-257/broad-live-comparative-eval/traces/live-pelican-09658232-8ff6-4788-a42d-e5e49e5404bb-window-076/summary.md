# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e90ba91fa2d821b34e7d50d49031d2ca2e725469eba7413ed1eefcf887d0f975`
- fixture hash: `sha256-b66c57ad146f945a1113822081ae1bceec873a0abb858cfb6bafe580d07b22c8`
- score hash: `sha256-a561bd68aa166d66fa75a78ba176d1adaa73a21d5e0c907cc251eddf05fc1150`
- bundle hash: `sha256-ca412b52ce2b63eb4a7ddc67feed3cd455f7270bfa1da691f8edb84dfa063c26`

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
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d7a8f01e83ed8ac33586c073703951c8627b99bf4e9aa0272b865992ce2738f9 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d3bf602e2ae993be59b7ff3f4208d481fa2f0149db347a023634b4d59fc7aab6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a939553b276d493808290226d62489d366496adaa27fc8c49a49f0b5e046cb54 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5537978664b49d9e24d76fbba06076ca9469d69254b5d5c0b052f2b902e4aaac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-12c2c2eb | sha256-42114672babf5493ca407e544b716fe63e71ed46969696d5ee1321714eabb719 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-12c2c2eb | sha256-7aaad18309254f5091a66319b15c89ec7aa2513fc01aec6ebe44117e71baf5cc |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3701aaa0 | sha256-6415565ec426a35f30fde698cfa62ee33c0da596d6fc5f24baeaab161fa302bd |
