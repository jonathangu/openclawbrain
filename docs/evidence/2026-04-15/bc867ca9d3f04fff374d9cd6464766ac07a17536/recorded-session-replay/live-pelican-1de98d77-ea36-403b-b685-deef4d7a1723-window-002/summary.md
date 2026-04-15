# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b39ef4fc4945a82dff034380c9080960d0e6ed5fe56fe5b4657351529db21cd7`
- fixture hash: `sha256-a795947af952aa839da230500896d2e52bf78e338ce72dd740b6a925befadf59`
- score hash: `sha256-003cdf5cde00cf2725864c30a2da469791dbd7ed24e4b9adb67654c175dec39b`
- bundle hash: `sha256-c061cc1449b96e67ba2b97eca4fbfce1b39d7646241eeab4deb55efce5b54d15`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64e7031bab11acf7ca7c6563e45ebf707e8feb9b8d59eced338f7e5e56bc854a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ef07f7d009d1e8322dbcc8d79db46680781c8907bdc24e75fb53325ab9d4ef2b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-cf98b9e95980261c364505f46d74b4b7cbb2200bc932abb028f333cd5b06c67a |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-b5eaa80a558a6424e8d8769585939c1501dc5f5daa5ee6d8193e2a40a4a923e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4bcf0739 | sha256-7476ab3b1f61e3a4a779c4ddba67429c530b6f7d634438dd800fb7c2a504555e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4bcf0739 | sha256-f5a83ae5845983faff550ffb146c0e4d7b802ea1aacd27af50b4320394179a9d |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-4bcf0739 | sha256-7476ab3b1f61e3a4a779c4ddba67429c530b6f7d634438dd800fb7c2a504555e |
