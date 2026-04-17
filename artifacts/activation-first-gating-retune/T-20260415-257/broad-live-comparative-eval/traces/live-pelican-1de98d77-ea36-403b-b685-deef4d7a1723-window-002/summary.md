# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b39ef4fc4945a82dff034380c9080960d0e6ed5fe56fe5b4657351529db21cd7`
- fixture hash: `sha256-a795947af952aa839da230500896d2e52bf78e338ce72dd740b6a925befadf59`
- score hash: `sha256-ca723012b95240a58794fefdecf806379dd3da417ba8d0f1a0767132b4874e83`
- bundle hash: `sha256-10ed20a70f9117a050d440ca2d15b407a0e4f3fe45d3665699123205de0b359b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64e7031bab11acf7ca7c6563e45ebf707e8feb9b8d59eced338f7e5e56bc854a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-07c9fa992065da03035ebdcb18722c1ee4091717a09f12a9bf1cc6281032aa24 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-54002d23808bbf18c4f4b445604c09641b8f661f65b1573434a36a7f93c0f8b5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-61dc0b1d0474a2ecf4275329aba9ad85c575b95530c7f440abb1cac9e07a0813 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c31f4cc3 | sha256-7d3685070083cc850b758d13fb480a7a803ea60a86b0ff3cc763ba5715216909 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-c31f4cc3 | sha256-6247093e83779e81e971914fc0657b866f27203b2b1a4fb9e7f64688fa398658 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9e34a4d8 | sha256-24a4e2e9c501f6295687291b4e16184175ac2bfa085cb6b466d5d2c6c6a06331 |
