# Recorded Session Replay Proof Bundle

- trace id: `real-trace-plan-execution`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ffe77287785709ddf30daa15b90604efbf2488116d6838dc758339e02ae313c`
- fixture hash: `sha256-66e6a6c2557f40f0dad6f86cf26a94325382032443bb05d591838be08e44ba88`
- score hash: `sha256-057f725813571fd061306ff14982a287f39ef6306faa6d1a8b5d2f05232f872c`
- bundle hash: `sha256-f42c989802fdba3db7d588977d2ee5eb3992a4e7e119cc47bacf53b339b69bba`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 7/16
- phrase hit rate: 0.4375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 0.75 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-271076f35fa438fafb2771d3e4fdf49b2bf41b0468ccbbb99a0d1f5bee4f354a |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-0fb936da0077a9490ddf8dc5143aedb3eddf24a3085d9fc4026b54f73bd25353 |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-7b394491a3113b2d59e2a4ed81a87e53281c76761f557419060142570312cc06 |
| learned_route | 3 | 3 | 3/4 | 2 | 2 | 3 | 2 | 0 | sha256-2937102dfb538596b20fa1da4873abd85a182f7584330c8d8a1290b029bb4941 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | plan-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| vector_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| vector_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| graph_prior_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| graph_prior_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| graph_prior_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| learned_route | plan-turn-1 | 100 | yes | 1/1 | no | yes | pack-2dd625f8 | sha256-9b9b8162cb19180ac2c20fd3328c6876bc3fc58390234331d8ced337dd177f5f |
| learned_route | plan-turn-2 | 40 | yes | 0/1 | yes | yes | pack-c71b234b | sha256-6dd3dafa608bfc6da4282a38e09a1599828656786d457f5a5bb467ce975fe802 |
| learned_route | plan-turn-3 | 100 | yes | 2/2 | yes | no | pack-1eb69ca1 | sha256-9c9e0225bf216474ab9c6c80526a16da8ebca0a5f69338c9a8a5d7293271dafe |
