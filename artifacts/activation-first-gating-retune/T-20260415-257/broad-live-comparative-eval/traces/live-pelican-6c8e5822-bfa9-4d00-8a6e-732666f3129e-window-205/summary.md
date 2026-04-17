# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-9e2ba734958c647d2b52c5a5d64322b166e0e5bd4ddfbd84d19e807ff0f7a538`
- bundle hash: `sha256-549960008043f85e89a30fd6c2d6819ef21a09b89940fd0e20b9bf714e71f318`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-86f98e13f5486f34d07aee2d052d188bccd38735cc31a3e58504ac97500619fd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fee5728e493797eb2480893ccfb11a9d61ababb50e0f89bafb62ab6aa4b5c88 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-94033e2baf5acf96c019eae9b784bf3d9e970a27c040397ee0a09766fba6c3db |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c29cadc4 | sha256-bf6c831910541051d4fc16b68172d90b510bbf63487a413ae7dc822529269227 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c29cadc4 | sha256-e735f9779c41b8b65d43ed252085f1567218a617209ed68c698defc1c2d8f017 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-05a20a13 | sha256-6fb08e4e3f3c9f1777732aeb6b310b5e46f33864be3b6b6bb2b05f4fe92c753c |
