# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b39ef4fc4945a82dff034380c9080960d0e6ed5fe56fe5b4657351529db21cd7`
- fixture hash: `sha256-a795947af952aa839da230500896d2e52bf78e338ce72dd740b6a925befadf59`
- score hash: `sha256-65fdf89e7d947b87ff8c4173501e1efa6ac202fc83ae0b57680de24657fb725a`
- bundle hash: `sha256-af3ca565c0495241da191a1bedc6c82494425301cab47b2b3e7cfcaca7c71002`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-448a18792ccaa2132160baf08b93d23706df318e65329e5d5648976decad1057 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1040ee7b9e9d15988b03ea511df9e107b7dc94c35584e7099d6f0876bde4b65b |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-9f842eb4d99f16b7dbafc663299ebe0162f5b263992f00f601b5b03ab84bf21c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-68f1e001 | sha256-2656d1d1a3e426b6ad571126a544fa3b855ef4e8bb58db3e2ce22af6f2e4c3a2 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-68f1e001 | sha256-dcc64fb5e8813516419a55386c842cebf1bbf4a074f07d9e6590f6c7123a19fb |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-44073816 | sha256-24c43d24c3c172d575cbba7ee3f2a6c989c398548400f55033a8bfeea3da9c47 |
