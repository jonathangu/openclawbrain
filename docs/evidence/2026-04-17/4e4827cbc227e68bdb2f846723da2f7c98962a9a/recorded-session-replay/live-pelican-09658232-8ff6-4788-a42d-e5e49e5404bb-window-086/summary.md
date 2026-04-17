# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c70b7e6acafa9f174da3df163120ba16044bc767e199909b1a7b96f75ed37549`
- fixture hash: `sha256-bf91f869d3956bf5fde31cf4fcbfa13c4356f4c344c72e681c59e051bd04b628`
- score hash: `sha256-7ac46e6a279935b182c0196a25585f23a1ff1f3c880b280ff0e29bc96e503c50`
- bundle hash: `sha256-2d6b24bdad6646c7c4b1843c3b7420809468d6f9c6b1a49c5a51140278926060`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b139f94f37d6885531ef5b31e5bde18e900dc87fd64f0c8059b9943917b139d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f78f8d1c0442a5b39f3eecf1b4dfe9cbc6f7af28a1c24d3211ab2da740657011 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88395f50a19fa0cb55c827b5bd0f8d44bb25b49d8712e89d85146cd5f782edb3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ab224013e1487d6e80504e7b47847230b5b956313a0eaf92df43c55a900f15a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba26cba9 | sha256-c9e47bc756c087fe740b554f97e83bcf688991df612bf58f89c81a620f641456 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba26cba9 | sha256-b6e3bed0b587fbf521f4c943f773c1617a3d15c6cd4f446e3a00492e6f2b229a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-941580ac | sha256-fe80da2d84e8b8ac8503f2c45b244007e37d9278da4293638c913ef594026e7e |
