# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-76273bf572d2d6df7f5708306c7a325e0d7fc022256a8c48664d2d2c99f93d6b`
- fixture hash: `sha256-5c64fa2fb4319875db5a6403e087c56d2c1a468e1ff9a819a4e71ac1b0668ff8`
- score hash: `sha256-2a2c9e0cac70cde93ed838dac10e1bff8c05f427a61fb46fb1f343f0b752b08e`
- bundle hash: `sha256-9592bea1f70281d43b07f473e3016081e5b3e65c21bdd81af5f2c4367951ef70`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd39b973453e541375d3824b8f2b46f3993347e9ee385b937ee54648b5838113 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7e940e7e8c1462630b77b49ef3c617ac7ba8c9d84680cfa5d1f92a1d528794ce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e535f8a708ea19602b44249780e13e2d6ac8b9bd3bbdfd2f601397d928b372d4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fd2afbfef1438f7176e62c84ba7b763f0f0af06fe8347f82619292e9c31b91ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-426e966c | sha256-51638a8b2f1e846952923d121d32b655e0a5757d68801c1e781d99a8828741d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-426e966c | sha256-51638a8b2f1e846952923d121d32b655e0a5757d68801c1e781d99a8828741d3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d89a2ca9 | sha256-8851a073ff4f50f83046fc0f5d4dd2195b8be90e7744e372a4926b4fd63fea0b |
