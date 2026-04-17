# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-823695d70f7872b1ae9eafb6d1d27250c7a30f3c8da0fb3fac149eb03366ef43`
- fixture hash: `sha256-bd84df8e56b4c53a26fb492fdea7511a22aab4ac1b787c58633c40d2b1aa4455`
- score hash: `sha256-985c00a70ef4c48eb5c0bfa53c55f8eaa91593a4cd6407243f86f7bc9454b455`
- bundle hash: `sha256-92b5637da04eef0ff62b3edd9c2159b9b7586d5894f5dfa216bc535d4d954d87`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 70 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/8
- phrase hit rate: 0.125

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a1db4bbe90ab058f57bc7ae6a54f5aaf2daac0fc5ad242f5b0e6f3a965eb8e61 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-81cc8a5ab9e464c87576247420fc5afa939361d9710c781ba9ec1a294b51f4fe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-741e6d77dc33af9b84fb4c02920d7de4c953d6249337c611ecf3f9a5b409aa24 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1ccd0d243a9f409c838fdf51a3d35b4f7d8b3037aba0916ed533ed2f3aa53658 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-6011576d | sha256-5388291f9f554669172609dde75dc7280ab7b835d1d280f294206378f6a09da5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6011576d | sha256-57ba7674228f6e86b221cb507e2bfb9b1ae5baba089815bfb65b5f803b7de3ae |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-e9841ca6 | sha256-ace555e3e8ed2a4132d9146a3c057e4a87b1770d1f69744c9ddfb1bd292ea133 |
