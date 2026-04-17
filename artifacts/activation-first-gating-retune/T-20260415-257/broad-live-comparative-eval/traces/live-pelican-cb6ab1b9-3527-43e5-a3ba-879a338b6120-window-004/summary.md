# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f94a31a75f6f674deb4ed72bb4e73c45b90c17561480d33d8d146b93540cfdaf`
- fixture hash: `sha256-55ad28e1e1c0e357b90d71c5a61455a338c2e0a4ef3a7f6c092d3616039ed272`
- score hash: `sha256-50b4f68b82c87251145ac3f1efd33a1b71b30c64dfca7e966a23450da8a083f9`
- bundle hash: `sha256-82441119b2b535979ec501190d50821079d26ffdcbbaad8882cfb7b9454b2c2b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a7e02cf5e88271092f868ed8daefe51bb787b99a9e0166c0444d9f0e9eabb76 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-385c6d6d1f46ab368fbee800fd6ab1ce513e15e603ff64693e10ee4334d6c264 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e13aca932a9587930dc4ebeedeeccff4ad48cef05b79916ed373164627c7ffac |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-97287761e969e72614923295e1c49797d77766c04d490b37d93b8fc8eeee9e20 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5244cfbd | sha256-fe39b738fb22bef4469e62f7d27c9f88288b99ef3f1b76de8c582ccb0cb23772 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5244cfbd | sha256-9286e00a0cad1e44f5d7704aa8251aacc5c4215e6f6927725b594db3bce57f36 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2de53c82 | sha256-b6720149ff9ab40caa7fb3d41de43c2887ec736c785637a01799e7774dac6914 |
