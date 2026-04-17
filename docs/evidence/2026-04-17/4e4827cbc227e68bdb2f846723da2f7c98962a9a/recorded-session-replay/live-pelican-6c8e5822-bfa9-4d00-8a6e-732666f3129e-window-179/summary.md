# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-40f160b5df33d2495dd2561fc5b6e9d09096faa6a1778fb2e501b92787702234`
- bundle hash: `sha256-467389dc39f81b740d994ce7c35b4347549afa605d2434e9bc1803ff776d2ab7`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0e2397208053e0ee828148f13e43ac00df5c035e81ae074b8973687a1cc079f4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c42a5acbf8ab29bc6e05f91248a598df90d6d7f5b5a2c90265644c362a213bc5 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-61fdfe8e4adaef9396e2b6d49aaa3b64fd3c70fdec7ed55a9f81892fe404c12f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f8e784bc | sha256-17cdc2db257a4a4d15ba005b96ee77372f93414d3b8e9c60602eff833f6113bd |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f8e784bc | sha256-8e6c8aa2892eed3c4a2e9e84dbcf5d74047ae02dbc4b8d47f4a151361dfbe710 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-9f1632f1 | sha256-37e9d8319d80d50042ee1c67955f160b998edc2c9dfc02fd0adcf573b0b4fa67 |
