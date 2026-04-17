# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c46bcb381e0fc3efac0e09438f41834359285b619fa2a6877dd357e64e821071`
- fixture hash: `sha256-1ef85417b722b0f394a6f903af4947a78bca3d01432416a0cb17a206ec104c37`
- score hash: `sha256-ae18f3a83d0e5f6e12b397aa846627c91bff1cf65190a364be6af487b47a0755`
- bundle hash: `sha256-717df485683ab8e3d16c4499571f2887e3121a1886234f65861a757b39bfb971`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d30e174da168a75e39ffd3536c03dfe75f2623e328eaa1807c5de3d00819572a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bccc12dfb18009629661dc4faaa11e0b4ee079bed9dd89f42c27ea9e6b386282 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4bd64970ce069b01d17d35fad3918072bcb6ae6d72749b8eacbc8e62621c0dc |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5146bd4093267dcac1f908ea753d60b8533c75b18da2b5a4fc26e4bcd5855cc3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f538735b | sha256-84682fa9572a58309d569d658c55f2d6cbf7dfa3058ce9c099bcb82b8691b9c0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f538735b | sha256-467be63bca53e0ef064ccdebf39fcc77c2b404225f6ae74b9609927f4b084839 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-e4093802 | sha256-6ee0190fd2d5f46e5a31ddbd6adf68b04d60ee4b6ea1b6560909f8dba41fbc63 |
