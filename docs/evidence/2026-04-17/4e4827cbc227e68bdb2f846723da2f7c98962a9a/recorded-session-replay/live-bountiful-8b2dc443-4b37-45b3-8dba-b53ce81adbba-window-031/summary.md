# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c591f5c65a17f8b728581b0a64e58f54d550808c4d6d87e9681919456c4e7956`
- fixture hash: `sha256-7d10d338cf842d955d253c17c711f61a919941b05a2192e292201851e3214a2a`
- score hash: `sha256-8aae205ae0d510bf78ff7c474042bcb5e7f1d1cfcf6818619070091e68a80fad`
- bundle hash: `sha256-231073034aa6d66399a2d88dd529bbc5b57542bdb16d35c4b899d170048b3da8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc716d2fe029608c9fe52ecb8defed0c2de7ebf60cb8d8503f70a55a165b4d33 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1d7635a1bf4362e0477ca2cceeb24f2c5089335948649e5d24f68f80350a19b0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7705b53cd25d913ef5f2d808b7332591c7fa683f87a27ec81706d7579ae19f34 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-cd628024e299dfd1ae0486048adc4c73f130f9f7d75146c8f199e0e700cda9fe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f3121764 | sha256-ba269224a94c0095439bd679bfed810416f71e85f36fbc3e13570bcfd6e77ffe |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f3121764 | sha256-c9a2861136dcc84aaa288ca3711d14a4ccc72b18f16c835365f8e280420f550c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5c67aa23 | sha256-75ce5496d2285ec7e8e375c1243edfacd199372a0d66a3b737a250ac82e1842d |
