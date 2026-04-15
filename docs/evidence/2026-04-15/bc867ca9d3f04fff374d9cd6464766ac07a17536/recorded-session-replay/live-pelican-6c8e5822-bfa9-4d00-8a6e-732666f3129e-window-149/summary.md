# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149`
- winner mode: `graph_prior_only`
- trace hash: `sha256-299806353ab465c5dc0556cb46d4c0ddab82caef7c74016e1f229b80f14988f5`
- fixture hash: `sha256-6a01dd18700c95a1ef47fa69bd96f40af05494c628d07e9816bb8fa24129ae15`
- score hash: `sha256-bdb2dec9c72d9680a30f0ea37ae11a5e69c7ebc59572cce2e385d9f5dceaa5b8`
- bundle hash: `sha256-c1f692bf0335ef60d92c6ca4b1a88ae6279f3a957c40b861032e4b68295c88ca`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2b5dc7e86f33e7f35222bc8995c5714891af6e48c0b188589cfd85f30ab7cd |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-726131ec94323d55a285ca387d74af30dfcb8306cd5671336d2a26b23515af6e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d09b7a3a0ad38a20a8fa62f50f64a215c94c4435ed81966d1ec3cdd727162ac0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5c509f120faa96efed8db9f534d5ec71973d3c3507a02a326802bea70a691a73 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7ae28608 | sha256-5042350904b52a702e5f42139c3219d0f97cda381ecaef44749de34b198e0739 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7ae28608 | sha256-b34832bf6d76ee35c4d940ff4f9aabf0cbc677b6c1420b583dfdcb6efa4f5e73 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7ae28608 | sha256-5042350904b52a702e5f42139c3219d0f97cda381ecaef44749de34b198e0739 |
