# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149`
- winner mode: `graph_prior_only`
- trace hash: `sha256-299806353ab465c5dc0556cb46d4c0ddab82caef7c74016e1f229b80f14988f5`
- fixture hash: `sha256-6a01dd18700c95a1ef47fa69bd96f40af05494c628d07e9816bb8fa24129ae15`
- score hash: `sha256-32e68e56cd66378a3386688105a137f1f28026b3b5fee13e1383f2214f942de0`
- bundle hash: `sha256-371ea0c903dec73328dcaf80f48ff6c9408446bb5115c28b97f3fbb09d99b540`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2b5dc7e86f33e7f35222bc8995c5714891af6e48c0b188589cfd85f30ab7cd |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58a662f86ddb606cfe20c87e31ca4bd2c6b3e4b6fe26964aef153de195c4a544 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2acf1763487c83648d41e659ceca1212072fc504a65cf143226cfecff6103077 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f6027779834518f46c497c47e96df36142cde3f49a806123cebb185c8f9ff1df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9eff7898 | sha256-31e328c9566e0bf216694aef6ded19e0cb35b8cd7a04b4df7b1c5a11dac4b2f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9eff7898 | sha256-a1a0f07e99a72c37a9f30687db68f78d4112b243a9ef596848f3aff54c61b15b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-44dc7805 | sha256-c2907610e2f9d95ca891a29e3c20929a0a8325ba51a93c95cdebd625344cd86a |
