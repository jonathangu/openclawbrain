# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13cb1bca8722ed39c54b48c9d170af84a0229da5a1be3326ad569cdcb6c86e93`
- fixture hash: `sha256-3649ce5ca20580b372f2a2005a8164ef24eb19856bac1831bacfdfc2aeeebd5b`
- score hash: `sha256-9234f2cc72e3e44eaa7b88bfbcd83bc6e32707c5c2342eb0a8b2044c969be458`
- bundle hash: `sha256-b00eb6ba19a51ad1b74d3849316ebbbeb0a4e62722a4173a742aa683a338e2b0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fcc253b9510f29399fe22001359326c4d47b1fc87658fed51c53d2aa08bb9eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-656b229c01e3614168bef845ef99a79ac839a5bb5d38711c8796da877212ad22 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97d202805d255b5b28fdec8d029717d4b4675b89370c83b4ce75ee52e1754b13 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fa8fad5d5a5a49e36e1d56ca6d2a7c1347fa974b1d7a8fdceaa5cbb6ec8eda4f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f219db4a | sha256-d8f9a586408a55145a3d9178c11ce381a0f10ee4f0eb1189791fa3a55594d70e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f219db4a | sha256-83cde2fde508901ad8bc0c2d762db7ca22d2288c1d2b697e30fe6cd844d489bd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-df1ada77 | sha256-d840cb2ff2a952633f361aef622aa3f9a9ce649df62924a31ecb9bd2137c7937 |
