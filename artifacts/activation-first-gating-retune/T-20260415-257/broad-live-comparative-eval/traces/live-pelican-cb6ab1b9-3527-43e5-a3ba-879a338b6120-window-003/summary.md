# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8226e38f2d583af41a4327f3b8df4e5b434ae18ebbdb89d67531a4a854359a44`
- fixture hash: `sha256-e3733e9aa09beb01fe43936408b2069d985913ff1742752483045d9debec0829`
- score hash: `sha256-43c3558bce7a2f33d7daa1f6f45208f0643516f2fbd7ba4452584df5c674cdd7`
- bundle hash: `sha256-15ae3da3c46870552ff324da65dda0528cd6d503b0be4da220596f816e095b35`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f1f95cc8e218fff5d5905cf899fc04d3d3c62a98c1d684ae5ae4dffaa6f7bd10 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0d4cd921aaf9d745692aa47fd9978a4d78ffb2ccb902c61afe873b8c16b4b479 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eb1b39b23f9a7519ce0c40b03741427387e0237921d1023e6ab8369bff554fe5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-15d3dac8b98ea9c1c6bb23c483bbfa038a758fbaabadd09efe83199ffffb3282 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f337262f | sha256-0c48060e7a5ae5576faefee98e168b85c4ef3b2cdf242a62f8795b3f4e4a8188 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f337262f | sha256-59f6bf2a26afa9e67df924a0a16fb2ad74ccacd04e64964fabf930500df968a1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9c69773c | sha256-7950c3191b1426dd57cf4c67007d099258a872dfa48684b54f1d5ef77a7695e0 |
