# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176`
- winner mode: `graph_prior_only`
- trace hash: `sha256-480f373d763bedd2f5766cb9a1a8860701112223bd910911c4830c2fc4277912`
- fixture hash: `sha256-2055d04a6856d7cf43d112e858be3651b8402ee14faf73331e6f59144245384e`
- score hash: `sha256-5ac90a394fc5d7fedde4c7c2df31184b6615a844c568c6ca49ed88d59d89ffe4`
- bundle hash: `sha256-015abb8b2dbfecef907d8431a2a6ee355083e0e00adbf53cb7684b589cd1942e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cc6a8a76ddb7a25937feec38e19ee175087db4867c24970d2759b39f1c9b4bd1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ea78989b78ca1f579aa559dbc246a69dfc26b9f930c51f2949c6391303612ef1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd36ab8ef6af77eb2a351cad47d344872ce7e939a44fa09964ae95e4728509e3 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1c7e847ab9b227910d6fcfe98bf7875c02f4fecfd686285aff4003979bcb6636 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0f836eab | sha256-cb3fea8634f024facc1409105cc10113f2f509684942ea7ea24a9377a34cdea7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0f836eab | sha256-2513de0fdf8014c41e626a55c1aad9b6151c41c7537d53a4222828c38decdbeb |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-f418bd76 | sha256-459b671ed51dab34143fd3bea5245a6f57d477c6acf4edc00c2c31b162b344d9 |
