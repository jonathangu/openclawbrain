# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8470650c25739a12e09f620484c19ed76e535bf12ac60fa5bc19c4d9e71da263`
- fixture hash: `sha256-1a290f6c39ed84b2ca073e21a57823e82667ab7d1408676870645010e286d76d`
- score hash: `sha256-69a90e650032cc210290611209c13c308638be1a939013ecd044d86d13c235a0`
- bundle hash: `sha256-9d47b4906a08265b3c2e1d62d77c30d07b4cfdb534d302525288adc08b83fbea`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-575c5da35d27014ccbfd8fb043d25f84e1287b134d1db92502cb2f005c370afe |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3e602b8a1fcc9f0c4727d625d90a995bde699ee31662210d2d44f74c9655ccc2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-daf9c54540189180a7654b63371902a4e74e0b3b2c48f8f5735d7a1f726bd81f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6949e620ca66df80b7acc6e39ee9ae1628a6ef661bbdd3207fc00b519413e6e7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e322152f | sha256-0f05b824aae2df0827be29c4feb3f0d965c9f520a9487b9f146a4c540150f549 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e322152f | sha256-ffcab39662f7c65c7a78a205565e8fe6b1c0714345ca2ce3e2f700057b338c60 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-adb0037c | sha256-1ef247320158bac59784af5a1dde2fee95cd900aa2a9e0558483a0b52c1e3fac |
