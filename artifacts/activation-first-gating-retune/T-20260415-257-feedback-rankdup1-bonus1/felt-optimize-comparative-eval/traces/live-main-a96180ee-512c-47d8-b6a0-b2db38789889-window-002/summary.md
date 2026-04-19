# Recorded Session Replay Proof Bundle

- trace id: `live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-93228e668a08c975492dc6af4e3bb4c71052274e3e003bc535d1e798cb5b7551`
- fixture hash: `sha256-de7894b208900137452009cdc652956a77d2f2658869966be1c1f8a47a12873b`
- score hash: `sha256-8088faca4ae069397998376d4cf794b24a052e58871ceacd11dbf52842113b6d`
- bundle hash: `sha256-1d9043da7115b18c4acca6819d90a586558bf29f66fd3116e173a971403d8e63`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c56ee3ae997453b8eb93280de0f46e35ef0156aa279e2ba51ceb2f8a8bfd749a |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-1e46db9d33739f132b7bfebe8933b449ab5e218f151f6e071e04cfae8d9c83ad |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-50d89a5fa70cf501afd71cadade15a50008612d68b98ecc82128153c0af14385 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e38da0c7da11898f36abab533fb367b7ff12369a2e04f273b64ed8dc84fbc4b4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4d50fc4f | sha256-5c079ca0cf12cd9da656d96d7fabe6ce8c42ddb91e45e523b077e5897fd9cf30 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4d50fc4f | sha256-f92b82be9793140f5b8bd473f5e8c2947170bedabbd560940bee0e9b751e3694 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4d50fc4f | sha256-ab9df3c4e2cb1a8e229a3c91059ba3a76a649f8f654d371e74a8a07f88240662 |
