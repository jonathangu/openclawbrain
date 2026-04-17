# Recorded Session Replay Proof Bundle

- trace id: `live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-93228e668a08c975492dc6af4e3bb4c71052274e3e003bc535d1e798cb5b7551`
- fixture hash: `sha256-de7894b208900137452009cdc652956a77d2f2658869966be1c1f8a47a12873b`
- score hash: `sha256-dc325454259729ab59d8ddf5d818ac8cd3177d6a84f388584d8a0f9422685c80`
- bundle hash: `sha256-a534b176d303088eb1653105ee0066474e0468ee3d295913ffed720b66128e9a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c56ee3ae997453b8eb93280de0f46e35ef0156aa279e2ba51ceb2f8a8bfd749a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dfe7f735a977a026c8b490ded557605899f2e9d168611c231227a84651f904a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-912f62d082a3bdb615e262b9721ee5d1e6e6af9dc7438cdb05bd3ed98f27198f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-13dce712e2fc80ed76b8a8e600d6c68f43fdd19acc8caef4abc28cec4b8aee12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b5313065 | sha256-17ede72e6aadadcd64d8887e299a157a44c893ca040f4de29607cdf25332a29f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b5313065 | sha256-b9b302572d4699ba69a3c5ab99a0874b84b318b69aa8d1cced10851da2b98227 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-af27059e | sha256-84c12d8fa6192b971ceaf8e5c686778f47aae88cf5e9355aa923fcefdabe9d44 |
