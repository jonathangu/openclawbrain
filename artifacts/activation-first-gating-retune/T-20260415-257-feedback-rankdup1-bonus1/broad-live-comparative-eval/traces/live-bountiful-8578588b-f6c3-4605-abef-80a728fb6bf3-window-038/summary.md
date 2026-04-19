# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2efa642ded94688f4afd0eecf5358476119ded4a7cc615d3107a42bb9da56bc6`
- fixture hash: `sha256-8ea6f9e0676b2eee5f6719ce0cfff479ea74f9e0b50aafd2d0b799110ba4611f`
- score hash: `sha256-f5a6b96ac0898422c8bfd16bd8857010e7b0bcf788e3e71ebf253245c388768d`
- bundle hash: `sha256-faeac268be16d5bf8c6d5cfa28d8c7fe4c863bda5b6a095a38225f9c4481fa76`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31edefd25aff21f4c99c143ff7e36055bdfeb17ee0aa654ab5907f063e5a4c87 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-17aa652fec7f9b1a75a7a9166ffb52d7d9c66b738fb2fa4435c12cdcefa5b13d |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-5dbe4cc7caf2da533b7578ff4921e5b22b11c40a7d4b6787105ff6185ebd12eb |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-715d3ebb7b5d34b61d3403699b39a46e0f74733c0c7076188b4b6f62ef8bacfc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-74067fce | sha256-9c4ba46e0360f1bce95b995734fb262c73e411b9698229f62f384d64d59bad2c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-74067fce | sha256-82b97852e48031b96f5b4069d4c80b24002aad8429a2783c18c7aa7484637810 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-74067fce | sha256-9c4ba46e0360f1bce95b995734fb262c73e411b9698229f62f384d64d59bad2c |
