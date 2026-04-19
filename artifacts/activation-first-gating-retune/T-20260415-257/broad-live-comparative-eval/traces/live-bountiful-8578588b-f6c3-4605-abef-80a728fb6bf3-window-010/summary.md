# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5176937ed6fc9464e75096c1c5b5dd165c66db577451f32573da064aacbbd206`
- fixture hash: `sha256-c4ac3ff5736ddcdc283a0e74e44fb3cc2c8ef5acca6ffc633b4bdf3e072f174a`
- score hash: `sha256-6cda9367d323aab475ec20e2cade267509613ca9d4ba1cf648ac663a906a34e0`
- bundle hash: `sha256-e299c13ff4e0004b0ea0b2b8e83f26b0c3338b7dbe8f49b21e8a122b21461bc6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-303b09693534000a07eecf53da8c48518fa490f57e0684dd58a4834c42c9644b |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f465953201ea976bffc8385806ad5869fe0d4779b351d653813ac6267d284143 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-b33e5d7d081779da767357382e254a85cb58a5ade1f82617885ce5fa1401e03e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b26eff59ffac969877267fd467300fc5a69a796dcd0c62485eaacc1934db2f80 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-0d0cf381 | sha256-b0985c53b7778001905c056f0d9d9efc6b4516440af8dac786be4f5ecc536e1a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-0d0cf381 | sha256-9c5294c617a942f2cc2867c418559380cee3cd2fb9398da5b350e62eff42ca3e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0d0cf381 | sha256-b0985c53b7778001905c056f0d9d9efc6b4516440af8dac786be4f5ecc536e1a |
