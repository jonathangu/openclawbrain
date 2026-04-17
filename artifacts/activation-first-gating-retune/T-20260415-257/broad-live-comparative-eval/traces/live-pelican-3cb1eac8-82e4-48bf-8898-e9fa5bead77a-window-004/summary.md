# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-79532f3b0ed4010e846f65618be48c2307af13b97dd581f294dd9d5e6325f8eb`
- fixture hash: `sha256-6cb2c0584a4478c43146057da013c5b788958c7763f4cd2e66653360656b5ed8`
- score hash: `sha256-2ceca027aff3ab2b3eacb6eb89bb32bbeb8ac4726778d6f5ff52423b574802f0`
- bundle hash: `sha256-0cc5cf8e954b0570d7ef5ecc8a9eb15414b815b1bdcbaabbc5ccc7716bd979da`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9133ee007dc762137868e0af1d2b1845ed239e3a386c6ccc9187f6b355e2ae22 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c21869922686a6a2f8013bfda0fa3ae8afd5ab858059a299541b482368cec8b3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58e8d7884218a586363bc254251847c0efa3220f3386e64f6fbbf7227c6fca12 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9f4824c4c03a2c2c5d9e262443352c9e69a1d80b194af3b686bc4bbf60116ff2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-363fb9fd | sha256-31fe021ff67a775c2f91cad14f6f3a83a3d08f279ae7d43dec5a371c4b10abd2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-363fb9fd | sha256-e2cb1fe59afdb3fbd7fc5e0227784d2430e29099efbb65ef4725d3817be3ef20 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5e0c699e | sha256-b810b988f12af67c86fb3cb1e5f8e2519ef67cb2cff40b11eab696b1c54a0c6b |
