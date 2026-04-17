# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f896b3d7889710642e066f81d9ef38f09a0375e7c4550a3de44bd42b8be0c728`
- fixture hash: `sha256-02b10f8d55f27089a7a2cdde95f78ea9472dddb1b95943a1431bda089a73cd5e`
- score hash: `sha256-3c4e4f73413eccf2e04a5766c37dd792b4b594ba7275356a78273efccb0e5fab`
- bundle hash: `sha256-5a7e3f615192dd1d89ce983f4a3130bf47149c9041a01ebbeed509b8af99826a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6e466bcaf85a4528dcaf1f22f57a3cde69a22135dbdc628862617cea9e4f77f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45162f3565520fcba8051f1b9ffdc2ad49a06a00eb7752933519b09301f29b6f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd11cc65749405e52e63eef43858ee72662a41086ae95602c6d7fce1de71f61d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f4230d76ce863842c027b391261962aff3fac7d22c63a71099e024e57ff5d40f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-804b31ca | sha256-58e6abc0eab8c8accf95fff3ac72d48d9c45c6a8de1a1e9570ad126a0e7095d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-804b31ca | sha256-2d8baa7e49f080bf5734d97c606d2865ff1e22801ef9e0ffbea3f402465ae654 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d531dfa5 | sha256-a9b078c2bcedee4208899e40dde6c7c5260662564db257994b2c1107e9096308 |
