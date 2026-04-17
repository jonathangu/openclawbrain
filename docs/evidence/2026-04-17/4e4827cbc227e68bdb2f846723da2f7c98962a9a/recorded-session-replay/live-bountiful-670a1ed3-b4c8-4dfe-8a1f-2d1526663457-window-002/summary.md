# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df424694932b0793aaedff791f54d5ac971c24ed551452ee216f10c505396c8d`
- fixture hash: `sha256-cdd5cd85fb616c8f44b236f115a79978bc2dcad4597a177039207ba517f1bddf`
- score hash: `sha256-57eb38169c93ebb67be43cfc6fa34ba0d866b25d9ea33415f228aa045e54ccc6`
- bundle hash: `sha256-057354448f25e8ea9912120589fe1220afbd0eba97e2737070b6ab811df4ed4e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df3745ac4e10090248775f0174e4f7f9517bcadad1b8588a0276c1d2f867a57c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-701f40da5ff8a90ca59b6d95fd59c01cbe843ee9f0782109a2df5b7179b9a6ae |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1df16950d82efa8be77e931bbd3d4cc4a728a3c6b4e40e21ab6791a1d41f751 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5eab565d3de696556fe8250c9e29434caf7e3b912edbe21e6a6020406232a757 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-deecac4f | sha256-ed870031747f684b6b77d714d39ddab39feed1cc4bef91fcc21c3949860ccfd7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-deecac4f | sha256-ed870031747f684b6b77d714d39ddab39feed1cc4bef91fcc21c3949860ccfd7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ea1b9c04 | sha256-6f7145942b4abacb5a8502c039f5a35421e075786c54c140d56c51f53a11414c |
