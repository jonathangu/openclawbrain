# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31f15910a7f37f6942dfc1fa59eebabc12b733c2fdbc101bb92672de7f721f0c`
- fixture hash: `sha256-d3d3b7c9daea7f5dceb8bcbc7d0b182082662e4eea5368602c8cfc65a5234e7a`
- score hash: `sha256-10a50dbbf0270ebf965e70bd823f7d1e73c598b4ec7847077bd8310ad9a61149`
- bundle hash: `sha256-f5b16f76fe8f3a0572023051036cb3e24854e06ff385548ac684b04d41ebdc60`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eee7751ada66c814393120538ba88242a0ad04eb627a4b24f36524aa1be2a704 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9371daddfaa407093b2b20733d9c3579a458ccd91780befc60d8a1401115b4b2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-85577d80b3fe8c94b532c07755b43c83c5ed9ea5ffaccb1800054114bfb71825 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f9cbec6b70917e2b0ddb15cea4a55253b7d6c713bacfcb93c73c216ccb890296 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40e48b2c | sha256-7849bb41c9917cf4c2259b92ab0daec050ca9f392dac48beedfbf956d91d55cc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-40e48b2c | sha256-776de342f6c0c2722bf173b89fd9f59d42978cf18425fc029c6ca362616063ed |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-59cbf145 | sha256-d4228d7c5913671f7e8e277771b65ae0716025fc52c88a800ed9deb15ffa45bf |
