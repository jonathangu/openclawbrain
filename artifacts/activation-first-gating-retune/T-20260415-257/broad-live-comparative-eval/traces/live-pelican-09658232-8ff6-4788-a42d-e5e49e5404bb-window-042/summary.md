# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8470650c25739a12e09f620484c19ed76e535bf12ac60fa5bc19c4d9e71da263`
- fixture hash: `sha256-1a290f6c39ed84b2ca073e21a57823e82667ab7d1408676870645010e286d76d`
- score hash: `sha256-cc9576319d4caf8ccf5d6da28bf9e9506f52b14e2e0d45f843478e400872b1df`
- bundle hash: `sha256-61cddb9af54c69bbc3367284e73ad69ba9df50b8bf853c5488d2ca92ba7fcd2a`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3a6ab913dba1980e898112aa64be6937d0b6ed21b1816a8e5ec6996bd2ca05f7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57db2a40ef0777b55e5264c1c6415eceac8b145b00435afd98e13b575d7055dd |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-942703cf884a5ef5f13e10295528d048005d71022fd68698eac66b175c5bdded |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83c4b2e4 | sha256-4ebed4df3910ef100b5ca5ead8c92cb35f0b669f4c26dd8c4f14455f999c333c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83c4b2e4 | sha256-89dd64207489196d738382539ed6eb898896baadc1b463ff5bd8c4a5fb37d986 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4e52a131 | sha256-29b110f47e0ef254ceb3151d5b555e5d221e8e8c7a05adc288b2363b6c095e38 |
