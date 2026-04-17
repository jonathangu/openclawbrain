# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68581f69a97780aac278954522193e99993d4befdc39acceb8ff881974cc0178`
- fixture hash: `sha256-d2931cc864933b7e6af27eb1382872e22dbe9358020b6cefacd8fc78d2489792`
- score hash: `sha256-98fbebae018b850ad94367a8a96a4e608f1072307b23bc8326d0bf14dcf61515`
- bundle hash: `sha256-2cdb06a4eef8ae5dee9bdb9a2a13761e41d0d49da42805c3b1eebc4280c5c841`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181208f7b843fa2c39286593bf1b96c7f44d97e1cb317cd9b55efb3be3bcccb4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-72cfac29876ae3ed28b71be1a73f275c35f33386553823531cc2a2e5ccbfd917 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8329921e5a0e117ce1c19605a2e9df513e508701bb8c0d83f145ac5165004e6a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3549b49cb835c75d62899bd58ec5466990db8779b2c0b4f4885c0fa1e8c584ac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70f4c32f | sha256-725a128f80019e5cfc583109d118c33bb650939865f6d478b6a47b7f3f7be203 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70f4c32f | sha256-5b3813674981afadc6eb67f5dc441551582a650f65f48b4db9e31371976952d7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b58b5c90 | sha256-f24b1c81c9d6393fd12d90f693373bc7d705dc43212d7ad4cb9a671924d0ff7b |
