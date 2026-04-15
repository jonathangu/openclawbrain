# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba20557b6de7502d32e3b83fc90ffd6f8ac19ac17c3a2b682f0895bf8bb69c7c`
- fixture hash: `sha256-024c24fcc3f69f4d62a086b795f3d8c9e3625b36454d26d1e235e1664a651060`
- score hash: `sha256-289241e05acec8f482dce39e91d11778fbad91b111745a91e16d6807388b9b06`
- bundle hash: `sha256-3d23400b4edc61d2a11f56ce84b4bdb6a85d164adea942d1ebd8d77df510c1fa`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91ace2bc1ee23370cade1ee9720612db55ace83c09692395cb273562a40c2beb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-be09cfea75d4ad7fb7458e33da4f8251989986c58111812052ab02d1d16f0cef |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e78abcc85ef9456f49a889a2c0872f235dfed45d9cb5f268ac59bd64e1e798f9 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-33ae55ea753a6fb5beeac867fe189b018b6ece8dc02ef25b3dc8bd34a8d0beaa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-270be382 | sha256-f7d0e13fe51b0bfa1a3a1b49e424266d234d05e99063225c3d5166b1de1a4a58 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-270be382 | sha256-66af9707def9ae6c6b061f4ba6ff999fbeb525b0bd5f9673fbe31b34438e7c30 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-270be382 | sha256-f7d0e13fe51b0bfa1a3a1b49e424266d234d05e99063225c3d5166b1de1a4a58 |
