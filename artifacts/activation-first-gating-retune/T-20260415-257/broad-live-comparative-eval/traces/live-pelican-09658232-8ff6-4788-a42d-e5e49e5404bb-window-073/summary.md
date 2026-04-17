# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4344938604e067860a8ae5cde1fca1ccd4f50c2742543e1ed5dbbab203e23d74`
- fixture hash: `sha256-835a3394dac9a8b8023e71ca801b0ad86f7853cc9e826a2ddbfdbf3c56dd351e`
- score hash: `sha256-7717fcc322d3fb3f7e568b70417724a17522bc91c054179df853b4e92b5da16e`
- bundle hash: `sha256-353261f72332d1400f084e3ebf63b35ebe44b264d15bf813546bd5f512e25453`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-95f6b633b00bd779574dbd24baa772f0fb4eebc8350ac2c13ddc54230525a7fa |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4d01f9ad004b1a732391e3e0057ed7e6684c290611ce04fd730eff9eb5365fd6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-61546099146eb24350dfa8406d00a5080efa6129ff43bc29f84684f345f5146c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-58d01e2deb87a244f6c7379b66cd180449c49cc22db35104803fa37b5475a48c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a84aefdb | sha256-8cd74a1db2b20e5012ae8b1823c862b283843c3871540fc25b843d519feeb145 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a84aefdb | sha256-01b5599e74ce133840bb236a68a598d65cdf5c79428658a8c00c91cfb062064c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f126eb9a | sha256-c501d2d3d46003e782ad5b17250d0d3412b51bfed59ef262eb20447644ce0110 |
