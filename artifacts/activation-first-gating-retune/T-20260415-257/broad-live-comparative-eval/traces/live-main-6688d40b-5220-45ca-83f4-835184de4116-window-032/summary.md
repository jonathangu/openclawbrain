# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92a53a83b75391e6ea2e19694e75cc46987c1fd7f2482c72c3850eb3ee758d5b`
- fixture hash: `sha256-a7a70c06edd57e7fef42061ce44261270b10f99213ced50cea189f13c03e8e7a`
- score hash: `sha256-d6bebbabc93671dd283c9044e70ece121bf38d967a3fb8213d1df62c5da2c616`
- bundle hash: `sha256-9f7cc5313a17ece55be4985f21e6068f7705cd9dd1c259a241ea15b03dcc1880`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-467d90a6c748c6c78cf3c7ceb933156139020979bf5f7ad7e3a8103479da429a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b3a73804eb586096236f322688f1b7d3f63e12e9c8e107b0459cf946b3025dff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f2fcb29a8b9a2c21f541c38cbcd5691f41ab38f22ecb2214850785f20271d3d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5305e2333519bb224402a61298d49f0c4d807884efefb9943eea73ee6f2a97d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4dc67db | sha256-e61cb355a3556a5fcad2ea977029dd8c2235941d9455322544132442ae8959ca |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4dc67db | sha256-b5c5764ba97c2d9fcde610e6ce0c0d0f2c0650f0f23c1118d1c580591d9faf80 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f2407bf0 | sha256-878b5667ab118f4efc9939c273ab771fe5e6052577135c4d0ff6ae16f50923ed |
