# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92a53a83b75391e6ea2e19694e75cc46987c1fd7f2482c72c3850eb3ee758d5b`
- fixture hash: `sha256-a7a70c06edd57e7fef42061ce44261270b10f99213ced50cea189f13c03e8e7a`
- score hash: `sha256-bccc7eb28901dacdd16e69fa14a5f42bb9d37ddab57659df7c708b3dde6ac5bd`
- bundle hash: `sha256-23e8aa3a4297ab687b5ebc534eca7c6bad7538fd6407e701ea8e16248c785766`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-467d90a6c748c6c78cf3c7ceb933156139020979bf5f7ad7e3a8103479da429a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8057a4304ed090f913a814f17ec115f248d7b2b105d386790792b5e34086b42a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f342eec21975fc39c8c2891237dda091122fb166d503d59b290b4e87bb4fe3df |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e77d3ea266c52c57aa708057ad551e682495e2db74ee27f9b77fc29c245d0e17 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86d1f577 | sha256-eb5a58f029ab397d74280fc20503417b46c652b781cc40d3c97938374cc71666 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86d1f577 | sha256-619eee77b08235bffc0a40814b3c8350c645a8ba822a24068ea6be474be16b40 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a436098c | sha256-d0a6c02e7586de94242cac0a64bdbb51cc1f366b1d058554a774e587c6e2a60e |
