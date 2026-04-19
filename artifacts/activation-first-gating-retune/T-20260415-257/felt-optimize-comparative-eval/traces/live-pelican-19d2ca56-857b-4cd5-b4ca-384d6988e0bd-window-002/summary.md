# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3587f37a965ca48e3b14fe490f41619f4a64d9248201fd791da49328673f2fd`
- fixture hash: `sha256-72b30c69deb757e882e827610fa9efbae23b0f6f41cd081abb4ab731c8f4dc73`
- score hash: `sha256-d6d2b1fcddc260fafbe52f6229f87c1c446505b8d835d3aa83afbb5ab7eb7100`
- bundle hash: `sha256-cd665d28d64568808056b4a4e97bd1c5e35a9e7f0423c20f5fba13d93bf82545`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9cbe4dccbf152c20742a7a0d9f6d7f345aa7c7916722159d8bcf4f7a084bf5a1 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-993827b8c785bec1db02ce338b69afd067b9ca4cba5895c38b10867c3f7efc40 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-b62eb15d152c9af4c0d71ab8e25f6ac4b38bb9ffe638a229eea6c053a0210305 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6ca15eb43a09e1c602207c64e1a318c8d7f192da6c8e61156ab76cf5ed951095 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ef2aacfa | sha256-955c3a84350dc6b123b805e47b390cd01dc1a22ae4a64b21032ce56e4d7069a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ef2aacfa | sha256-b175d7987b03874508856d6a634a3d2e979a80aafd502f5a33361b767a6d3cc6 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ef2aacfa | sha256-a3a1837425ba9999fefdebfb441d8f66bc28dae7f9285fc1f917083e3017eb44 |
