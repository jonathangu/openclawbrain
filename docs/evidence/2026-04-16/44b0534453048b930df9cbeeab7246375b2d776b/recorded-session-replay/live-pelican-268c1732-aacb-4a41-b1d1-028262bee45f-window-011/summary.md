# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c761e47d47331bf575b7d002c131402195e8ac49d688ce355a015b14825d3acc`
- fixture hash: `sha256-128e53eb7404fc5c5e08cc33f7657166db8766e76b0fe254b4c32e80c9220dde`
- score hash: `sha256-48b653a75b374c41e454fa08799e9bc36b1b40aac4fccffb88b8f896b508ba20`
- bundle hash: `sha256-38130db941f0f4f5cd845c730ce26d83499141425a82fd499434c0590b0f8acc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1631644a30db1cf3d02f7c72d9e973f8085c5ed6318e74e1e83701e3e901455 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8b44f84db209417db7d00e1fb46e09e6c72a00faa28f08c4ab163abb05b2b320 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-abb18be1066d104da53eade1931842ccc0e9ad6f030ae2b40a42f5d4e393dcdc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-cbf9e9c209e19103a31eef2f0de6ce6a1fe8ee35dec4f85c5415ce41cd24f283 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b22befd5 | sha256-95df8cd31a6e4d7719f4c69bc838044c9451ec2db7f9476524e4f4a3c27ffd1c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b22befd5 | sha256-394123c308a2b870707a5ff48057c0323ea6dbc610ba0ccc0ae2104c1f22d87f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-50f328c8 | sha256-b1ce6102c6f103116bb9a061605e7f6bd3cebb262abfc7f3cb218a0b494d00da |
