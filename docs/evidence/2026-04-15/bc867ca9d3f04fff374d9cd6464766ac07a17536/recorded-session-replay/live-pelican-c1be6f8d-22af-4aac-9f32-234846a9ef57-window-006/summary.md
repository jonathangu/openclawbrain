# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60561f5cd4b9679d1d07ec70fb93c8ce09ef36cd5a40b0352b67931141e9e246`
- fixture hash: `sha256-d68e77ff5a53346b0fae859928eb6131851ab9f7d88f52a94509c0f85b109391`
- score hash: `sha256-6eab78edb94ca18baaa492d59df880dc96e13e31ba8212c63adf9091f7e9c112`
- bundle hash: `sha256-bdbec37609ec8d5a5fdef0cce38385b6bb9c43f131d1e0067cdc04502e5ab89d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-350a961987884e875cb36c29ae1cb810ef961abe38158c92bab3e2c95369cbcc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-02dd7e1016c0bd4d18313e478435167ea6355f104632271a0ca9de08656a68bf |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca32cc21ff0ad84aefb9a15215354df6e98ff786a061bba708ca42f1c07533b1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-17eb416c66b29aed1b9615db8ce0e35bbca0beb78b6113adc4f63a7cc330adce |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ac3efcc | sha256-f903d5f37037afa896ebfdd1f3d2307924fa9804c3fd9c456200b2291fdb2d7c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ac3efcc | sha256-ee28515c5926df12df639ded0447f31450a00b1eb029aaebca694598a25ccf19 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2ac3efcc | sha256-f903d5f37037afa896ebfdd1f3d2307924fa9804c3fd9c456200b2291fdb2d7c |
