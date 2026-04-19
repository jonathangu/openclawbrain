# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-816a27f51a89ee3ab61ecb9dcf7fe22803e339036dac94e7aa31864fd0968283`
- fixture hash: `sha256-c096ed79c46bcc788a54db7b73d6166a0d80aa4dd8f8479c075722de69b2b170`
- score hash: `sha256-556287d12b2151b44e26369d325880ba8e8dd5c409e2ae9919c0dbc17e86fbc9`
- bundle hash: `sha256-2d8c26cdb6d739e57c206271bf83c4ffb104c98105c1f45f66f5a6ae1ea23a92`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b145c71aa4e121e88e077c9e9aba7ef5e72b3964bc6945ed620719f5b1c0299d |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f1d3c78d3417dd24ebff18dc75bb9fae1c8d601c04cd9ee7c719c171f043896f |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-d4053fc2b2b34c136ec8d69d974bb4a01f090370dda7ce7f3b20bad49bf698c0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d7c92291071c71a6cb8d0308a6b5f9edaed6462b39637f186ad2afe09f609378 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-97a8457a | sha256-4ec05e4f229854cd5d184ca38fa7f1d664a420384414056c047392f952f39595 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-97a8457a | sha256-9338478c5cf163c226294007f37156b6f1a94d680f3c8b7242ac83072cd081fc |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-97a8457a | sha256-4ec05e4f229854cd5d184ca38fa7f1d664a420384414056c047392f952f39595 |
