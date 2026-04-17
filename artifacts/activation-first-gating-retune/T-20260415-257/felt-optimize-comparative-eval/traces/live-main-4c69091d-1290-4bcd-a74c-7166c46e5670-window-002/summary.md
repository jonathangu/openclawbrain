# Recorded Session Replay Proof Bundle

- trace id: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-44fbcb6576f6f004911009e236161c3fb072626b9bf71fadcefa3c9dfc1347dc`
- fixture hash: `sha256-178059882cfa4f40ce27919272b11654587c109af203796638882d20de0899c6`
- score hash: `sha256-9135c101a85f069872f80dae7143b239130b0329562dde14004fec01c2c97d93`
- bundle hash: `sha256-e828d5e91edddb64508018bcc61c38479618031121e9e35bbfd41ce17e8632f1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc575aa7f7247c73c93f72af53d1bbeba87c049e551196e9ed2534df2a742d2 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a345e07da113cb0a2c2933494a7ae610665a61b04fc1ee88d5e229dc5724d5a5 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e90d0ed843db9e22612db567beacfec5a26da186d4c155eebcc799afe5a1ae0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f559c02dec3369fb91a533ba10b50aed63e7ce86ece65f3653cfa882998c607c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-151edfb1 | sha256-34456322552a5c7612337d9c2f5476ad1dc8b89b7ee59f951ac3e410dcffbc56 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-151edfb1 | sha256-6a2ad74c158ad3ae760b7bbc064e7b7840d5218c2d874bc1a49a7e33938cdab8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0ac89658 | sha256-d4b1182d73d2bcd7212713caf792144912dc592adf0bcd26188bc5c748d34332 |
