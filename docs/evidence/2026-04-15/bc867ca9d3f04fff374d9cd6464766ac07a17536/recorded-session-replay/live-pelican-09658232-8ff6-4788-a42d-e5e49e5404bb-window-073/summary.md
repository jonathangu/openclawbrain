# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4344938604e067860a8ae5cde1fca1ccd4f50c2742543e1ed5dbbab203e23d74`
- fixture hash: `sha256-835a3394dac9a8b8023e71ca801b0ad86f7853cc9e826a2ddbfdbf3c56dd351e`
- score hash: `sha256-22ecd2d98c44779dfeeb4332f179c86dfaf970c1ea7b31d105a2ef6199c466ce`
- bundle hash: `sha256-b7731ea0e9f3748db749001f122d7d12488162a93c6107dc95449e6472108ea0`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1127e96d3cc8efba8b0b495ce0293094b88613393e1f2f8034ace4e3e2126703 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-50fa0654fec4fc368bc23031e67eb097a3f981c40885b3742e80bb32b3e9b1d0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8ff1dddaf632209785e38dcb95f75b39760e624b05d875d413a1d8364a6f6380 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e9a4ad0 | sha256-784d265d05a64d60f5723f8f4ac155752d6821ffb3e11646f21c4eb4543d81e2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e9a4ad0 | sha256-0a896c28a39fe2ebea0c825ecd6af5547975dd3fbee8d6593adaad768b4bc017 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8e9a4ad0 | sha256-784d265d05a64d60f5723f8f4ac155752d6821ffb3e11646f21c4eb4543d81e2 |
