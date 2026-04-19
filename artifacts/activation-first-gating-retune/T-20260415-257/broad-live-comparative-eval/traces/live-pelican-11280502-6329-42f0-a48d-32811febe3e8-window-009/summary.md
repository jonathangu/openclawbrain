# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f6c0119a771d3d4002ad2796a568648f1ef4a576c6d777928a579e3ee482d2c`
- fixture hash: `sha256-bc5d36362845b850e526b2c0c66165097088057015558a1287a8837f47ec0645`
- score hash: `sha256-055ef8ed0835bcc57482ac6fab30da8d0a498d080df9d1c8ed79f27b638b0f1a`
- bundle hash: `sha256-8f4dc4823172b7a6c6559c3535501cdaff019fb4cc2d839f5d8e69f82fe1f457`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23fd42a717917abc9a657ded0642e878a093677ccd38ec86a1e6fb99c57037fe |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-b861e3c9c9357e372c8212e337ff34823f2c965854087a9eac536e6fc0750c86 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d9086a128d1c333cc2420bd0ad58d3238df10a71e77641ce351450694bb6e133 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1ff181ecdcea021cb5666087100f760cc21fd784331c5e8442e331b12d113669 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-ca4e78bb | sha256-9bf5500b244901b6bb48a2a19dc8fbe7386ccdf62586409c2c5998faf37b672e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-ca4e78bb | sha256-49ab82fa521cc58165ea9e59d9422b7e3ee854439f63433bdc4d8dfab4d2d79c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ca4e78bb | sha256-9bf5500b244901b6bb48a2a19dc8fbe7386ccdf62586409c2c5998faf37b672e |
