# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df541968ca52654e5efa48a1a6713bb4511f8366d389ef30e36174b0478a0f72`
- fixture hash: `sha256-10a1d9d424d59bf74d6edef2d25c3d9864b38e04e75b6ff4b28dfed92245cd1e`
- score hash: `sha256-e61846382672f86dbf7c62993eb99003d24e70e823fad7a6bde0c34dfa68589b`
- bundle hash: `sha256-9dac8498b2dcd6fc0879d74b02df582e93e126095c197a5912eba32bec3b512c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe456f8f9f99c14a6c26ae3cfe1240fa644752e272eded6c0df3fca37912d301 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b5e1b698ce29bd64685aa5258a638a36fd2ee585a82d42878e1224dcf5524961 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-763d6f2c77268ad36847e3c50863fc8ae9aa1204726ce22adec13653fed25ffe |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ce721ddac17248c7ca8437340b3e11d7f6d0b6fb84841b46c16ec2dc02c8407d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-164ba441 | sha256-e9e1273574fc864068105c49f140ef081b431331f1a40576090af23003f507cd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-164ba441 | sha256-c48bc205e186ff7012607fcc9ed9cd94ee2ceb491093d7a8c37c73dc5beeab07 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-164ba441 | sha256-e9e1273574fc864068105c49f140ef081b431331f1a40576090af23003f507cd |
