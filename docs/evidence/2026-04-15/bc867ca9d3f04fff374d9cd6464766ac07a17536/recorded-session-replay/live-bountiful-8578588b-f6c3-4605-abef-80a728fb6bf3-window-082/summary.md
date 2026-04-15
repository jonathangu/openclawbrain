# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1d2242059d4efafadb64c6610106d98feb5c5f961bb4e1b42df971f4c261a60`
- fixture hash: `sha256-0fc9b302b2dcdfe1c12ce6973a204a93e663e9ebf3a3fa850cdd1e41f05e02a3`
- score hash: `sha256-17dbf9a407cd1b45fad6129a13ea23273205b88de285604fe089b8c0be4b4e00`
- bundle hash: `sha256-82a753e36965e45e6ea7b503a0859d0c96f397b0386e240d432d2c349fd743d1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1459c2fccdd36ae63455329e95c444cf7e45a5cf69fb7b55a421593b88bbe48 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-67cb46c2676ed32c3cf0ee0ee357765ca050cd2b2a47993dc0d121132084ded3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-de4c21db125d9a72833f148092f85b159edeef1292b75adf5e7855633bb13690 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-928e3aeda917035bcbcae2da3947fcaeb427d5ab2c65e90039c83435480dd736 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5b9aa08c | sha256-3f8ca63e5a333d55b3a0c01e220331d3a3cac4902a85488ec333d0409e007bd6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5b9aa08c | sha256-e0c689a76c83ab3f43197c53d8909019d14abc86b370a757262ae17caba19fbf |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5b9aa08c | sha256-3f8ca63e5a333d55b3a0c01e220331d3a3cac4902a85488ec333d0409e007bd6 |
