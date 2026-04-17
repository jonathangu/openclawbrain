# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-020e5fa0ec60c9180b8ca12d4a8cde03c3eaf93efdc6e1249456178218366170`
- fixture hash: `sha256-2fab851b07744bef46921e5dde6e3c44cc707f0e47e7a2b971ff5ea69c88de53`
- score hash: `sha256-44ed371aafc1d36ff0260cb6dc4fafbe4202073d44703846906543ff0818e0e2`
- bundle hash: `sha256-4dfde11fc749273301ef896c798312424b3b59a177140ed8fef6275965cab8a0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84e9ddc31e34697064a9e60de43374da82ef3d65551bc6676137ee0e90f5d63 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2ada4682fc944958f81e8a552f47ca3df6520320a73c32d3ba1b4253b6e7a4fd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-01272c5a49bc380bd706087edc47c0e58986065e56fce7e65ec17dfc7ea9d4b4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-13050129ab840393e53b43ee36d2f1bb0f4c8c12159e7131213538df584c2a34 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dcc9f294 | sha256-639034575e9bf887a712e4efee5b1725566073323475287c8e5aecc44c863e84 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dcc9f294 | sha256-8349c7fb6093e654a33bcd0632201835ade262cac01ac9ded36066d6039b36af |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-780ba765 | sha256-752f6fc75df4d02c7f61a88b236dd1db1dba5441e93b7014226ee17997a30ffa |
