# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6225db554a4364f98319fe1020858f0210ac404609a5303b38b0c9b7d31d658`
- fixture hash: `sha256-ed619df46c192591981773c10de0d36b74c9e5dca7f78e3181e7b1aa8b066c66`
- score hash: `sha256-daece61a73e87360eb1c3faff0185e26ec34a09f805387fe9fe20ad05226f62d`
- bundle hash: `sha256-72bdcaddebd5af3fbbe66ae90bf479a19c89a75afd1b38e87bcab34b8baf655d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88c859d8e45ae0f9acc890628cb16be0104a115fdd6d7748c85da1fbfab29bae |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a539f88778ceff2ad075d936776ccd9f17775d4d6d8f3360a4a4ec8d2bb7ddce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8e7310be3f8b59c7632b000b5bfdfb82a59e6e242aff0c56908b60041f0faf46 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-68c7464f21b75902616797a404d25b35e50186d3da53590a3f0a4cea62175133 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83f27657 | sha256-38640604e094ab12aa0ea6555b3c7138f44766b3907edf563dd0e04a6df1f7b8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83f27657 | sha256-38640604e094ab12aa0ea6555b3c7138f44766b3907edf563dd0e04a6df1f7b8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-843d583e | sha256-39548bbf45791c02a135989d70932a4276d9f51d7f0dcb77e0b3980507fc7d2e |
