# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bf37fe09634fbaed69f393758385015698c30e4ffe16d85a6ab728cb7cfe25b6`
- fixture hash: `sha256-dc83b67fd93a911909b6e6a0822040e20903fda7a3d9b344617db1a16b36190b`
- score hash: `sha256-3c2d0f4893b085f123251f7474ee104b30e7e19fa180fc2a9dd525d929fd7be1`
- bundle hash: `sha256-36c24f32e8f9d22bcecded6147fbf181a578b5c0744ee410310600318419acf5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6a3c5859c5aa675b38ed66866de5ac4f6b502c35d08a72874cf67deb2a63be26 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0461d9f914c59e449c92edfec1ea8199486bae7d9a5485179640972d6f3e2d4f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b74cd4ce356c81969e91fb57a36a89507975bb240d81abc70688073e5007920a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2fc441f3aed910b9376be3dc41a7369ffc46de3707864e70e8ec97ed2a3138a1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-25ff8add | sha256-bcc618dea6e1b7555827b4e7a550ecf0bb4a719fb45cd604b4f60563f707f85a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-25ff8add | sha256-6a9cca4482ee6b19db38b427a559614258502da76e468829e30d1e5338ca2735 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-25ff8add | sha256-bcc618dea6e1b7555827b4e7a550ecf0bb4a719fb45cd604b4f60563f707f85a |
