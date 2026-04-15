# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004`
- winner mode: `learned_route`
- trace hash: `sha256-414029967a4dfaeacf3048f9cc246c927617fc5206e50ba6c1c2944d9dd8d93c`
- fixture hash: `sha256-2f96d4d80b85de0482bdf816d900c02ecf0137642687879ce902112bb8056ccc`
- score hash: `sha256-dc7776c27c7444cfd5197dcd792b0a95e1eb63d95ffe2abed8262a44233e1616`
- bundle hash: `sha256-f4ed8fdce9331a4716d45addd35160f2973f3d983a89ce48eadc1c76d7290b5c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
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
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d02c67e07266eced41424ec8d8650df73f7c0173cd9e14609381c09dbbd89d1f |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc4250777154b5647576d858aa12f0498c4dc760c03f2d5a71ebbe736206842 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8db1d077364828fd22569a84553cf862eaa0313d217c7eb2eb2cde5b0eb6a6ba |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-76a412b76bb7240071bf30efab9ddb970594d5265d6dc1b218cc0c1cab5b2ec0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-cee3a843 | sha256-507ed4a3f1d6162c6c51331c154440bcc82a8aaf18b1c5912f2a62af32d5c6c0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cee3a843 | sha256-aff92df29d14c8feb0e9fbeea1f9050091a34a115ba53121f99cfe42b2a86cd4 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-cee3a843 | sha256-507ed4a3f1d6162c6c51331c154440bcc82a8aaf18b1c5912f2a62af32d5c6c0 |
