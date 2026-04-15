# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e0cde3a25a3b093a3328111ec29f970fc82068378dbcdb19446f77be2e4c1e`
- fixture hash: `sha256-a657cfcc3c13a64972df27ba9b34b582252db2226ee691420ef45e3b6a2bad38`
- score hash: `sha256-44590bbeb4e5c7462805283c8221c1cc4ed4154eeebcfec2dd7894fc78d67dae`
- bundle hash: `sha256-f556dcbf7bc80ebab72f66ff57a07b88a895e8bc0e42d35e5b0da0135485c259`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0b3855bdbfdfdd31f2cf7aedce5b8a8e42a2e757ba398c67ba975aa86dd21ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-04d28f21c2e8dc7676985abd6d9e552b37eefab3db90fbf646595764eab58f7d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1cf4d5df61b3a544cd4af7d058d3d72918fe217b2be10f13f6505e0899401d0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ac156a8b0a9cc4a054aa18f174ed51328caed064808c0d28f56f078a30bcbc48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebb217df | sha256-828d2214698659e78961f4522b29ad0b6eb0dbab7703f1b7ce7ebed38acb0c6e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebb217df | sha256-76665dc5e7a285a0a2595a4ca7d34a1e791791aa47053cf6e931f51d43b90c1c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ebb217df | sha256-828d2214698659e78961f4522b29ad0b6eb0dbab7703f1b7ce7ebed38acb0c6e |
