# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3188dea0835fbe3a5c0a4bde0dceb823e483b3aad858e66d490cbabd38ee5d72`
- fixture hash: `sha256-c551d2cb8b10201e8079d270837622ef96e1675624dead00decec0e3fb02a4b9`
- score hash: `sha256-ff609046ac24e4c966f61ece0d62cfad09086733b81100f399608cc4d899f08e`
- bundle hash: `sha256-68c8ca70174478beddeafb364229e2eea62264346eaeffcf4b18444e1b63d155`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7e263c626ee395c7a005cdac6d8c14b4d8e92d0d3065cdc0b98a11e431231d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6841d0d43b51f1af9ea04fe40e0802058f06d844354f9ac0cd497813f677c80 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b43067bb03e727b14337bb78f3fef66dd014ff5353eb9ce0c94e8e0254befea |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-90e02a8ce6ae77ae3f65ab39df8345c97863ca9bc2a51976f5e1677eb117e477 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5424b18d | sha256-b8b5437942c418fdf05ccf0ef93bc01bdca0ff09f69631bdb9944cd228674306 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5424b18d | sha256-d234198a60dd030d310bbc5c719f734e35f1f646ba8f02435b31b469660d7166 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8fea6f32 | sha256-4eeaea96c5bfe8eafbea70cbf9c00471bba5419389b4731e56e2d8168445ce36 |
