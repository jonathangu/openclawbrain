# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b29852ddcae763768818b925bfa513423dbca5c8ad934450c78f0838b90cfab`
- fixture hash: `sha256-f828a5ef63881667b78ea5f5530e5417bb5590176f57bdcf8c4590150136788a`
- score hash: `sha256-3c8ed76116de62847b16fb5ff2976506a017eafb7eae33cfdb3ef0ce616c8656`
- bundle hash: `sha256-22682fc4fc5a31ae7b810bdf6330f68b89c46904233d4e9bc9bc754a36f2bc8e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d23197d4b519cff22649347398dfab9ce049fcf294afab672a8b41fd8ebcbbad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-26aaddaf94ab6814611c18765703b1385129c6aeb78f0064613e404f27fa119a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f1695db2a454332d0c19215eeb4aa5386be9922eb557db74a6ffdcf402efef8c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3a3e07cab3bf19790fabe4a2693cb89774c68c1ea5ad90d8fc2f51e49ecd1945 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-adf348d0 | sha256-4a6390467c4d2916c3854daf259d316e666b90caf226740724f9d291614d3f72 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-adf348d0 | sha256-b8e0735c893a228957c1979ee98f5a86d146e33caf0b2e1072f1b261028fc4da |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6df6ab11 | sha256-46632c011d9f7e8230c9bd19d5ed4674be9a5646434ecece8b8ea0491a92f502 |
