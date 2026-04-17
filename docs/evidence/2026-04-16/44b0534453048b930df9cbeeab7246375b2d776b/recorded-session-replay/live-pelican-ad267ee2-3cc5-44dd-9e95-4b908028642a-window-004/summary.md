# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-9f6ebd737abaefd12c2c00ccde90878c5b6be8608902403a2490d096d9784e7b`
- bundle hash: `sha256-ee8b4d739898adbff1435cb3a4a71fc9e835768f61a0a97ffad2c48da2fc7729`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-394d692f2aa5412e9da10dfc0baf182beb2043f517fb99b07451a27af9201624 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b392cb8d473ecc6e84204a75f6835af9682e32b10b4d1b318b350839c3baeb47 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af9803608a199e74f714b0d9ce045818eb109e1e05cea65314a286cfe345162b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-92ecf7d3bc5ddb72796c32ab316fe8ade7d784a9b73538aa4eaa74cd3065bc45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3a99a951 | sha256-61a2569b393113cc43ad75362ea4777cbba8cf146041135a339b6b4890fdf1a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3a99a951 | sha256-c6debf9e1a5b3eff13aea3858a059dc9e08f2131a4d59016ed97de8a4fe7aac5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2fbd6fdc | sha256-cd49c794ba0bb748866ca12e5f62a4658bc1a572e1086f070eb08a05f95dcfa7 |
