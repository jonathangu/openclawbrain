# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30a54314d984e83263bc7ddfcb852ce4d67a835461588938c047eabba74d7daa`
- fixture hash: `sha256-a669f6ac0947e4907b9b5ff0ba78d765904f903d2ac7c540eba1f40434878bd9`
- score hash: `sha256-1a737ccf95f7e4919411ffb87de220434820ce7f27fbe4a552083162778f3c9d`
- bundle hash: `sha256-7ba01d4d8d30d630058a9c69840445cce44a77d84463e86feeb1e581ff4ef3b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ddef39ad20fc1c3136dfb625c29bf78d555d4df3233592558f3107ec01752a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7e80dcfcd7185b2284325ce14ddfc2485bcc9f0047b11fe41eda52c1b8505e5c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cfbf926e65a131f5011ae4c2da96e521c34d0b282876f9533e748a39d548a06 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b6e9ecea75c8818b00a800817a5c76a671de41ccc8c5354e72ffa559720029ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd2f662e | sha256-a4600a6e70cc0c60ad51489a8c23c76bdd66aa345eead60cdc39bd9b235a2880 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cd2f662e | sha256-520b92fc34e766b9eab30c3267d0e40f872be25992ea3d2579fda0172a6c1b85 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d645169d | sha256-a6a6fe09e76e60c83afa3deb51056666069074ccb3d946a70cb165ecd4303e4f |
