# Recorded Session Replay Proof Bundle

- trace id: `live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ab9e7c92fecacf60147f27b8c27fc049edab247767d390d6cfd938c3433d0a10`
- fixture hash: `sha256-924c4ed1186166bf0f6b342b0967c241c06f6d79d3f88b2eb96f947a6b1061b7`
- score hash: `sha256-38db68af0c7b7248bc6df6cd2f9c21a17060c4309db6a8f61fff3bbd924c4d9a`
- bundle hash: `sha256-12a57086e35a040d3e0b5031e2b6104da93c0e7fff696f24fb09a594f20761ba`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af4d866d80b0149b53a0921726b1499466ad574098653e689171c2b2c56dcca1 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f8545749e82b975de48902e8fde959776424d533ea1fddf577dbd7ee8e6c6be1 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-e9a5cfc8c3c4af7460dd443a9b34adf267699363f98f21804ab1d22fa2df25eb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fa075e403f46883acb65bdc78ad0681a071713e98c1d41b441dd0f6d73866d4a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-8087f620 | sha256-54a85ef36467c5ff17dc3be99edecec0f556d74be2697c0a9280bdb9eb702837 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-8087f620 | sha256-da47a505a9b10ed4a1b800d7d146c8413dc07f5d2e0f42f512307589c85597ba |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8087f620 | sha256-2ab8e9644f13ddbf3be93a1c59e156934a8c380abea34d30654c13d77a8c6f04 |
