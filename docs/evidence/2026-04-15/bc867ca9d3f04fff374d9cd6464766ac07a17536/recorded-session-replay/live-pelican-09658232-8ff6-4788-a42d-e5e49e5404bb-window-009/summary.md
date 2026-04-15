# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1078b1a70bcd22daa0ead376beedaa52bfe2cf8765ec6a491cb29b47f4429da`
- fixture hash: `sha256-48416b4518f830c212c5a38183605df066ce4a1235bd3582b824c27bcab21c53`
- score hash: `sha256-f073931d8aa1b0bab2ab5cd70d87d040db30016a90800dd87611f0f1d93125d3`
- bundle hash: `sha256-42146f8b80b0b37ca16ec398c409cd2e4fe60426b53bbd1cc861238dbab1ad18`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57e6cc1ff0fcf88903029010179cd9e85affa629951b704a6bd53f2a38e4810e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-69305262c35ccfaf9dcc2090990f0d390e7848a24df86256a4286f6c68efb077 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-19ef32d8f1dd66c7edf522c879cbdcc2e89780c8d1fded0d228fd710f7c2209b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-54542adcc621958297441173f9d3abeffc164d1950b6e0fd942d4cbaf84aad48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d8e8d72c | sha256-4f7d10e5b71582d7361820de83cceca807f46c6f2c1568d196785a8849d41556 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d8e8d72c | sha256-4f7d10e5b71582d7361820de83cceca807f46c6f2c1568d196785a8849d41556 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d8e8d72c | sha256-4f7d10e5b71582d7361820de83cceca807f46c6f2c1568d196785a8849d41556 |
