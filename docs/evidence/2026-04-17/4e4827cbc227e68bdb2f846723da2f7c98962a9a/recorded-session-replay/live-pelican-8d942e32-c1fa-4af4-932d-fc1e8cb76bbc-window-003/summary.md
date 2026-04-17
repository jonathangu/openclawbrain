# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9bc9ae6eca1afa6b83b6e75670188dd332e3c4f110971932dc47f6d6a315c1b4`
- fixture hash: `sha256-802793623b4c122b10687187b8ed29a08f9e42bf4ed06ad0911f576e3bb3e669`
- score hash: `sha256-96b44db960039e5058d64499c1b177d7ac9b315e77a0b9e630d8e1144cae6081`
- bundle hash: `sha256-15ead4dbaaff9839395bbd391cc85b87fa8d40ade088cc255b2c36d9b5040887`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c6cbf75055b2f2066145c893826e751aed7af61508ee55203c7ec8985a9cd38 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-735dee7e580ab7f98a4dfcbd94415811d9b85bd3c0753a3e96d76addbd7a3d99 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c5de55e3df866b5f2909fe0bdd7fc680890709a8dd9ed3de3a4c58e8740b8e64 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6bf5f87499a60e29b9693c8e091b51db06e256af29c0b19c4334657eb453da30 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d4d86542 | sha256-ae2fabf47623744c53ef7b3a99bc5f282bb0b36c7ee81e179e9f0ad7961c42bf |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d4d86542 | sha256-96776789c8ae62ea2caf243c3180b8e136dadec77dea0fecb4e8bc4a948bf61b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-03bb588b | sha256-79cc291bd4301adcd76defe2cf614ebe36f5c397a6ddc027c0cb01358d61a480 |
