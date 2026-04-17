# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9bc9ae6eca1afa6b83b6e75670188dd332e3c4f110971932dc47f6d6a315c1b4`
- fixture hash: `sha256-802793623b4c122b10687187b8ed29a08f9e42bf4ed06ad0911f576e3bb3e669`
- score hash: `sha256-27f39138e2369eb819bbcc7f0322774533049b1b7b6b430489f181b69cd98b81`
- bundle hash: `sha256-91e298f55cb2596cdc758207731918d32996267a36adf2c44e82dfde86d6bf29`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c6cbf75055b2f2066145c893826e751aed7af61508ee55203c7ec8985a9cd38 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c694e826b54d17f0fda0a617b4b6e5be3cb324f0bde58cc6d5fee57c9d34a54 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0b9ac5423173098cdb7cc061fb90da79057ce594fbfb97a54b5f8a5fa4494959 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d87df53761eb1a283d8a14c7f53cdcb9cdc967ac7c4191d8a24d579d76d27682 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b4d900a4 | sha256-11a1d0384079cc1b045c7552121c53ac3378bf57609c5d2d66700e1fd6b27147 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b4d900a4 | sha256-f270d9e6585560666ff4f8adf11ad34034bbb738f5427a4a264d8a0dba6cb00a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e3bbf3ed | sha256-a18692f685afdb9a007351e73daa0c9a4cccc8bb518687e3738936c5dd8989a5 |
