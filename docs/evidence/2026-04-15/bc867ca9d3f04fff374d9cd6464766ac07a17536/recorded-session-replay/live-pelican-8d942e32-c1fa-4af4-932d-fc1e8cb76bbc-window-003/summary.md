# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9bc9ae6eca1afa6b83b6e75670188dd332e3c4f110971932dc47f6d6a315c1b4`
- fixture hash: `sha256-802793623b4c122b10687187b8ed29a08f9e42bf4ed06ad0911f576e3bb3e669`
- score hash: `sha256-2d4d1c321da061d45a4db0812e88cc80a9500b4f1c19d085332ada8bc117d4dc`
- bundle hash: `sha256-41e020f30fad48c762fc529830e27367188bf97e487296cfbfe4fd4e90bba6e9`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-59b577b3a75be68415d24692f624a4a3d4d68f36ae9d8d6779765be067821191 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d4574b630aaa244b49c486d6eb0acdca30b134e3bbbd288941c82225a039e768 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c9065c5f5f0e1959304a897ff690ebdad3172cf220b93cd6ef133c436f57c880 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8f9cda63 | sha256-d92faf0c210984357eb010ecde84be578a1e551059d65c0c23c603b1581a7a48 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8f9cda63 | sha256-29c4d4a77344029dea366356e006dece8245d724a5bdddce1b7c417d71da03fd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8f9cda63 | sha256-d92faf0c210984357eb010ecde84be578a1e551059d65c0c23c603b1581a7a48 |
