# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5d4b7ac6ed69712b1588ada5d64482dda6216ae5bbb670a70c4e5011448ae050`
- fixture hash: `sha256-c583a0a30dc7272198329e0ce06b64ff4fe39dce1f96b56a4f82e04f4a924ee7`
- score hash: `sha256-4520d6dda05c1ae6ce99a4458c53e961df85f7c2943aa6e92937a772cdd42da3`
- bundle hash: `sha256-291a424cdf5557cd027f2298bc13378f90bd196557a691c835af3b7a86dda8e4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-646908dce1c2aa715ec563720c445a9dc7233e215511f30956abcb8a6c0f9113 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb4bb357de125f77ba5daf453760f924fa189c58cdf3070d60af34a41cc1bc98 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-442d30d727c2811fa5f5afcbaed07ef222dff3fd5c1d90f779a90f8f3d1c5790 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1cc178faa5dcc12b834e9eb41726cc0bc2141a1855a3d90a4fd388d1f258b112 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-123934ed | sha256-d3fdd9dfb2beacabdf754b37772fd2243f6d3f69ab0bafc7af7802ff7a951ba6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-123934ed | sha256-e64968070574fcecef3c28821a73eff8db94b2e18a7a51a9aba725959aabe915 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ffa65770 | sha256-15a46de6b5176e3ff2ce9712900954c374da5699f5de6023da6dba637c487238 |
